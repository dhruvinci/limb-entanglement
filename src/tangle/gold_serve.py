"""Loopback server for the gold-set review (items from gold_build.py, verdicts appended to verdicts.jsonl).

  ../.venv/bin/python gold_serve.py      ->  http://127.0.0.1:8767/
GET  /items.json, /assets/*      review pack
GET  /verdicts                   latest verdict per item id
POST /verdict                    {"id", "type", "verdict", ...}  appended (last write wins)
GET  /stats                      accuracy per item type and stratum so far
"""
import argparse
import collections
import hashlib
import json
import time
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

from serve_review import RangeHandler

from paths import WORKSPACE as ROOT  # noqa: E402
GOLD = ROOT/'reviews/gold-v1'
VERDICTS = GOLD/'verdicts.jsonl'
INTERPRETED = GOLD/'interpreted.json'  # notes read by Claude: effective verdict + corrected labels (notes override keys)
PAGE = Path(__file__).with_name('gold_review.html')
OK = {'mask': lambda v: v['red'] == 'good' and v['blue'] == 'good', 'position': lambda v: v['verdict'] == 'correct',
      'identity': lambda v: v['verdict'] == 'same', 'event': lambda v: v['verdict'] == 'correct',
      'strike': lambda v: v['verdict'] == 'correct'}


def latest():
    out = {}
    if VERDICTS.exists():
        for line in VERDICTS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r['id']] = r
    return out


def effective():
    """Latest verdicts with note interpretations applied; also the ids whose note is new or changed since interpreted."""
    interp = json.loads(INTERPRETED.read_text()) if INTERPRETED.exists() else {}
    out, pending = {}, []
    for k, r in latest().items():
        note = r.get('note') or ''
        x = interp.get(str(k))
        if x and x['note_sha'] == hashlib.sha1(note.encode()).hexdigest()[:12]:
            r = {**r, **x['effective'], 'interpreted': True}
        elif note:
            pending.append(k)
        out[k] = r
    return out, pending


def stats():
    items = {i['id']: i for i in json.loads((GOLD/'items.json').read_text())}
    agg = collections.defaultdict(collections.Counter)
    rows, pending = effective()
    for r in rows.values():
        it = items.get(r['id'])
        if not it or r.get('verdict') in (None, 'unclear'):  # note-only records have no verdict yet
            continue
        for key in (it['type'], f"{it['type']}/{it['stratum']}"):
            agg[key]['n'] += 1
            agg[key]['ok'] += OK[it['type']](r)
            if it['type'] == 'mask':
                agg[key]['actor_n'] += 2
                agg[key]['actor_good'] += (r['red'] == 'good') + (r['blue'] == 'good')
                agg[key]['actor_wrong'] += (r['red'] == 'wrong') + (r['blue'] == 'wrong')
            if it['type'] == 'position':
                agg[key]['position_class_ok'] += r['verdict'] in ('correct', 'detail_wrong', 'top_wrong')
    out = {k: dict(v, acc=round(v['ok']/v['n'], 3)) for k, v in sorted(agg.items())}
    return {'notes_not_yet_interpreted': pending, **out}


class Handler(RangeHandler):
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            return self.send_bytes(PAGE.read_bytes(), 'text/html; charset=utf-8')
        if self.path == '/verdicts':
            return self.send_bytes(json.dumps(latest()).encode(), 'application/json')
        if self.path == '/stats':
            return self.send_bytes(json.dumps(stats(), indent=1).encode(), 'application/json')
        return super().do_GET()

    def do_POST(self):
        if self.path != '/verdict':
            return self.send_error(404)
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        body['t'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        with VERDICTS.open('a') as f:
            f.write(json.dumps(body) + '\n')
        self.send_bytes(b'{"ok":true}', 'application/json')

    def send_bytes(self, data, ctype):
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8767)
    a = ap.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', a.port), partial(Handler, directory=str(GOLD)))
    print(f'Gold review: http://127.0.0.1:{a.port}/', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
