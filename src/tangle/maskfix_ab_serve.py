"""Loopback server for blind A/B mask reviews (packs from maskfix_ab_build.py, tangle_review.py).

  ../.venv/bin/python maskfix_ab_serve.py                                  ->  http://127.0.0.1:8769/  (reviews/maskfix-ab)
  ../.venv/bin/python maskfix_ab_serve.py --pack reviews/tangle-ab --port 8770
Verdicts are appended to <pack>/verdicts.jsonl (last write wins); key.json is never served.
"""
import argparse
import json
import time
from functools import partial
from http.server import ThreadingHTTPServer

from gold_serve import Handler as GoldHandler
from autolabel_common import ROOT

OUT = ROOT/'reviews/maskfix-ab'
VERDICTS = OUT/'verdicts.jsonl'
PAGE = __import__('pathlib').Path(__file__).with_name('maskfix_ab.html')


def latest():
    out = {}
    if VERDICTS.exists():
        for line in VERDICTS.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                out[r['id']] = r
    return out


class Handler(GoldHandler):
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            return self.send_bytes(PAGE.read_bytes(), 'text/html; charset=utf-8')
        if self.path == '/verdicts':
            return self.send_bytes(json.dumps(latest()).encode(), 'application/json')
        if self.path.startswith('/key'):
            return self.send_error(404)
        return super(GoldHandler, self).do_GET()

    def do_POST(self):
        if self.path != '/verdict':
            return self.send_error(404)
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        body['t'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        with VERDICTS.open('a') as f:
            f.write(json.dumps(body) + '\n')
        self.send_bytes(b'{"ok":true}', 'application/json')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--pack', default=str(OUT))
    ap.add_argument('--port', type=int, default=8769)
    ap.add_argument('--host', default='127.0.0.1', help='0.0.0.0 to review from another device on the LAN')
    a = ap.parse_args()
    OUT = ROOT/a.pack if not a.pack.startswith('/') else __import__('pathlib').Path(a.pack)
    VERDICTS = OUT/'verdicts.jsonl'
    server = ThreadingHTTPServer((a.host, a.port), partial(Handler, directory=str(OUT)))
    print(f'A/B review ({OUT.name}): http://{a.host}:{a.port}/', flush=True)
    server.serve_forever()
