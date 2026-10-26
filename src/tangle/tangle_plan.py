"""Plan a tangle-tracking span for one interval (CPU): where the primary pair is last/first separated around the window.

Input: a 2 fps scan of SAM 3.1 person detections over [start - W, end + W] plus ffmpeg scene cuts (tangle_scan.py).
The pair is followed outward from the window edge with a box association that tolerates merges (one detection
covering both fighters is assigned to both). An anchor is the nearest frame, in the same shot as the window edge,
where the two fighters are distinct confident detections whose boxes do not touch. Identity at the anchor is not
trusted here (it is resolved after tracking by voting against the window labels).
"""
import json
from pathlib import Path

import numpy as np

from autolabel_common import ROOT, box_iou, rle_decode

SCORE_SEP = 0.7     # both fighters' detections at a separated anchor
SEP_IOU = 0.0       # boxes must not overlap at all
MIN_AREA_FRAC = 0.25  # a candidate detection must be at least this share of the actor's previous area
LOST_MAX = 6        # 3 s without a matching detection ends the walk


def pair_boxes(frame, prim):
    out = {}
    for i in frame['instances']:
        if i['actor_id'] in prim and i['area']:
            out[i['actor_id']] = (list(i['box_xyxy']), i['area'])
    return out


def walk(rows, cuts, edge_t, start_boxes, direction):
    """Follow the pair from the window edge outward. rows: scan rows sorted by time. Returns the anchor row
    (dict with t, boxes of the two fighters) or None, plus the reason the walk ended."""
    rows = [r for r in rows if (r['t'] < edge_t if direction < 0 else r['t'] > edge_t)]
    rows = rows[::-1] if direction < 0 else rows
    cur = dict(start_boxes)
    lost = {a: 0 for a in cur}
    for r in rows:
        if any((r['t'] <= c <= edge_t) if direction < 0 else (edge_t <= c <= r['t']) for c in cuts):
            return None, 'shot_cut'
        dets = [d for d in r['dets'] if d['score'] >= 0.4]
        match = {}
        for a, (box, area) in cur.items():
            cands = [(box_iou(d['box'], box), j) for j, d in enumerate(dets) if d['area'] >= MIN_AREA_FRAC*area]
            best = max(cands, default=(0, None))
            if best[0] > 0.1:
                match[a] = best[1]
                lost[a] = 0
            else:
                lost[a] += 1
        if any(v > LOST_MAX for v in lost.values()):
            return None, 'lost'
        for a, j in match.items():
            cur[a] = (dets[j]['box'], dets[j]['area'])
        if len(match) == 2 and len(set(match.values())) == 2:
            a, b = (dets[j] for j in match.values())
            if min(a['score'], b['score']) >= SCORE_SEP and box_iou(a['box'], b['box']) <= SEP_IOU:
                return {'t': r['t'], 'boxes': {k: dets[j]['box'] for k, j in match.items()}}, 'separated'
    return None, 'span_end'


def choose_pair(L):
    """The two athletes/instructors that dominate the clip: largest summed area over frames where both are visible,
    weighted up when their boxes touch. Astra's interval-level primary flag picked a background athlete in open-mat
    footage (a2s278: P2+P3 flagged, the closed guard in the foreground is P1+P2)."""
    roles = {a['actor_id']: a['role'] for a in L['actors']}
    ok = {a for a, r in roles.items() if r in ('athlete', 'instructor')}
    score = {}
    for f in L['frames']:
        ins = sorted((i for i in f['instances'] if i['actor_id'] in ok and i['area']), key=lambda i: -i['area'])[:5]
        for x, a in enumerate(ins):
            for b in ins[x + 1:]:
                key = tuple(sorted((a['actor_id'], b['actor_id'])))
                touch = box_iou(a['box_xyxy'], b['box_xyxy']) > 0.02
                score[key] = score.get(key, 0) + (a['area'] + b['area'])*(1.0 if touch else 0.5)
    flagged = sorted(a['actor_id'] for a in L['actors'] if a.get('primary'))[:2]
    if not score:
        return flagged, False
    best = list(max(score, key=score.get))
    if len(flagged) == 2 and score.get(tuple(flagged), 0) >= 0.8*score[tuple(best)]:
        return flagged, False
    return best, True


def plan(iv, scan):
    """{'entry': anchor or None, 'exit': anchor or None, reasons, window cuts}."""
    base = ROOT/'autolabel/v1'/iv
    L = json.loads((base/'labels.json').read_text())
    prim, pair_fixed = choose_pair(L)
    s, e = L['time_range_s']
    rows = sorted(scan['rows'], key=lambda r: r['t'])

    def edge_boxes(frames):
        for f in frames:
            pb = pair_boxes(f, prim)
            if len(pb) == 2:
                return f['time_s'], pb
        return None, None
    t_in, b_in = edge_boxes(L['frames'])
    t_out, b_out = edge_boxes(L['frames'][::-1])
    entry, why_in = walk(rows, scan['cuts'], t_in, b_in, -1) if b_in else (None, 'no_pair')
    exit_, why_out = walk(rows, scan['cuts'], t_out, b_out, +1) if b_out else (None, 'no_pair')
    inside = [c for c in scan['cuts'] if s < c < e]
    return {'iv': iv, 'prim': prim, 'pair_fixed': pair_fixed, 'window': [s, e], 'entry': entry, 'entry_reason': why_in,
            'exit': exit_, 'exit_reason': why_out, 'window_cuts': inside}


if __name__ == '__main__':
    import sys
    scan_dir = Path(sys.argv[1])
    for f in sorted(scan_dir.glob('*.json')):
        sc = json.loads(f.read_text())
        p = plan(sc['iv'], sc)
        s, e = p['window']
        print(f"{p['iv']:22s} entry {('%.1fs before' % (s - p['entry']['t'])) if p['entry'] else p['entry_reason']:>14s}  "
              f"exit {('%.1fs after' % (p['exit']['t'] - e)) if p['exit'] else p['exit_reason']:>14s}  cuts in window {len(p['window_cuts'])}")
