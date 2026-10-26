"""Per-clip QA table for a tangle variant (CPU): shots, anchors, identity hand-offs, coverage.

  ../.venv-pose/bin/python tangle_summary.py --variant v2 <trial/interval ...>
A hand-off is a pass that ends on another anchor: its masks there are compared with that anchor's masks. 'ok' when
both fighters match better than swapped; 'swap' when the swapped assignment matches better (an identity conflict
between the two anchors, or a swap during the pass); 'lost' when both IoUs are below 0.3.
"""
import argparse
import json

from autolabel_common import ROOT


def handoff_state(h):
    keep, swap = sum(h['iou']), sum(h['swapped_iou'])
    if max(h['iou'] + h['swapped_iou']) < 0.3:
        return 'lost'
    return 'ok' if keep >= swap else 'swap'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', default='v2')
    ap.add_argument('targets', nargs='+')
    a = ap.parse_args()
    tot = {'ok': 0, 'swap': 0, 'lost': 0}
    print(f"{'clip':22s} {'frames':>6s} {'shots':>5s} {'cuts':>4s} {'anchors':>7s} {'lab cov':>7s} {'handoff ok/swap/lost':>21s} {'min iou':>7s}")
    for iv in a.targets:
        f = ROOT/'autolabel/v1'/iv/'tangle'/a.variant/'tangle.json'
        if not f.exists():
            print(f'{iv:22s} (no output)')
            continue
        T = json.loads(f.read_text())
        anchors = sorted({p['k'] for p in T['passes']})
        st = [h.get('state') or handoff_state(h) for h in T.get('handoffs', [])]
        st = ['ok' if x == 'weak' else x for x in st]
        for s in st:
            tot[s] += 1
        cov = len(T['label_masks'])/max(1, len(T['label_to_span']))
        mins = [min(h['iou']) for h in T.get('handoffs', [])]
        print(f"{iv:22s} {len(T['times']):6d} {len(T['shots']):5d} {len(T.get('frame_cuts', [])):4d} {len(anchors):7d} {cov:7.0%} "
              f"{st.count('ok'):>11d}/{st.count('swap')}/{st.count('lost')}{'':5s} {min(mins) if mins else float('nan'):7.2f}")
    print('hand-offs', tot)


if __name__ == '__main__':
    main()
