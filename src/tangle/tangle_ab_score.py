"""Score blind verdicts of a tangle A/B pack by question: which method won (last write per item; notes printed so they
can override keys). Usage: tangle_ab_score.py <pack dir>"""
import json
import sys
from collections import defaultdict
from pathlib import Path

pack = Path(sys.argv[1])
key = json.loads((pack/'key.json').read_text())
last = {}
for line in (pack/'verdicts.jsonl').read_text().splitlines():
    if line.strip():
        v = json.loads(line)
        last[str(v['id'])] = v
tally = defaultdict(lambda: defaultdict(int))
for i, v in sorted(last.items(), key=lambda x: (not x[0].isdigit(), x[0].zfill(5))):
    if i not in key:
        print('video note', i, '|', v.get('note'))
        continue
    k, verdict = key[i], v.get('verdict')
    win = k.get(verdict, verdict) if verdict in ('A', 'B') else verdict
    tally[k['question']][win] += 1
    print(i, k['question'], k['src'], f"A={k['A']} B={k['B']}", '|', verdict, '->', win, '|', v.get('note') or '', f"[{len(v['dots'])} dots]" if v.get('dots') else '')
for q, t in tally.items():
    print(q, dict(t))
