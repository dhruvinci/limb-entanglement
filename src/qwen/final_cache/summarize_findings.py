"""Summarise the final cache campaign (2026-10-13) from its findings.json into the tables of E03's RESULTS.md.

Standard library only. Every number in E03's final-campaign tables comes from this script:

  python3 src/qwen/final_cache/summarize_findings.py experiments/E03-audits-final-cache/data/findings.json

findings.json holds one group per configuration (metrics averaged over source x seed cells), the localized
contact fixes/harms per frame, the temporal-control results and the campaign's own limitations.
Cohorts: 'armbar' = the one armbar clip (8 held-out frames of the same clip); 'breadth' = four other clips, each held
out whole in turn (clip-held-out). Actor IoU = mean of the two athletes' IoU over labelled cells.
"""
import json
import re
import sys
from collections import Counter, defaultdict
from statistics import mean


def load(path):
    with open(path) as f:
        return json.load(f)


def groups(d):
    out = []
    for g in d['groups']:
        c, m = g['configuration'], g['metrics']
        extra = {k: v for k, v in c.items() if k not in ('kind', 'cohort', 'representation', 'mode', 'control', 'epochs')}
        out.append({'kind': c.get('kind'), 'cohort': c.get('cohort'), 'rep': c.get('representation'),
                    'mode': c.get('mode'), 'control': c.get('control'), 'extra': extra, 'cells': g['source_seed_cells'],
                    **{k: (m.get(k) or {}).get('mean') for k in ('actor_iou', 'contact_accuracy', 'contact_margin')}})
    return out


def find(gs, **kw):
    hits = [g for g in gs if all(g.get(k) == v for k, v in kw.items())]
    if len(hits) != 1:
        raise KeyError(f'{kw}: {len(hits)} matches')
    return hits[0]


def f3(x):
    return '-' if x is None else f'{x:.3f}'


def layout_table(gs):
    rows, leg, cor, old, rep = [], [], [], [], []
    for layer in range(27):
        a = find(gs, kind='linear', cohort='armbar', rep=f'legacy_full{layer}', mode=None)['actor_iou']
        b = find(gs, kind='linear', cohort='armbar', rep=f'full{layer}', mode=None, extra={'evaluation': 'val_and_test'})['actor_iou']
        c = find(gs, kind='linear', cohort='armbar', rep=f'pooled{layer}', mode=None)['actor_iou']
        e = find(gs, kind='linear', cohort='armbar', rep=f'repool{layer}', mode=None)['actor_iou']
        leg.append(a); cor.append(b); old.append(c); rep.append(e)
        rows.append(f'| {layer} | {f3(a)} | {f3(b)} | {f3(c)} | {f3(e)} |')
    print('### Layout: the same linear probe on legacy vs corrected token order (armbar, 128-channel projection, 1 seed)')
    print('| vision layer | full grid, legacy reshape | full grid, corrected | pooled 30x30, legacy | pooled, re-pooled corrected |')
    print('|---|---|---|---|---|')
    print('\n'.join(rows))
    print(f'| **mean of 27** | **{mean(leg):.3f}** | **{mean(cor):.3f}** | **{mean(old):.3f}** | **{mean(rep):.3f}** |')
    print()


def representation_table(gs):
    print('### Linear readouts by input (actor IoU; armbar 3 seeds; held-out clips 4 clips x 3 seeds unless noted)')
    print('| input | armbar | held-out clips |')
    print('|---|---|---|')
    for rep, label in (('rgb', 'RGB only'), ('geometry', 'geometry only (actor centres)'), ('full11', 'Qwen layer 11 (corrected)'),
                       ('merged', 'Qwen merged embedding'), ('full11+merged', 'layer 11 + merged'),
                       ('full11@native', 'layer 11, all 1,152 channels (1 seed; clips: 4 cells)')):
        a = find(gs, kind='linear', cohort='armbar', rep=rep, mode=None, extra={})['actor_iou']
        b = find(gs, kind='linear', cohort='breadth', rep=rep, mode=None, extra={})['actor_iou']
        print(f'| {label} | {f3(a)} | {f3(b)} |')
    a = find(gs, kind='linear', cohort='armbar', rep='full11+merged', mode='random_labels')['actor_iou']
    b = find(gs, kind='linear', cohort='breadth', rep='full11+merged', mode='random_labels')['actor_iou']
    print(f'| control: layer 11 + merged, shuffled labels | {f3(a)} | {f3(b)} |')
    print()


def neural_table(gs):
    print('### Small neural decoders on layer 11 + merged (80 epochs)')
    print('| decoder | armbar actor IoU | armbar contact acc. | armbar contact margin | held-out clips actor IoU |')
    print('|---|---|---|---|---|')
    for mode, control, label in (('appearance', 'real', 'appearance only (no language states)'),
                                 ('global', 'real', 'global semantic fusion (real states)'),
                                 ('local', 'real', 'query-to-patch adapter, real states'),
                                 ('local', 'random', 'query-to-patch adapter, matched random states')):
        a = find(gs, kind='neural', cohort='armbar', rep='full11+merged', mode=mode, control=control)
        b = find(gs, kind='neural', cohort='breadth', rep='full11+merged', mode=mode, control=control, extra={})
        print(f'| {label} | {f3(a["actor_iou"])} | {f3(a["contact_accuracy"])} | {f3(a["contact_margin"])} | {f3(b["actor_iou"])} |')
    print()


def ablation_table(gs):
    print('### Ablations on held-out clips (appearance-only neural decoder on layer 11 + merged)')
    print('| setting | held-out clips actor IoU | contact accuracy |')
    print('|---|---|---|')
    rows = [('default (width 24, contact weight 1, all training frames)', dict(control='real', extra={})),
            ('width 12', dict(extra={'width': 12})), ('width 48', dict(extra={'width': 48})),
            ('contact weight 0', dict(extra={'contact_weight': 0.0})), ('contact weight 4', dict(extra={'contact_weight': 4.0})),
            ('25% of training frames', dict(extra={'fraction': 0.25})), ('50% of training frames', dict(extra={'fraction': 0.5}))]
    for label, kw in rows:
        g = find(gs, kind='neural', cohort='breadth', rep='full11+merged', mode='appearance', **kw)
        print(f'| {label} | {f3(g["actor_iou"])} | {f3(g["contact_accuracy"])} |')
    print()


def fixes_table(d):
    agg = defaultdict(Counter)
    for x in d['localized_fixes_and_harms']:
        held = 'armbar' if '-armbar-' in x['candidate'] else 'held-out clips'
        cand = re.sub(r'^\w+-[a-z_]+-full11_merged-\d+-', '', x['candidate'])
        if 'mode_appearance' not in x['baseline'] or 'eligibility' in x['baseline']:
            continue
        agg[(held, cand)]['fixed'] += x['contact_fixed']
        agg[(held, cand)]['harmed'] += x['contact_harmed']
    print('### Contact cells fixed / harmed against the appearance-only decoder (all frames, seeds, clips)')
    print('| cohort | candidate | contact cells fixed | contact cells harmed |')
    print('|---|---|---|---|')
    names = {'control_real-epochs_80-mode_local': 'query-to-patch, real states',
             'control_random-epochs_80-mode_local': 'query-to-patch, matched random',
             'control_real-epochs_80-mode_global': 'global fusion, real states'}
    for (held, cand), c in sorted(agg.items()):
        if cand in names:
            print(f'| {held} | {names[cand]} | {c["fixed"]} | {c["harmed"]} |')
    print()


def temporal_table(d):
    rows = defaultdict(list)
    for t in d['temporal']:
        rows['armbar' if t['heldout'] == 'armbar' else 'held-out clips'].append(t)
    print('### Temporal adapters: does frame order matter? (mean actor IoU over cells)')
    print('| cohort | cells | pass the 10% error-reduction screen | ordered | reversed | shuffled | best simple baseline |')
    print('|---|---|---|---|---|---|---|')
    for k, v in sorted(rows.items()):
        m = {s: mean(x[s]['actor_iou'] for x in v) for s in ('ordered', 'reversed', 'shuffled')}
        print(f'| {k} | {len(v)} | {sum(x["passes"] for x in v)} | {m["ordered"]:.3f} | {m["reversed"]:.3f} | '
              f'{m["shuffled"]:.3f} | {mean(x["strongest_simple_iou"] for x in v):.3f} |')
    print()


def paired_table(gs):
    print('### Matched DINO / Qwen readouts on actor crops (held-out clips, 12 cells)')
    print('| representation | actor IoU | contact accuracy |')
    print('|---|---|---|')
    for rep in ('dino', 'qwen', 'combined', 'gated'):
        g = find(gs, kind='paired', rep=rep)
        print(f'| {rep} | {f3(g["actor_iou"])} | {f3(g["contact_accuracy"])} |')
    print()


def main():
    d = load(sys.argv[1] if len(sys.argv) > 1 else 'findings.json')
    gs = groups(d)
    print(f'Groups: {len(gs)} ({dict(Counter(g["kind"] for g in gs))}); temporal cells: {len(d["temporal"])}.\n')
    layout_table(gs)
    representation_table(gs)
    neural_table(gs)
    ablation_table(gs)
    fixes_table(d)
    temporal_table(d)
    paired_table(gs)
    print('### Limitations recorded by the campaign')
    for x in d['limitations']:
        print(f'- {x}')


if __name__ == '__main__':
    main()
