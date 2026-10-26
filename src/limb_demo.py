"""Figures for the limb-entanglement READMEs (matplotlib + OpenCV; CPU only).

  python src/limb_demo.py --out docs/assets [--h4d-work DIR]

Writes:
  h4d_tangle_example.jpg     one Harmony4D frame (MIT): photo, ground-truth ownership, the old tracker prompting and
                             the tangle tracker (needs the benchmark work folder of src/tangle/tangle_bench.py)
  chart_*.png                result charts; every value is copied from an experiment's RESULTS.md (line noted)
  layout_illustration.png    the token-layout bug on a small synthetic grid (no data needed)
No frame of any third-party, broadcast or user video is drawn. Colours: blue / orange / aqua, grey for baselines.
"""
import argparse
from pathlib import Path

import numpy as np

BLUE, ORANGE, AQUA = '#2a78d6', '#eb6834', '#1baf7a'
GREY, INK, INK2, SURF, GRID = '#9a9993', '#0b0b0b', '#52514e', '#fcfcfb', '#e6e5e0'


def plt_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9.5, 'axes.edgecolor': GREY, 'axes.labelcolor': INK2, 'xtick.color': INK2,
                         'ytick.color': INK2, 'axes.facecolor': SURF, 'figure.facecolor': SURF, 'axes.grid': True,
                         'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
                         'axes.spines.right': False, 'text.color': INK, 'legend.frameon': False,
                         'axes.axisbelow': True})
    return plt


def save(plt, fig, path, dpi=130):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, facecolor=SURF)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- Harmony4D example
def h4d_example(work, out, seq='032_grappling2_cam01', frame=46, crop=(380, 60, 1040, 700)):
    """Photo | truth | old prompting (points, forward) | tangle tracker v4 core (masks, both fighters, fwd + bwd).

    work: the tangle_bench.py work folder (frames/, gt.npz, pred_points_fwd.npz, pred_mask_nonov_bidir.npz).
    Frame 46 of 032_grappling2 cam01 is one of the largest old-vs-v4 differences on the benchmark (picked for
    illustration); the benchmark-wide numbers are in experiments/E04-tangle-tracker/RESULTS.md:45-48.
    """
    import cv2
    plt = plt_setup()
    v = Path(work)/seq
    img = cv2.cvtColor(cv2.imread(str(v/'frames'/f'{frame:05d}.jpg')), cv2.COLOR_BGR2RGB)
    gt = np.load(v/'gt.npz')['gt'][frame]
    old = np.load(v/'pred_points_fwd.npz')['labels'][frame//2]       # tracked at stride 2
    new = np.load(v/'pred_mask_nonov_bidir.npz')['labels'][frame//2]
    cols = {1: np.array([0x2a, 0x78, 0xd6]), 2: np.array([0xeb, 0x68, 0x34])}

    def overlay(lab):
        o = img.astype(float)
        for k, c in cols.items():
            m = lab == k
            o[m] = 0.4*o[m] + 0.6*c
        return o.astype(np.uint8)

    body = gt > 0

    def score(lab):
        right = ((lab == gt) & body).sum()/body.sum()
        wrong = ((lab > 0) & (lab != gt) & body).sum()/body.sum()
        return f'{right:.0%} of body pixels right, {wrong:.0%} wrong athlete'

    x0, y0, x1, y1 = crop
    panels = [(img, 'photo (Harmony4D)'), (overlay(gt), 'truth (rendered SMPL bodies)'),
              (overlay(old), 'point prompts, forward\n' + score(old)),
              (overlay(new), 'tangle tracker (full-mask anchors)\n' + score(new))]
    fig, axs = plt.subplots(1, 4, figsize=(12, 3.6))
    for ax, (im, title) in zip(axs, panels):
        ax.imshow(im[y0:y1, x0:x1])
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    fig.suptitle('Whose limb is it? Same tracker, same seeds: point prompts let one athlete swallow the other',
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out/'h4d_tangle_example.jpg', dpi=105, facecolor=SURF, pil_kwargs={'quality': 85})
    plt.close(fig)


# ----------------------------------------------------------------------------------------------- charts
def chart_tool_chain(plt, out):
    """E00: where the tool chain broke (experiments/E00-tool-chain/RESULTS.md, lines noted in the boxes' sources)."""
    from matplotlib.patches import FancyBboxPatch
    stages = [('Frontier VLM\n(GPT-5.6 Sol)', 'identity kept, armbar named;\nbut its arm point lands\non the torso, not the arm', True),
              ('SAM 3 / 3.1\n(boxes + points)', 'trapped forearm masked\nas the wrong athlete; no\ncandidate covers all of it', False),
              ('PromptHMR\n(single frame)', 'can pull the arm to the\nright athlete (42.9 px), but\nbodies collapse at 3.5 s', False),
              ('PromptHMR-Video\n(temporal)', 'smooth (bone variation\n0.180 -> 0.010), arm error\n216 -> 255 px (worse)', False)]
    fig, ax = plt.subplots(figsize=(11, 2.9))
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 1)
    ax.axis('off')
    for i, (name, note, ok) in enumerate(stages):
        c = AQUA if ok else ORANGE
        ax.add_patch(FancyBboxPatch((i + 0.06, 0.08), 0.82, 0.84, boxstyle='round,pad=0.01,rounding_size=0.04',
                                    fc=SURF, ec=c, lw=2))
        ax.text(i + 0.47, 0.78, name, ha='center', va='center', fontsize=9.5, weight='bold')
        ax.text(i + 0.47, 0.4, note, ha='center', va='center', fontsize=8.5, color=INK2)
        if i < 3:
            ax.annotate('', (i + 1.05, 0.5), (i + 0.9, 0.5), arrowprops=dict(arrowstyle='->', color=GREY, lw=1.5))
    ax.set_title('One armbar clip, four tools in a chain: each hand-off loses whose arm it is', fontsize=11)
    save(plt, fig, out/'chart_tool_chain.png')


def chart_fixed_decoder(plt, out):
    """E01: one trained decoder, only the actor states swapped (E01 RESULTS.md:158-165)."""
    rows = [('action, 4 fps', 0.6659), ('identity, 4 fps', 0.6655), ('contact, 4 fps', 0.6648), ('action, thinking xhigh', 0.6651),
            ('action, reversed frames', 0.6651), ('action, shuffled frames', 0.6651), ('matched random code', 0.6639),
            ('A1 and A2 swapped', 0.0658)]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    y = np.arange(len(rows))[::-1]
    colours = [BLUE]*6 + [GREY, ORANGE]
    ax.barh(y, [v for _, v in rows], height=0.6, color=colours)
    for yy, (_, v) in zip(y, rows):
        ax.text(v + 0.01, yy, f'{v:.4f}', va='center', fontsize=8, color=INK2)
    ax.set_yticks(y, [n for n, _ in rows])
    ax.set_xlim(0, 0.8)
    ax.set_xlabel('athlete IoU on held-out frames (same decoder weights throughout)')
    ax.set_title('Six different Qwen actor states, one answer: an actor code', fontsize=11)
    save(plt, fig, out/'chart_fixed_decoder.png')


def chart_trapped_arm(plt, out):
    """E02: trapped-arm ownership margin (E02 RESULTS.md:96-100). Positive = right athlete."""
    rows = [('vision only (static decoder)', -0.7882, GREY), ('+ Qwen action states', -0.9789, BLUE),
            ('+ norm-matched random states', -0.9708, GREY), ('+ identity states (seed 7)', -0.9940, BLUE),
            ('+ contact states (seed 7)', -0.6107, BLUE)]
    fig, ax = plt.subplots(figsize=(7, 3.0))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], height=0.6, color=[r[2] for r in rows])
    for yy, r in zip(y, rows):
        ax.text(r[1] - 0.02, yy, f'{r[1]:+.2f}', va='center', ha='right', fontsize=8, color=INK2)
    ax.axvline(0, color=INK2, lw=1)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(-1.3, 1)
    ax.set_xlabel('trapped-arm margin, P(true owner) - P(other athlete), mean of 27 cells\n(> 0: right athlete; < 0: wrong athlete)')
    ax.set_title('Every decoder gives the trapped arm to the wrong athlete', fontsize=11)
    save(plt, fig, out/'chart_trapped_arm.png')


def chart_layout(plt, out):
    """E03: synthetic layout illustration + per-layer probe IoU, legacy vs corrected order (E03 RESULTS.md, table
    'Layout', generated from data/findings.json)."""
    legacy = [0.270, 0.210, 0.193, 0.188, 0.190, 0.185, 0.209, 0.215, 0.232, 0.236, 0.246, 0.248, 0.238, 0.223, 0.215,
              0.208, 0.207, 0.222, 0.211, 0.193, 0.202, 0.202, 0.190, 0.178, 0.201, 0.203, 0.159]
    fixed = [0.410, 0.357, 0.353, 0.353, 0.360, 0.385, 0.413, 0.438, 0.459, 0.485, 0.507, 0.506, 0.502, 0.518, 0.532,
             0.498, 0.508, 0.491, 0.505, 0.493, 0.470, 0.501, 0.482, 0.493, 0.459, 0.474, 0.413]
    h, w = 16, 24                               # a small token grid with a figure drawn on it
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    shape = ((yy - 7.5)**2/36 + (xx - 7)**2/30 < 1) & ((yy - 7.5)**2/12 + (xx - 7)**2/10 > 1)   # a ring
    shape |= (np.abs(yy - 12) <= 1) & (xx >= 13) & (xx <= 21)                                # a bar
    shape |= (np.abs(xx - 18) <= 1) & (yy >= 2) & (yy <= 11)                                 # a post
    img = np.ones((h, w, 3))*np.array([0.988, 0.988, 0.984])
    img[shape] = np.array([0x2a, 0x78, 0xd6])/255
    grouped = img.reshape(h//2, 2, w//2, 2, 3).transpose(0, 2, 1, 3, 4).reshape(-1, 3)   # processor token order
    legacy_img = grouped.reshape(h, w, 3)                                                # the old direct reshape
    fig = plt.figure(figsize=(11, 3.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 2.2])
    for k, (im, t) in enumerate(((img, 'image order'), (legacy_img, 'what the direct reshape built'))):
        ax = fig.add_subplot(gs[0, k])
        ax.imshow(im, interpolation='nearest')
        ax.set_title(t, fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
    ax = fig.add_subplot(gs[0, 2])
    x = np.arange(27)
    ax.plot(x, fixed, color=BLUE, lw=2, marker='o', ms=4, label='corrected order')
    ax.plot(x, legacy, color=GREY, lw=2, marker='o', ms=4, label='legacy reshape')
    ax.text(26.3, fixed[-1], 'corrected', va='center', fontsize=8, color=INK2)
    ax.text(26.3, legacy[-1], 'legacy', va='center', fontsize=8, color=INK2)
    ax.set_xlim(-0.5, 29.5)
    ax.set_ylim(0, 0.6)
    ax.set_xlabel('Qwen3.8 vision layer')
    ax.set_ylabel('athlete IoU (same linear probe)')
    ax.legend(loc='lower center', ncol=2, fontsize=8)
    ax.set_title('Same probe, same features, tokens put back in place', fontsize=9.5)
    fig.suptitle('The token-layout bug: 14,080 of 14,400 grid locations misplaced', fontsize=11)
    save(plt, fig, out/'chart_layout.png')


def chart_blind_ab(plt, out):
    """E04 + E05: blind A/B verdicts by the reviewer (decisive verdicts only).
    E04 RESULTS.md:67 and :126; E05 RESULTS.md:41, :115, :199."""
    rows = [('tracker vs production masks\n(28 dev clips)', 13, 2), ('tracker vs production masks\n(12 test clips)', 7, 4),
            ('dot corrections, narrow\nvs tracker', 12, 5), ('dot corrections, wide\nvs tracker', 21, 0),
            ('+ round 2, memory, bone fill\nvs wide corrections', 19, 6), ('+ round 3 corrections\nvs round 2', 28, 6)]
    fig, ax = plt.subplots(figsize=(7.6, 4.1))
    y = np.arange(len(rows))[::-1]
    ax.barh(y + 0.17, [r[1] for r in rows], height=0.32, color=BLUE, label='newer method preferred')
    ax.barh(y - 0.17, [r[2] for r in rows], height=0.32, color=GREY, label='older method preferred')
    for yy, r in zip(y, rows):
        ax.text(r[1] + 0.4, yy + 0.17, str(r[1]), va='center', fontsize=8, color=INK2)
        ax.text(r[2] + 0.4, yy - 0.17, str(r[2]), va='center', fontsize=8, color=INK2)
    ax.axhline(4.5, color=GRID, lw=1)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.set_xlim(0, 32)
    ax.set_xlabel('frames (blind, one reviewer; ties and "both bad" not shown)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title('Blind A/B by one reviewer: every step won,\nbut the last four steps were fed by that reviewer\'s own dots', fontsize=10.5)
    save(plt, fig, out/'chart_blind_ab.png')


def chart_dots(plt, out):
    """E05: dots right near a dotted frame vs on random frames.
    Near: held-out final-pack dots, v4w2mk 129/172 -> v4w3mk 155/172 (E05 RESULTS.md:157,160).
    Random: fresh random-frame dots (153), v4 69% -> v4w2mk 72% (E05 RESULTS.md:122,125)."""
    groups = ['10 held-out frames in the clips\nof the new corrections (172 dots)', 'random frames nobody had\ndotted (153 dots)']
    before, after = [129/172*100, 69], [155/172*100, 72]
    labels_b, labels_a = ['round 2 (75%)', 'tracker (69%)'], ['round 3 (90%)', 'round 2 stack (72%)']
    fig, ax = plt.subplots(figsize=(7, 3.5))
    y = np.arange(2)[::-1]
    ax.barh(y + 0.17, before, height=0.3, color=GREY, label='before')
    ax.barh(y - 0.17, after, height=0.3, color=BLUE, label='after more dot corrections')
    for yy, b, a, lb, la in zip(y, before, after, labels_b, labels_a):
        ax.text(b + 1, yy + 0.17, lb, va='center', fontsize=8, color=INK2)
        ax.text(a + 1, yy - 0.17, la, va='center', fontsize=8, color=INK2)
    ax.set_yticks(y, groups, fontsize=8.5)
    ax.set_xlim(0, 115)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('reviewer dots on the right athlete (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
    ax.set_title('Corrections carry a few frames, not to the rest of the video', fontsize=11)
    save(plt, fig, out/'chart_dots.png')


def chart_other_cues(plt, out):
    """E06: Sapiens resolution floor (E06 RESULTS.md, section 1 table) and SAM 3D Body depth order on the leg pack
    (section 2 table: v4 261/54/47, s3dmf 280/69/13)."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.3), gridspec_kw={'width_ratios': [1.1, 1]})
    a1.axvspan(150, 400, color=ORANGE, alpha=0.12, lw=0)
    a1.text(275, 50, 'athletes in our\nfootage', ha='center', fontsize=8, color=INK2)
    a1.plot([300, 512, 1024], [0, 0.1, 53.9], color=BLUE, lw=0, marker='o', ms=7)   # measured points only
    for x, v in ((300, 0), (512, 0.1), (1024, 53.9)):
        a1.text(x, v + 3, f'{v:g}%', ha='center', fontsize=8, color=INK2)
    a1.set_xlim(100, 1100)
    a1.set_ylim(0, 62)
    a1.set_xlabel('image height in px (the person fills most of it)')
    a1.set_ylabel('pixels Sapiens labels human (%)')
    a1.set_title('Sapiens segmentation: nothing below ~500 px', fontsize=10)
    names = ['tracker (v4)', 'SAM 3D Body\ndepth order']
    right, swap, hole = np.array([261, 280]), np.array([54, 69]), np.array([47, 13])
    x = np.arange(2)
    a2.bar(x, right, 0.5, color=BLUE, label='right')
    a2.bar(x, swap, 0.5, bottom=right + 2, color=ORANGE, label='wrong athlete')
    a2.bar(x, hole, 0.5, bottom=right + swap + 4, color=GREY, label='nobody')
    for xx, r, s in zip(x, right, swap):
        a2.text(xx, r/2, str(r), ha='center', fontsize=8.5, color='white')
        a2.text(xx, r + s/2, str(s), ha='center', fontsize=8.5, color=INK)
    a2.set_xticks(x, names)
    a2.set_ylabel('leg-pack dots (362)')
    a2.legend(loc='upper center', ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    a2.set_ylim(0, 420)
    a2.set_title('3D bodies, nearer owns the pixel: more right, more swaps', fontsize=10)
    save(plt, fig, out/'chart_other_cues.png')


def chart_timeline(plt, out):
    """E07: the thread's timeline (dates from each experiment's RESULTS.md)."""
    ev = [('2026-08-26', 'E00 tool chain on one armbar clip', ORANGE), ('2026-08-27', 'E01 Qwen3.8 cache + ownership probe', ORANGE),
          ('2026-09-05', 'E02 decoders: 0/27 trapped-arm cells', ORANGE), ('2026-09-21', 'E03 provenance audit', GREY),
          ('2026-10-12', 'E04 tangle tracker (blind 13-2)', BLUE), ('2026-10-13', 'E03 layout audit + final cache run', GREY),
          ('2026-10-14', 'E05 dot rounds (blind 28-6) / E06 cues', BLUE), ('2026-10-22', 'paused: dots improved no model', GREY)]
    import datetime as dt
    d = [dt.date.fromisoformat(e[0]) for e in ev]
    t = np.array([(x - d[0]).days for x in d])
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.axhline(0, color=GREY, lw=1)
    lev = [0.9, -0.9, 0.6, -0.6, 0.9, -0.9, 0.55, -0.55]
    for (date, text, c), x, l in zip(ev, t, lev):
        ax.plot([x, x], [0, l*0.85], color=GRID, lw=1)
        ax.plot(x, 0, 'o', color=c, ms=8, mec=SURF, mew=2)
        ax.text(x, l, f'{date[5:]}  {text}', ha='center', va='center', fontsize=8, color=INK)
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(-8, t[-1] + 8)
    ax.axis('off')
    ax.set_title('Whose arm is it? Aug-Oct 2026 (orange: VLM route, blue: tracking route, grey: audits/decision)',
                 fontsize=10)
    save(plt, fig, out/'chart_timeline.png')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='docs/assets')
    ap.add_argument('--h4d-work', help='tangle_bench.py work folder; skip the Harmony4D panel if absent')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plt = plt_setup()
    for f in (chart_tool_chain, chart_fixed_decoder, chart_trapped_arm, chart_layout, chart_blind_ab, chart_dots,
              chart_other_cues, chart_timeline):
        f(plt, out)
    if a.h4d_work:
        h4d_example(a.h4d_work, out)


if __name__ == '__main__':
    main()
