# E04 — The tangle tracker (2026-10-12)

**Objective.** Single-frame models merge two interlocked grapplers into one person ([RESULTS.md:15](RESULTS.md#L15)).
Ownership is clear when they are apart and is conserved through a tangle. Can a video tracker carry it through?

**Setup.**
- SAM 3.1 multiplex tracker, frozen. Anchors = frames where ownership is trusted (the two apart, or labels that
  agree with the video masks). Both athletes seeded with **full masks** in one session, tracked forward and backward
  between anchors, fused per pixel ([:19-25](RESULTS.md#L19)).
- Benchmark: Harmony4D grappling (MIT), 56 videos, 5,500 frames at 10 fps; truth = both SMPL bodies rendered, nearer
  owns the pixel ([:39](RESULTS.md#L39)). Then blind A/B by Dhruva on 28 production clips (public online videos).

![Harmony4D: point prompts vs the tangle tracker](../../docs/assets/h4d_tangle_example.jpg)

**Results** (Harmony4D, frames where the bodies touch)
| tracker variant | right owner | wrong owner |
|---|---|---|
| point prompts, forward (before) | 0.845 | 0.056 |
| full-mask prompts, forward | 0.880 | 0.054 |
| + pixel competition, forward + backward (**v4**) | **0.882** | **0.053** |
Source: [:44-47](RESULTS.md#L44).
- v4 beats the old prompting on 55 of 56 videos ([:50](RESULTS.md#L50)); re-anchoring from the truth every second adds
  only 1.7 points more ([:90](RESULTS.md#L90)).
- Blind A/B on production clips: **13-2** for v4 (sign test p ~ 0.007), 8-0 where the earlier mask fix had failed
  ([:67-70](RESULTS.md#L67)); 7-4 on 12 test clips picked in advance ([:126](RESULTS.md#L126)).
- Negative: skeleton anchors, predicted-IoU fusion weights, zoom and BBoxMaskPose-guided masks all lost or tied
  ([:94](RESULTS.md#L94), [:127](RESULTS.md#L127), [:148-151](RESULTS.md#L148)).

**Verdict.** **Pass, modest.** The best mask ownership of the thread, and it held up blind. Wrong-owner pixels barely
moved (5.6 -> 5.3%), so the gain is mostly body pixels no longer missed, not swaps fixed ([:52](RESULTS.md#L52)).
The image shows one of the largest differences on the benchmark, not a typical frame.
**Learned.** Seed the tracker with whole masks, never points. Check a directly driven tracker by swapping its seeds:
it silently ignored them for half a day ([:32](RESULTS.md#L32)).
**Next.** Feet and knees in leg entanglements are where it still fails (55% right on dotted pixels;
[:149](RESULTS.md#L149), [E05 RESULTS.md:27](../E05-dot-corrections/RESULTS.md#L27)), so Dhruva started marking the
truth with dots ([E05](../E05-dot-corrections/)).

Files: `RESULTS.md`. Code: `src/tangle/` (`tangle_core.py`, `autolabel_tangle.py`, `tangle_bench.py`).
