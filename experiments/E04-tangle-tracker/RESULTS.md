# E04 results: the tangle tracker (curated record)

Curated from the private workspace record `dataset/reviews/autolabel-v1/tangle/README.md` (2026-10-12 to 13).
Edits: review-server ports and session bookkeeping removed; "the user" -> "the reviewer (Dhruva)". Clip IDs
(`a2s...`, `a3m...`, `trial...`) are internal identifiers of public online videos; no frames from them are published.
Numbers are otherwise unchanged. "Current"/"production" = the auto-labeller's SAM 3.1 masks before this work.

# Tangle tracking (Stage 10, experimental) — 2026-10-12

Goal: correct pixel ownership between the two primary fighters through close contact (leg entanglements, guard,
back takes), where the mask fix ("dense" replacement) was judged only slightly better by blind A/B (9 better / 4 worse
/ 13 same / 6 both bad of 32).

## Design (first principles)
Single-frame models (SAM 3.1 detector, RTMO, ViTPose, SAM-3D-Body) all collapse an interlocked pair into one person,
so ownership has to be carried through time from frames where it is known.
- **Shots**: track only within a camera shot; cuts = ffmpeg scene cuts from a +-50 s scan plus hard cuts found on the
  decoded span frames (thumbnail difference > max(35, 3x median); ffmpeg at 0.35 missed same-arena camera switches).
- **Anchors**: frames where ownership is trusted: separated moments outside the window (scan; found for only 9/56
  window edges — rolls rarely separate), label frames whose text-detection masks agree with the dense video masks
  (IoU >= 0.9, 96% precise on gold), gaps > 60 frames filled with IoU >= 0.8 agreement frames, last resort the most
  confident label frame.
- **Segments**: SAM 3.1 tracker seeded with full masks for both fighters in one session (pixel competition before
  memory encoding), forward and backward between consecutive anchors, outward to the shot ends.
- **Fusion**: per-pixel probabilities, weighted by proximity to each pass's anchor x the tracker's presence belief
  for each fighter x hand-off quality (a pass that reaches the next anchor lost/swapped gets weight 0.05).
- **QA**: hand-off IoU at every anchor; forward/backward agreement per frame.
- Pair selection fix (`tangle_plan.choose_pair`): the dominant athlete pair by area + contact, not Astra's
  interval-level flag (a2s278: flagged P2+P3, correct P1+P2); changes 58/1526 intervals.

## Lessons
- Driving the multiplex tracker directly requires `propagate_in_video_preflight` (memory encoding of the prompts).
  Without it the tracker ignores the seeds after the prompt frame; v1–v3 results of the day were void because of it.
  Sanity test: swapping the two seed masks must swap the outputs.
- 6 GB GPU: the tracker keeps ~14 MB of GPU tensors per tracked frame; outputs > 32 frames behind are pruned
  (bit-identical results).

## Results
### Harmony4D ownership benchmark (03_grappling2, 56 videos, 5,500 frames at 10 fps, GT from SMPL)
Seeds at first/last frame from GT; scores are pixel fractions of each fighter's GT body.

| variant | correct (all) | correct (tangle) | wrong owner (tangle) | swallow >30% (tangle) |
|---|---|---|---|---|
| points_fwd (old prompting) | 0.853 | 0.845 | 0.056 | 0.063 |
| mask_fwd | 0.888 | 0.880 | 0.054 | 0.053 |
| mask_nonov_fwd (+ pixel competition) | 0.888 | 0.880 | 0.055 | 0.053 |
| mask_nonov_bidir (v4 core) | **0.891** | **0.882** | **0.053** | **0.052** |

- Full-mask seeds are the gain (+3.5 pts); pixel competition adds nothing; bidirectional fusion adds a little.
- v4 beats points_fwd on 55/56 videos. The regression (038/cam16, -0.16) is a dropped fighter mid-video
  (correct falls to ~0.4, wrong-owner ~0), not a swap; mid-span anchors in production would re-seed it.
- Wrong-owner barely moves (5.6% -> 5.3%): the remaining tangle error is real swaps/swallows,
  concentrated in sequences 025, 037, 038, 033. That is the next target; these videos are its test set.
- Not run (dropped for time): points_bidir, mask_nonov_bidir_pres, bidir_flip.

### Production-data QA (28 clips, v4)
236/236 anchor hand-offs ok (0 swap, 0 lost). Label-frame coverage 100% except a2s025 93%, a2s355 88%,
a2s497 98%, trial032 35% (7 shots, 1 anchor). Single-anchor clips (unchecked): a2s060, a4g113, trial089.

### Blind A/B by the reviewer (Dhruva) (reviews/tangle-ab), 25 of 77 judged, notes override keys
| category | tangle better | current better | same | both bad | n/a |
|---|---|---|---|---|---|
| maskfix_not_fixed | 8 | 0 | 2 | 1 | 0 |
| contact | 2 | 1 | 4 | 1 | 1 (trial032, unjudgeable) |
| apart | 2 | 0 | 0 | 1 | 0 |
| gold_faulted | 1 | 1 | 0 | 0 | 0 |
| **total** | **13** | **2** | 6 | 3 | 1 |

- Tangle wins 13-2 among decisive verdicts (sign test p~0.007); 9 of the 13 are "slightly better" by note.
- Biggest gain on the frames the earlier mask fix failed (8-0). No harm on apart frames.
- Recurring miss in both: bottom fighter's lower body unmasked (item 1); 3 both-bad.

## Round 2 (2026-10-12/13 night): what limits the tangle tracker, and anatomy ideas

Scripts: `tangle_bench2.py` (Harmony4D oracle/ablation harness), `pose_h4d.py` + `rtmo_to_kp.py` + `pose_kp_eval.py`
(mask-free RTMO skeletons), `cot_h4d.py` (CoTracker3 joint tracking), `pose_judge_eval.py`, `tangle_holdout*.py`
(pseudo-GT on our clips), `tangle_review2.py` (test-set pack). Production flags added to `autolabel_tangle.py`
(all off by default): `--iou-weight`, `--pose-anchors <variant>`, `--zoom`, `--holdout`.

### Harmony4D, cam01, 14 videos, tangle frames (bodies touch)
| variant | correct | wrong owner | limb-sized wrong blobs (frames) | >30% of a body wrong (frames) |
|---|---|---|---|---|
| points, forward (old prompting) | 0.809 | 0.058 | 0.027 | 0.088 |
| v4 core: masks, both fighters, fwd+bwd (`ancinf_mask`) | 0.885 | 0.057 | 0.012 | 0.077 |
| + ORACLE mask anchors every 1 s (`anc10_mask`) | **0.902** | **0.049** | **0.007** | **0.059** |
| + RTMO skeleton point anchors every 1 s (`anc10_rtmo`) | 0.870 | 0.056 | 0.019 | 0.080 |
| + per-fighter predicted-IoU fusion weights (`ancinf_mask_iouw`) | 0.885 | 0.058 | 0.013 | 0.079 |
| v4 + oracle-skeleton painting (all 28 videos) | 0.891 | 0.051 | 0.044 | 0.030 |

- **Drift is a minor part of the error**: perfect 1 s re-anchoring gains only 1.7 points. Most of what remains on
  Harmony4D is not limb confusion: contact sheets of the worst frames show one wrestler standing almost fully behind
  the other, with thin slivers scored wrong against an unclothed, slightly misfit SMPL body. Harmony4D (standing
  wrestling, far camera) can measure drift and do-no-harm, but not BJJ ground limb confusion.
- **Skeleton anchors hurt** even though the joints are clean (top-5 RTMO joints: 75% on the right joint, 0.5% on the
  partner in tangles; both fighters fully found in only 54% of tangle frames). Re-seeding SAM from points discards
  the tracked full-body memory and rebuilds a partial body (same lesson as the mask-loop campaign).
- **Skeleton split as a judge** (one skeleton's joints falling into both masks) does not rank frames by error here
  (Spearman -0.05).
- **SAM's own predicted IoU does rank frames by error** (worst fighter, forward pass: Spearman -0.81; bottom decile
  34% wrong-owner vs 7.5%), but as a fusion weight it changes nothing (both passes are unsure at the same frames).
  Kept as a candidate review/uncertainty flag.
- CoTracker3 joint tracking (seeded from GT joints at both ends) kept only ~1/3 of joints visible through tangles and
  ran out of RAM on CPU (8 GB); dropped.

### Our clips: withheld agreement anchors (10 dev clips, every 2nd agreement anchor withheld)
Pseudo-GT = label masks that agree with the dense masks (IoU >= 0.9; 96% good on the gold set). Biased to easy
frames and to the old pipeline's boundaries, so it only catches large effects.
| variant | withheld frames mean IoU (34) | worst fighter | other unused agreement frames (123) worst fighter |
|---|---|---|---|
| v4 (holdout) | 0.922 | 0.897 | 0.927 |
| v4 + zoom | 0.920 | 0.892 | 0.924 |
| v4 + RTMO skeleton anchors in gaps | 0.905 | 0.862 | - |

Skeleton anchors also lose on our BJJ footage: worse on 8 of 34 withheld frames, better on none; hand-offs into mask
anchors degrade from 63 ok / 5 weak to 53 ok / 11 weak / 3 swapped / 2 lost. Same cause as on Harmony4D.

### Test clips (12, random, contact-heavy, picked before the experiments)
v4: 187/188 hand-offs ok (1 swap, a3m011/chunk02), label coverage 95-100%. Blind pack: `reviews/tangle-ab-test`
(current vs v4 on random contact frames; v4 vs v4+zoom where they differ most).
Zoom changes masks mostly at limbs in the tangle (label-frame masks agree ~0.98 otherwise); whether that is better is
for the blind A/B on the test clips.

### Blind A/B by the reviewer (Dhruva) on the test clips (2026-10-13 morning, ~35 frames, notes override keys)
| question | tangle v4 better | other better | same | n/a |
|---|---|---|---|---|
| current vs v4 (random contact frames) | 7 | 4 (current) | 5 | 1 |
| v4 vs v4+zoom (frames where they differ most) | 4 (all "slightly") | 3 (zoom) | 10 | 1 both bad |
Zoom dropped. The reviewer's (Dhruva's) notes name the failure: legs/feet/hands in entanglements ("hooked leg", "classic example
that we need to fix"). Next round focuses on leg-entanglement clips (`legs.txt`: 12 random intervals with >= 50%
leg_entanglement/half_guard/closed_guard frames and >= 5 leg_entanglement frames, unused so far).

### BBoxMaskPose v2 (external/BBoxMaskPose, .venv-bmp: py3.11, torch 2.1.2+cu121, mmcv 2.1.0)
- **PMPose given the tracker's masks as hints** (`pmpose_h4d.py`), Harmony4D cam01, 778 tangle person-frames:
  95% of confident joints on the right fighter's joint, 1.8% on the partner's, 96% of visible joints found;
  knees/ankles 97.5% right, 0.7% partner. (RTMO without masks: 76% right, 27% found.)
- SAM-pose2seg must be fed RGB (BMP's own code passes OpenCV BGR); with BGR the masks were speckled on dark clothing.
- `bmp_tangle.py`: tracker masks -> PMPose -> SAM-pose2seg (k most visible keypoints) -> repeat; `<out>c` keeps, per
  fighter and frame, the more pose-consistent of tracker and BMP mask (BMP's consistency score).
- `limbfix.py`: PMPose skeleton flags a confident limb joint (elbow/wrist/knee/ankle) outside its fighter's mask; SAM
  corrects that fighter's mask from its current mask + clicks on the limb; gain accepted only near the limb.
- Leg set run (11 clips; a2s230 had no usable pair masks): limb violations on 22-38 of ~43 label frames per clip,
  mostly a knee in the partner's mask. First limbfix version accepted gains in a disk around the joint and moved a
  partner's torso (a2s126#25, 24k px); now a capsule along the limb (2.7k px there).
- Blind pack `reviews/tangle-ab-legs`: current vs v4 (11), v4 vs v4+limbfix (33), v4 vs BMP-consistency
  choice with 6-keypoint prompts (22).

## Leg pack verdicts (tangle-ab-legs, 17 answered) and review dots
Effective verdicts (notes over keys): v4 vs BMP-guided (bmp6c) 6-0 for v4 (BMP drops a fighter or the legs); v4 vs
limbfix (v4lf) 3 v4 / 2 v4lf / 2 same / 1 both bad, a wash; current vs v4 1 / 1 / 1. Notes: feet and ankles in leg
locks still go to the wrong fighter; "a proper leg entanglement scenario ... the kind we need to work on" (a2s354).
BMP-guided and limbfix are not pursued further as-is.

From now on, review pages take dots: red / blue / neither, placed on any image, marking the truth at a pixel
(independent of A/B). items.json carries each crop's frame box; `tangle_dot_eval.py <pack> <variants...>` scores any
variant against them, so dots become a reusable point-level test set on the hard frames (and point supervision for
fine-tuning later: Cheng et al., "Pointly-supervised instance segmentation", CVPR 2022 — ~10 labelled points per object
recover 94-98% of fully supervised Mask R-CNN accuracy).

