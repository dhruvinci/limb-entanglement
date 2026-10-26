# E05 results: reviewer dots as corrections (curated record)

Curated from the private workspace record `dataset/reviews/autolabel-v1/tangle/README.md` (2026-10-13 to 14) and
the reviewer's later assessment (2026-10-22, last section). Edits: review-server ports, process IDs and session
bookkeeping removed; a private essay draft is not referenced; "the user" -> "the reviewer (Dhruva)". Clip IDs are
internal identifiers of public online videos; no frames from them are published. Variant names: `v4` = E04 tracker;
`w` = corrections with a wide (3.6 s) radius; `2`/`3` = correction round; `m` = corrections held in tracker memory;
`k` = bone fill; final `m` = margin bone fill; `e` = corrections forced to honour every dot; `bmpw` = BBoxMaskPose masks.

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

## 2026-10-13 (afternoon/evening): reviewer dots as corrections, hole filling, negative results
Dot sets: leg pack 362 dots / 24 frames (used as corrections below), near pack (`reviews/tangle-ab-near`) 82 / 18 and wide pack (`reviews/tangle-ab-wide`) 101 / 20,
never used as input. Outcomes per dot: right / swap (other athlete) / hole (nobody).

**Seeds carry the errors.** Leg-pack dots on v4's agreement-anchor frames: 27/41 right (66%), identical dot for dot to
the production masks. v4 by body part: feet 55%, knees 55%, hips 70%, hands 78%, arms/torso/head 96%.

**Negative results (leg pack, 362 dots; v4 = 261 right / 54 swap / 47 hole):**
- SAM 3D Body meshes (`s3d_own.py`, depth-ordered render): best variant 280 right but 69 swaps; depth order wrong at
  some crossings (trial046 #29). Mesh-sampled SAM clicks (`mesh_sam.py`): 267-270.
- Sapiens 0.6B seg (`sapiens_parts.py`): resolution floor. Meta's demo image: 53.9% human pixels at 1024 px person
  height, 0.1% at 512, 0 at 300; our athletes are 150-400 px. CPU/GPU, torch 2.1/2.5 identical. Dropped.
- Colour model per clip: 87% where confident, 55-61% on hands/feet (skin).

**Dots -> corrections -> tracker anchors** (`dotfix.py`, `autolabel_tangle.py --corrections --corr-radius`):
- dotfix (SAM-pose2seg b+, current mask as +-10 logits, own dots positive, others negative): half the dots as clicks,
  other half held out: 72% -> 78% (no mask prior: no gain).
- Leave-one-frame-out: narrow radius (6 span frames) = v4 exactly on all 22 withheld frames (the next agreement anchor
  re-seeds the old error); wide radius (36 span frames = 3.6 s) on the 8 frames 1.5-3 s from a correction: 74% -> 82%.
- Blind: near pack v4 vs v4d 5-12 (1 same, 6 dots only); wide pack v4 vs v4w 0-21 (6 dots only).
- Corrections convert swaps into holes: near+wide (183 dots) swaps 88 -> 33, holes 43 -> 74.

**Who gets a hole** (dots that are holes under v4w and that a rule fills): by BMP outline (`holefill.py`) 37/92 (40%);
by skeleton capsules (`bonefill.py`, PMPose COCO-17, feet = extended shin) 44/71 (62%). Nearest athlete is right on
only 21/74 hole dots (the trapped-limb paradox). `bonefill.py --near 40` guards against BMP masks on spectators
(a2s474/02 #19). Flip fill (`flipfill.py`: pixel taken from A by a correction -> B) scored 77% on the wide pack but was
designed on those dots and visibly wrong on its largest changes (spectator, a2s054/02 #24 purple gi = red, a3m009 #6);
strict version = no gain. Dropped.

**Clue judge** (`clue_judge.py`, logistic regression on per-dot cue votes, leave-one-clip-out, 183 dots): unconstrained
weights learn the reviewer's dot placement (negative weight on v4) -> 73%, not trustworthy; non-negative weights 63%
(v4w 0.52, 3D 0.67, colour 0.42, bones 0) vs 61% for the best single variant.

Corrections to earlier statements: a2s054/02 #24 outline fill (purple gi leg -> red) was right, not a failure; near-pack
ties were 1, not 3.

## 2026-10-13 (night): denser corrections, memory, motion, feet — what moved the dots
Clean test for anything built from the near-pack dots: the 97 wide-pack dots on the 19 frames that are not near-pack
frames (`eval_w2.py`). Right / swap / hole.

| variant | what | wide (97, independent) |
|---|---|---|
| v4 | anchored tracker | 24 / 50 / 23 |
| v4w | + leg-pack corrections, radius 3.6 s | 39 / 17 / 41 |
| v4w2 | + near-pack corrections too (`corr_legs_near.json`, 52 frames) | 51 / 15 / 31 |
| v4wk | v4w + bone fill (`--near 40`) | 61 / 30 / 6 |
| **v4w2k** | v4w2 + bone fill | **69 / 23 / 5** |

- **Denser dots help, and the gain lands on frames nobody dotted**: v4w -> v4w2 = +12 on independent frames.
- **Corrections in memory** (`--corr-memory 1`: every correction frame is a conditioning frame of every pass). First
  run crashed: SAM 3.1 multiplex refuses a second mask seed once its state exists ("No existing output found for frame
  N"); `tangle_core.track` now rebuilds the (deterministic, random=False) multiplex state before each seed frame.
  Leave-one-frame-out on the leg pack: 83/97 vs 80/97 for radius-only (`lor`), the +3 all on a2s354 #12. Small.
- **Motion, forward** (`pointvote.py`: CoTracker3 points seeded on each athlete vote where they land): abstains on
  every hole dot (points that start on a mask never reach the pixels the masks miss). Also needed
  `backward_tracking=True` (the offline predictor only tracks forward from the query frame by default) and WIN 25
  (50 ran out of memory on 6 GB). Leg locks never fully separate: whole-frame "separated" seeds found 0-1 per clip.
- **Motion, reverse** (`pointlook.py`: hole pixels tracked +-2.5 s, owner looked up where that spot was unambiguous):
  at hole dots 9 right / 9 wrong / 56 abstain; at swap dots 10 / 10. A coin flip. Its trusted owners come from v4w's
  own masks, so it agrees where v4w is right (42/44) and inherits v4w's mistakes elsewhere. v4wlk (lookup then
  bones) vs v4wk2: near 47 vs 43, wide 64 vs 63; noise-sized.
- **RTMW whole-body feet** (`rtmw_feet.py`, heel + toes, other athlete greyed out, kept only when RTMW's ankle meets
  PMPose's): feet for ~half the ankles, zero dot outcomes changed (v4w2kf = v4w2k, v4wkf = v4wk2). Bone-fill errors
  are identity errors (PMPose's leg chain reaching the partner's foot, trial046 #7-#24), not foot-shape errors.
- **Clue judge with motion cues**: all dots LOCO 65.6% (look gets the top weight, mostly by agreeing with v4w); holes
  only 52.7%, below bone fill alone. Not used.
- **"Unsure" flags do not find the remaining errors.** v4w2k accuracy at wide dots where it agrees with BMP 71%,
  disagrees 71%; agrees with old v4 53%, disagrees 85%. What is left are errors every method shares (inherited from
  the original seeds), and cue disagreement cannot point at those. Random frames are the honest next dots.

**Final for review: v4w2mk** = tracker with leg + near-pack corrections (radius 3.6 s), all correction frames held as
conditioning frames of every pass (`--corr-memory 1`), then bone fill (`--near 40`). Wide independent dots:

| variant | right / swap / hole (97) |
|---|---|
| v4w2 | 51 / 15 / 31 |
| v4w2l (+ reverse motion lookup) | 56 / 16 / 25 |
| v4w2m (corrections in memory) | 67 / 10 / 20 |
| v4w2k / v4w2lk | 69 / 23 / 5, 71 / 23 / 3 |
| **v4w2mk** | **75 / 17 / 5** |

Per frame (19 frames) v4w2mk vs v4w2k: better 5, worse 1 (trial046 #27, -2), same 13. Memory mode vs v4w2 masks over
all 944 athlete-frames: median IoU 0.994; the only large changes (trial018/02 #34-36, #40) are v4w2 losing the
white-shirt athlete entirely and memory mode holding both. Memory helps more with denser corrections (+16 here vs +3
on the leg-pack leave-one-out with sparser ones). Known look: bone fill can paint stripy red/blue patches where both
athletes' shins cross (a2s054/02 #16).

Blind pack `reviews/tangle-ab-final`: v4w vs v4w2mk, 44 frames on the
11 leg clips, none previously dotted; key.json records the picker per frame (diff 22 / random 11 / unsure 11) so
verdicts and new dots can be split by picker (does "unsure" find errors better than random?).

## 2026-10-14 (morning): blind verdicts on the final pack
`reviews/tangle-ab-final`, v4w vs v4w2mk, 36 of 44 frames judged (notes over keys: #1 "Both are great!" and #15
"Neither is better" count as same). **v4w2mk 19, v4w 6, same 10, both bad 1.** By picker: diff 15-2, unsure 4-2 (4
same), random 0-2 (6 same, 1 both bad). 326 new dots on 21 frames (159 red, 167 blue, no "none"), a fresh test set:
no variant used them.

| dots right (fresh) | all 326 | diff 111 | random 153 | unsure 62 |
|---|---|---|---|---|
| current | 60% | 65% | 67% | 32% |
| v4 | 59% | 62% | 69% | 27% |
| v4w | 61% | 68% | 69% | 31% |
| v4w2m | 63% | 71% | 71% | 32% |
| **v4w2mk** | **69%** | **79%** | 72% | 44% |
| bmpw | 65% | 71% | 67% | 50% |

- On random frames the whole stack adds ~3 points; last night's 77% was on frames picked where variants differ.
- **Correction:** frame-level "unsure" picking does find the hard frames (every variant 27-50% there vs 67-72% on
  random frames). The night's claim that disagreement can't locate errors holds only dot by dot, within frames.
- **A wrong fill looks worse than a hole.** v4w won #2 (a2s054/02 #16, "This has gotten worse from previous reviews"),
  #28 and #33, where v4w2mk's bone fill painted stripes or the wrong foot, although v4w2mk had more dots right on #2
  (5 vs 0/15) and #33 (7 vs 4/11). `bonefill.py --margin 0.5` (fill only when one fighter's bones clearly win) undoes
  9 wrong and 8 right fills over fresh + wide dots: the ambiguous fills are a coin flip.
- **Closed guard, legs traced through occlusion** (#23 a2s474/02 #12, reviewer: "This should be solvable!"): every
  tracker variant 0/10, bmpw 10/10. The bottom athlete's legs wrap the top athlete's torso and her hips are hidden,
  so the tracker joins the legs to the torso they touch. Bone fill cannot help (not holes). `--override 0.6,1.5`
  (confident leg bones take pixels from the tracker): never fires here (PMPose knees 0.44/0.47 < 0.5) and is slightly
  negative overall (fresh 222 vs 225, wide 73 vs 75). Here the outfits differ completely (leopard vs geometric print).

## 2026-10-14 (day): correction round 3 (final-pack dots, split)
The final-pack dots were split by frame (`split_final.json`): 11 frames / 154 dots became
corrections (incl. closed guard a2s474/02 #12), 10 frames / 172 dots were held out (incl. its neighbour #7, 54 dots).
v4w3m = v4w2m's corrections + these 11, same settings (`--corr-radius 36 --corr-memory 1`); v4w3mk / v4w3mkm = bone
fill without / with `--margin 0.5`.

**dotfix fix first.** Checked before the run: SAM's click-refined masks spread far past the dots. a2s474/02 #12 has
only red dots (tracing the bottom athlete's legs), and red took the top athlete's whole back (blue 58k -> 10k px);
on a2s270 #7 red reached spectators in red shirts. `dotfix.py --local 40 --body bmpw`: a pixel may move from one
fighter to the other only within 40 px of a dot of its new owner; unowned pixels may be filled anywhere inside the BMP
pair union (spectators are outside it), so a dotted leg the tracker missed fills whole instead of as discs. Off by
default, so earlier correction sets are unchanged.

| dots right | held-out final 172 | correction frames 154 | wide-independent 97 |
|---|---|---|---|
| v4w | 120 (70%) | 80 | 39 |
| v4w2mk (FINAL) | 129 (75%) | 96 | 75 |
| v4w2mkm | 129 | 91 | 72 |
| v4w3m | 153 (89%) | 142 | 65 |
| **v4w3mk** | **155 (90%)** | 142 | 73 |
| **v4w3mkm** | **155 (90%)** | 142 | 71 |
| bmpw | 117 | 96 | 44 |

- Held-out frames 2-5 label frames from a correction: a2s054/01 #21 22 -> 33/34, a2s354 #21 25 -> 32/34, closed guard
  #7 46 -> 51/54. trial054 #29 (13 label frames from its correction) unchanged.
- Closed guard: the leopard legs wrapping the top athlete are blue in v4w2mk from #10 to #15 and red in v4w3mkm on
  all of them, the top athlete's back still blue (visual check, not yet blind). One corrected frame (10 dots) carried
  about +-3 label frames. Not automatic: it needs a dotted frame per such sequence.
- Wide -2: a3m009 #24's correction pushes red over a sliver of blue's hand tucked under red at #30 (2 dots).
- v4w3mkm vs v4w3mk: same held-out, wide 71 vs 73 but 20 vs 27 swaps over both sets (fewer wrong fills, more holes),
  which the reviewer prefers.

## 2026-10-14 (day): outfit cue (negative on the current stack)
Probe `outfit_probe.py` (CPU): per clip, a small MLP on colour + texture (Lab, local mean/std,
gradient energy in 4 directions, at 5/11/21 px) learns each fighter's look from pixels the tracker owns >= 30 px from
the other fighter (plus correction masks), trained with the dot's own quarter of the clip held out; scored at dots
as an override of a variant ("anywhere") or as a filler of its holes. Clip separability on held-out trained pixels:
73-90% (skin is shared); lowest on trial046 (73%) and on the closed-guard clip a2s474/02 (75%).

| outfit vote, conf >= 0.8 | override anywhere | fill holes only (right / wrong fills) |
|---|---|---|
| on v4w2mkm, fresh 326 + wide 97 | 292 -> 323 (fresh +32, wide -1) | +34 (34 / 9) |
| on v4w3mkm, held-out 172 + wide 97 | 226 -> 214 | +8 (8 / 7) |

- It learns from the tracker, so it inherits the tracker's systematic errors: on closed guard #12 it gets 3/10 (the
  tracker had the leopard legs blue for frames #10-#15, so "leopard leg" was learned as blue). The morning's guess
  that appearance would solve this frame was wrong; the dot corrections did.
- Training on correction masks only (human-checked, few frames) is no better: hole fills 5 / 5 on v4w3mkm, closed
  guard #7 40/54 vs v4w3mkm 51/54.
- Before today's corrections it was a better hole filler than bones (v4w2m holes at fresh dots: outfit 50/72 right,
  bone fill 18 right / 13 wrong / 41 left). After them the holes left are the hard residue (feet, hands, skin) and
  it is a coin flip. Not built into masks, no blind pack.

Blind pack `reviews/tangle-ab-r3`: v4w2mk (FINAL) vs v4w3mkm, 44 frames on the 11 leg clips, none
previously dotted in any pack (diff 22 incl. closed guard a2s474/02 #10 and #11 / random 11 / unsure 11, unsure
scores from v4w3mkm, bmpw, v4w3m). Same design as the final pack, so verdicts compare by picker.

## 2026-10-14 (afternoon): blind pack r3 verdict, and dots that corrections drop
**Blind (reviews/tangle-ab-r3, 44 frames never dotted before): v4w3mkm beat v4w2mk 28-6**, 8 same, 1 both bad, 1
dots only (#7). Notes override keys: #20 "both good!" -> same; #37 (no key) "A is better in terms of the mask, it
just got one segment of the mask wrong" -> v4w2mk. #23 (closed guard #19) and #29 were "can't tell, probably".

| picker | v4w3mkm | v4w2mk | same | both bad / none |
|---|---|---|---|---|
| diff 22 | 19 | 1 | 0 | 1 / 1 |
| random 11 | 5 | 3 | 3 | 0 |
| unsure 11 | 4 | 2 | 5 | 0 |

Compare the final pack (v4w vs v4w2mk 19-6): there random frames went 0-2 against the change; here 5-3 for it.
Closed guard #10 and #11 both went to v4w3mkm (the leopard legs red). **v4w3mkm is the new FINAL.**

Fresh dots (61 on 11 frames, nobody used them): v4w 24, v4w2mk 29 (48%), v4w3m 34, v4w3mk 38, **v4w3mkm 37 (61%)**,
bmpw 31. Swaps 23 -> 10, holes 9 -> 14. Dot frames are few and hard (the reviewer dots what is wrong in both), so
the % is low.

**Losses share one cause.** All six v4w2mk wins are frames where v4w3mkm leaves part of a fighter unmasked:
trial018/02 #21/#25/#26 (a bare foot), a2s126 #26/#27 (a centre patch), a3m007 #21 (a thigh). trial018: the foot
sticking out at the left under the bottom athlete is the top athlete's (trapped foot); the reviewer dotted it blue on
#24 (a correction frame) and again on #26/#27 in this pack. v4w2mk has it red (wrong owner, looks complete), v4w3mkm
has a hole. The #24 correction itself had red / nothing at the two foot dots: SAM's click refinement stopped short
of the foot and `--local` cannot add what SAM never proposed, so the tracker took a wrong anchor.

Measured over every correction frame: **58 of 600 red/blue dots were not honoured by the correction made from them**
(44 holes, 14 other-fighter): legs 42/361, near 3/82, final-A 13/154.

`dotfix.py --enforce` / `dotenforce.py` (post-process an existing corrections file): each dot the correction still
gets wrong gets a SAM part mask from one click on it, other-colour dots negative; the smallest of SAM's 3 masks that
holds no negative dot, kept connected to the dot. It may take the other fighter's pixels within 20 px of the dot
(40 if the part holds another dot of the same colour: a traced part, like heel + toes of the trial018 foot); it may
fill unowned pixels inside the BMP pair or within 120 px; else a 6 px disc. Tuned by eye on the biggest changes
(4 rounds: best-scored mask -> discs; 40 px steal -> a half-disc of the top athlete's shirt on a2s270 #29; BMP-only
fills missed a2s054/02 #6/#22 legs that BMP also missed). Result `corr_lnfA_e.json`: 597/597 dots honoured (from
539), 113-26552 px changed per frame. Not yet tracked: the test is a tracker run (v4w3me) on the same split, scored
on held-out final-B, wide and these r3 dots, then blind.

**Tracked (v4w3me = v4w3m with `corr_lnfA_e.json`; v4w3mekm = + margin bone fill), 15:57-17:05:**

| dots right | held-out final-B 172 | final-A (in-sample) 154 | wide-independent 97 | r3 fresh 61 |
|---|---|---|---|---|
| v4w3mkm (FINAL) | 155 | 142 | 71 | 37 |
| **v4w3mekm** | **156** | 153 | 71 | **42** |

On dots no variant used: 7 gained, 1 lost (wide trial046 #13: a blue foot dot becomes a hole). r3 gains are the
trial018 trapped foot on #26/#27 (4/4, was 0/4) and a2s054/01 #11. Masks change only near enforced corrections
(~100 of ~470 label frames by > 500 px; a2s126 and a2s474/02 unchanged, so the a2s126 #26/#27 losses are not
addressed). By eye: trial018 #21/#25 foot now blue and whole; a2s054/02 #26 foot and #9 shin whole instead of split;
a2s270 #2 the arm round the head one owner. Blind pack `reviews/tangle-ab-e`: v4w3mkm vs v4w3mekm,
19 frames on the 9 changed clips (diff 10, random 9), none dotted in any pack.

## Paused 2026-10-22: state at the end of the thread

- **Final = v4w3mkm** (blind round 3, 28-6 over v4w2mk). Candidate v4w3mekm (enforced corrections): its blind pack
  (`reviews/tangle-ab-e`, 19 frames) was built and **never reviewed**.
- The planned next step (production corrections from every dot so far) was not run.

## Assessment (Dhruva, 2026-10-22)

On review, Dhruva judged that the dot rounds had not improved any model or pipeline: each gain measured how far the
reviewer's own labels carry to nearby frames, no model was trained, and on a clip nobody dotted none of it changes
anything. From then on dots count as a test set (or as training data for a model that runs on new clips), not as
per-frame patches, and improvements are reported on undotted clips or on ground truth such as Harmony4D. The thread
was paused and the work turned to measuring two-person engagement (see E07).

## Dot sets (summary added for this repo, from the counts above)
| pack | dots | frames | role |
|---|---|---|---|
| leg pack | 362 | 24 | corrections (round 1) |
| near pack | 82 | 18 | corrections (round 2) |
| wide pack | 101 | 20 | test only (97 of them on frames independent of the near pack) |
| final pack | 326 | 21 | 154 dots / 11 frames corrections (round 3), 172 / 10 held out |
| round-3 pack | 61 | 11 | test only (fresh) |
| **total** | **932** | **94** | all placed by one reviewer (Dhruva) |
