# E06 results: other ownership cues (curated record)

Curated from the private workspace: the tangle record `dataset/reviews/autolabel-v1/tangle/README.md`
(2026-10-13 to 14), its evaluation log (`day2.log`, 2026-10-13 14:25) and a workspace
note on Sapiens (2026-10-13). All three cues were scored against the reviewer's dots (E05): a dot marks which
athlete truly owns one pixel. "Leg pack" = 362 dots on 24 frames of 11 leg-entanglement clips (public online
videos, not redistributed). Baseline `v4` = the E04 tracker. Outcomes per dot: right / swap (other athlete) / hole
(nobody). Clip IDs are internal identifiers.

## 1. Sapiens body-part segmentation: a resolution floor (2026-10-13)

Script: `sapiens_parts.py` (Sapiens 0.6B, Goliath 28 parts, TorchScript release). Idea: Sapiens labels every human
pixel with a body part (thigh, shin, foot ...), not with a person; use it to fill limb pixels the tracker leaves
unowned (watershed from the two athletes' masks, variant `sapw`).

- On the leg pack it changed almost nothing: `sapw` 263/362 right vs `v4` 261/362 (day2.log).
- Cause, checked on Meta's own demo image with the official preprocessing (RGB, resize to 1024x768, the published
  mean/std, fp32): 53.9% of pixels labelled human at full size, 0.1% when the image is downscaled to 512 px tall,
  0 at 300 px. The athletes in this footage are about 150-400 px tall, so Sapiens labels next to nothing on them,
  also on per-athlete 3:4 crops. CPU vs GPU and torch 2.1 vs 2.5 give identical outputs, so the runtime is not the
  cause.
- First reported as "bad on BJJ"; the resolution test showed the real cause.
- Not tested: Sapiens' pose, depth and normal heads, and super-resolution before Sapiens.

| image height (person fills most of it) | pixels labelled human |
|---|---|
| 1024 px | 53.9% |
| 512 px | 0.1% |
| 300 px | 0% |
| our athletes | 150-400 px tall |

## 2. SAM 3D Body, depth-ordered ownership (2026-10-13)

Script: `s3d_own.py`. Idea (the reviewer's "human shapes as primitives in 3D"): fit one SAM 3D Body mesh per athlete,
render both into the frame with a depth buffer, and let the nearer body own each pixel. A mesh has volume, legs
attached at the hips and a front-to-back order, which a skeleton or a 2D mask lacks. Variants: ownership only inside
the tracker's masks (`o`) or the union of masks and rendered bodies (`f`); prompts from box + mask or box alone.
`mesh_sam.py` instead used the rendered ownership only to place SAM clicks (8 or 4 per athlete).

| leg pack, 362 dots | right | swap | hole |
|---|---|---|---|
| v4 (tracker) | 261 (72%) | 54 | 47 |
| SAM 3D Body depth order, best variant (`s3dmf`) | 280 (77%) | 69 | 13 (derived: 362 - 280 - 69) |
| mesh-sampled SAM clicks (`msam8` / `msam4`) | 270 / 267 | - | - |

- More dots right, but more swaps: it fills holes and moves errors rather than removing them. The depth order is wrong
  at some crossings (trial046 #29).
- In the later cue-combination test (E05, "clue judge", 183 dots, leave-one-clip-out) the 3D vote got the largest
  non-negative weight (0.67), yet the combination reached only 63% against 61% for the best single variant.
- Not adopted.

## 3. Appearance: per-clip colour model and outfit MLP (2026-10-13 / 14)

**Colour model** (`appearance_probe.py`, Lab histograms of each athlete from agreement-anchor frames, 7x7 patch at each
dot): 87% right where confident, but 55-61% on hands and feet, where both athletes show skin.

**Outfit MLP** (`outfit_probe.py`, CPU): per clip, a small MLP on colour + texture (Lab, local mean/std, gradient
energy in 4 directions, at 5/11/21 px) learns each athlete's look from pixels the tracker owns at least 30 px from the
other athlete (plus correction masks), trained with the dot's own quarter of the clip held out. Clip separability on
held-out training pixels: 73-90%.

| outfit vote, confidence >= 0.8 | override anywhere | fill holes only (right / wrong fills) |
|---|---|---|
| on v4w2mkm (before round-3 corrections), fresh 326 + wide 97 dots | 292 -> 323 | +34 (34 / 9) |
| on v4w3mkm (after round-3 corrections), held-out 172 + wide 97 dots | 226 -> 214 | +8 (8 / 7) |

- It learns from the tracker, so it inherits the tracker's systematic errors: on the closed-guard frame it gets 3/10
  (the tracker had the bottom athlete's legs as the other athlete's for six frames, so that pattern was learned with
  the wrong owner). The dot corrections fixed that frame; appearance did not.
- Training on the human-checked correction masks only is no better (hole fills 5 right / 5 wrong).
- Before the round-3 corrections it was the better hole filler (50/72 right vs bone fill 18 right / 13 wrong);
  after them the holes left are feet, hands and skin, and it is a coin flip. Not built into the masks.
