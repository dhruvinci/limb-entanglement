# Findings log

One line per finding, oldest first. Each links to the experiment that supports it.

## 2026-08-26
- A frontier VLM names the armbar and keeps the two athletes apart, but puts its "arm" point on the torso; SAM then
  masks the trapped forearm as the wrong athlete, and no SAM candidate covers it. ([E00](experiments/E00-tool-chain/))
- A body prior (PromptHMR) can pull the forearm to the right athlete, but collapses the two bodies in the tangle; its
  video version is smooth and still wrong. ([E00](experiments/E00-tool-chain/))

## 2026-08-27
- Qwen3.8-27B never named the armbar: its answer described a suplex and a triangle. ([E01](experiments/E01-qwen-ownership-probe/))
- Its per-athlete states behave as an identity code: six different prompts give the same map within 0.0011 IoU, a
  matched random code does as well, and the trapped arm stays with the wrong athlete. ([E01](experiments/E01-qwen-ownership-probe/))

## 2026-09-05
- A trained decoder on Qwen features reaches 0.875 athlete IoU within the clip, yet gets 0 of 27 trapped-arm cells
  right with the language states. Ordered, reversed and shuffled frames are indistinguishable.
  ([E02](experiments/E02-ownership-decoders/))
- Subtracting each athlete's identity state does not expose a usable action or contact direction (6 of 7 gates fail).
  ([E02](experiments/E02-ownership-decoders/))
- Companion study (dated only "by 2026-09-28"): fusing DINO and Qwen features raises body IoU but halves contact
  ownership. ([E02](experiments/E02-ownership-decoders/))

## 2026-09-21
- The cached Qwen answers swap athlete IDs between prompts, and one invalid answer still produced states. Earlier
  "Qwen knows, the decoder can't use it" readings were too strong. ([E03](experiments/E03-audits-final-cache/))

## 2026-10-12
- Seeding a video tracker with whole masks of both athletes, between frames where ownership is known, lifts correct
  ownership in Harmony4D tangles from 0.845 to 0.882 and wins blind 13-2 on real clips. Wrong-owner pixels barely move.
  ([E04](experiments/E04-tangle-tracker/))

## 2026-10-13
- The old Qwen token reshape misplaced 14,080 of 14,400 grid locations; all earlier pooled and full-grid spatial
  results are void. Fixed, the same probe scores 0.458 instead of 0.210. ([E03](experiments/E03-audits-final-cache/))
- With correct tokens and full controls (636 jobs), Qwen's language states still add nothing, unseen clips (private
  footage, numbers withheld) score no better than a geometry-only baseline, and frame order does not matter. The cache was retired.
  ([E03](experiments/E03-audits-final-cache/))
- Sapiens segmentation has a resolution floor (0.1% human pixels at 512 px); SAM 3D Body depth order trades holes for
  swaps. ([E06](experiments/E06-other-ownership-cues/))
- Reviewer dots turned into tracker corrections win blind 21-0, but mostly convert swaps into holes.
  ([E05](experiments/E05-dot-corrections/))

## 2026-10-14
- Three correction rounds: held-out dots 75% -> 90% and blind 28-6, but on random undotted frames the stack adds about
  3 points. Corrections carry a few frames. ([E05](experiments/E05-dot-corrections/))
- A per-clip outfit model learns the tracker's own errors and cannot fix them. ([E06](experiments/E06-other-ownership-cues/))

## 2026-10-22
- Dhruva's review: the dot rounds improved no model. Dots are a test set, not progress; the thread was paused and the
  question moved to measuring engagement in 3D. ([E05](experiments/E05-dot-corrections/), [E07](experiments/E07-what-it-led-to/))
