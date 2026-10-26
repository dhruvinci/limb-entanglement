# E03 results: provenance audit, token-layout audit, final cache campaign (curated record)

Curated from the private workspace: `qwen3-vl-seg-decoder/artifacts/grounding-audit-v1/RESULT.md` (2026-09-21),
`final-cache-campaign/preparation-layout-audit.json` and `progress.txt` (2026-10-13), and the campaign's findings
archive (`final-cache-findings-20261013`). Part 3's tables were generated from the findings archive by
`src/qwen/final_cache/summarize_findings.py`; nothing in them is typed by hand. Edits: cloud account, pod and
volume identifiers removed; process notes condensed; every result measured on the four private product clips
is withheld (the findings archive itself is not published because it holds per-frame records of those clips).

## Part 1: grounding provenance audit (2026-09-21)


This audit reads the original cached grounding answers and the `contract_valid`
tensor, without loading Qwen, Torch, CUDA, or activation arrays. `audit.json`
contains all 34 records with their SHA-256 hashes, canonical answers, initial
answers, repair answers, and contract status. Syntax validity is deliberately
separate from physical actor identity and semantic correctness.

### Findings that change the next experiment

1. **Actor identities change across prompt conditions on one private clip.** The identity and action prompts
   assign A1 and A2 one way, the contact prompt the other way. Stacking these states by filename alone is not a
   consistent physical-actor comparison.
2. **One private clip's contact grounding failed its own output contract.** The canonical answer is empty and
   `contract_valid` is false, yet the extractor used the format-repair text to generate the A1/A2 marker states.
   Those tensors pass numerical checks but are not a valid contracted contact query.
3. **Armbar temporal ablations have identity confounds.** The 2 fps and five-frame
   action answers assign A1 to the plain-black athlete. The 4 fps action answer
   assigns A1 to the patterned athlete. A decoder comparison must account for
   this independent of frame order or sample rate. Shuffled descriptions are
   less explicit and require visual identity review; no automatic mapping is
   inferred from text similarity.
4. **The main armbar action description is not a correct armbar account.** The
   exact 4 fps action/off answer describes a suplex followed by a triangle around
   A1's neck. Contact/off describes a wrist lock on A1's right arm. Action/xhigh
   describes a throw and top position without identifying an armbar. Correct
   action understanding therefore cannot be assumed for these cached states.

### Interpretation of earlier results

The recorded segmentation scores and contact errors remain observations of the
executed runs. The primary 4 fps identity/action/contact descriptions agree on
the patterned-versus-plain actor assignment, so the discovered swaps do not
automatically explain its negative result. However, that result is conditional
on the model's imperfect grounding text and legacy pseudo-label supervision.
It is not a clean test of whether *correct* relational understanding can improve
ownership. The broad claim that the model knows the correct armbar relationship
but the decoder cannot extract it was too strong.

The temporal outputs being similar remains an observed behavior. A causal claim
about temporal understanding requires consistent physical actor identities and
an audit of grounding changes across conditions. The cached contact arm on the private clips
also cannot be used as if all of them had valid, consistently ordered states.

### Changes and verification

- Added a standard-library grounding auditor with bounded header reads (1 MiB
  maximum), strict boolean parsing, JSON/flag consistency, metadata/path checks,
  and streamed hashes.
- Cache verification for the private clips now includes upstream grounding artifacts and
  rejects invalid grounding before launching a decoder job. It does not claim
  that valid JSON proves correct physical identities or action semantics.
- Original caches and prior results are preserved unchanged. No new model run,
  paid compute, relabelling, or retrospective best-performing permutation was used.
- Regression suite: 233 tests passed, CPU-only, under `MemoryMax=2G` and
  `MemorySwapMax=0`; peak RSS 818,096 KiB and elapsed time 11.95 seconds.
- Grounding audit: 34 records, 1 invalid contract; peak RSS 21,072 KiB.

(The audit's remaining sections set up an independent review of which cached query belongs to which physical
athlete; that alignment was later frozen and used by Part 3.)

## Part 2: token-layout audit (2026-10-13)

Qwen's image processor emits raw vision tokens in **grouped** order: each 2x2 block of neighbouring patches is
consecutive, so that the later 2x2 merger can concatenate them. The original extractor and readers reshaped the raw
token sequence straight into a (height, width) grid. The audit builds an image whose patches encode their own row
and column, runs the exact pinned processor (transformers 5.16.1, patch 16, merge 2, no model weights) and compares
(`data/layout-audit.json`):

| grid (rows x cols) | locations | misplaced by direct reshape | misplaced after `restore_grid` |
|---|---|---|---|
| 4 x 6 | 24 | 16 | 0 |
| 8 x 4 | 32 | 16 | 0 |
| **160 x 90 (the real full-resolution grid)** | **14,400** | **14,080 (97.8%)** | **0** |

`restore_grid` (in `src/qwen/final_cache/core.py`) is the fix: reshape to (h/2, w/2, 2, 2, C), swap the middle axes,
reshape to (h, w, C). The audit's own conclusion: "Raw vision tokens use grouped spatial order. Direct reshape is not
image order. Merged tokens require a separate contract; this fixture does not measure decoder-performance impact."
Part 3 measures that impact (first table). Scope of the damage:

- affected: every pooled 30x30 vision layer cached in August (pooled from the mis-ordered grid) and every full-grid
  layer read before 2026-10-13, i.e. E01's vision IoUs and layer rankings and the layer-11 half of E02's decoder input;
- not tested: the merged visual embedding (one token per 2x2 group).

## Part 3: final cache campaign (2026-10-13)

One last, broad analysis of the cached Qwen representations before deleting them, run on a single rented GPU under
a fixed budget (`data/protocol.json`): corrected-layout screens of all 27 vision layers, readouts and decoders with
matched controls (shuffled labels, spatial mean, displaced features, wrong and swapped references, random and
zero/mean query states), a query-to-patch semantic adapter, temporal adapters with ordered / reversed / shuffled
neighbours, matched DINO / Qwen crops, and training-size / contact-weight / capacity ablations. Selection used
development folds only; every package ran whether or not anything passed.

- **Completion:** `RUN_COMPLETE` 2026-10-13 18:45 UTC; 636 jobs; a findings archive of 25,427 files was verified by
  checksum, after which the bulk feature caches were retired (the cached Qwen layers can no longer be replayed).
- **Cost:** about $3.4 in total for compute and storage (difference of the account balance recorded before
  authorisation and before deletion), inside the $20 authorised.
- **Labels:** the armbar clip's conservative pseudo-labels plus one manually checked trapped-arm contact frame;
  four short clips of private product footage with sparse AI-reviewed labels, each held out whole in turn
  ("held-out clips"); their numbers are withheld here. Unknown pixels are ignored, so IoUs are over labelled cells only.

Groups: 215 ({'neural': 23, 'linear': 188, 'paired': 4}); temporal cells: 90.

#### Layout: the same linear probe on legacy vs corrected token order (armbar, 128-channel projection, 1 seed)
| vision layer | full grid, legacy reshape | full grid, corrected | pooled 30x30, legacy | pooled, re-pooled corrected |
|---|---|---|---|---|
| 0 | 0.270 | 0.410 | 0.427 | 0.531 |
| 1 | 0.210 | 0.357 | 0.403 | 0.535 |
| 2 | 0.193 | 0.353 | 0.383 | 0.570 |
| 3 | 0.188 | 0.353 | 0.343 | 0.551 |
| 4 | 0.190 | 0.360 | 0.303 | 0.545 |
| 5 | 0.185 | 0.385 | 0.356 | 0.628 |
| 6 | 0.209 | 0.413 | 0.349 | 0.620 |
| 7 | 0.215 | 0.438 | 0.391 | 0.615 |
| 8 | 0.232 | 0.459 | 0.450 | 0.627 |
| 9 | 0.236 | 0.485 | 0.471 | 0.610 |
| 10 | 0.246 | 0.507 | 0.475 | 0.692 |
| 11 | 0.248 | 0.506 | 0.473 | 0.633 |
| 12 | 0.238 | 0.502 | 0.418 | 0.636 |
| 13 | 0.223 | 0.518 | 0.387 | 0.676 |
| 14 | 0.215 | 0.532 | 0.351 | 0.676 |
| 15 | 0.208 | 0.498 | 0.334 | 0.650 |
| 16 | 0.207 | 0.508 | 0.376 | 0.682 |
| 17 | 0.222 | 0.491 | 0.363 | 0.648 |
| 18 | 0.211 | 0.505 | 0.345 | 0.638 |
| 19 | 0.193 | 0.493 | 0.338 | 0.640 |
| 20 | 0.202 | 0.470 | 0.348 | 0.639 |
| 21 | 0.202 | 0.501 | 0.346 | 0.653 |
| 22 | 0.190 | 0.482 | 0.339 | 0.645 |
| 23 | 0.178 | 0.493 | 0.348 | 0.661 |
| 24 | 0.201 | 0.459 | 0.350 | 0.632 |
| 25 | 0.203 | 0.474 | 0.360 | 0.672 |
| 26 | 0.159 | 0.413 | 0.296 | 0.650 |
| **mean of 27** | **0.210** | **0.458** | **0.375** | **0.628** |

#### Linear readouts by input (armbar actor IoU, 3 seeds unless noted)
| input | armbar |
|---|---|
| RGB only | 0.172 |
| geometry only (actor centres) | 0.269 |
| Qwen layer 11 (corrected) | 0.503 |
| Qwen merged embedding | 0.563 |
| layer 11 + merged | 0.628 |
| layer 11, all 1,152 channels (1 seed) | 0.723 |
| control: layer 11 + merged, shuffled labels | 0.148 |

#### Small neural decoders on layer 11 + merged (80 epochs)
| decoder | armbar actor IoU | armbar contact acc. | armbar contact margin |
|---|---|---|---|
| appearance only (no language states) | 0.822 | 0.074 | -0.843 |
| global semantic fusion (real states) | 0.823 | 0.086 | -0.815 |
| query-to-patch adapter, real states | 0.820 | 0.074 | -0.867 |
| query-to-patch adapter, matched random states | 0.813 | 0.012 | -0.953 |

#### Contact cells fixed / harmed against the appearance-only decoder (all frames, seeds, clips)
| cohort | candidate | contact cells fixed | contact cells harmed |
|---|---|---|---|
| armbar | query-to-patch, matched random | 0 | 5 |
| armbar | global fusion, real states | 1 | 0 |
| armbar | query-to-patch, real states | 0 | 0 |

#### Temporal adapters: does frame order matter? (mean actor IoU over cells)
| cohort | cells | pass the 10% error-reduction screen | ordered | reversed | shuffled | best simple baseline |
|---|---|---|---|---|---|---|
| armbar | 18 | 0 | 0.350 | 0.357 | 0.353 | 0.323 |

#### Limitations recorded by the campaign
- Exploratory reused source clips and sparse pseudo/AI-reviewed labels, not independent ground truth.
- Reference-frame appearance supervision is assisted, and reference frames are not scored.
- Old pooled tensors retain the original spatial grouping error; only raw full tensors can be correctly repooled.
- Most readouts use a fixed 128-channel projection; dedicated 64/256/native L11 checks test compression sensitivity.
- Causal temporal models never consume full-video semantic states. Hidden-center recovery is not validation of physically invisible anatomy.
- Raw scores and risk/coverage are not calibrated probabilities. Do not infer accuracy over ignored pixels.
- The four held-out clips are private, share venue/context and were studied previously. Seeds are not new independent videos.
- Per-cell gates are screening diagnostics, not campaign-level success claims; inspect matched controls, all sources and all seeds.

## Reading the tables

- **The layout bug was real and large.** With the identical probe, correcting the token order more than doubles the
  full-grid IoU (mean over 27 layers 0.210 -> 0.458) and lifts the pooled IoU (0.375 -> 0.628). Pre-October spatial
  results are void as measurements of what the layers contain.
- **Language states still add nothing.** On corrected features, the best decoder with real Qwen actor states
  (0.823) equals the decoder with no language states (0.822); the query-to-patch adapter with real states (0.820)
  sits next to the same adapter fed matched random states (0.813). On the trapped arm every decoder stays on the wrong
  side (contact margin -0.82 to -0.95). On the private held-out clips, real states fixed about as many contact cells as
  they broke.
- **Nothing transfers to unseen clips.** On the four private held-out clips (numbers withheld), every Qwen input
  and decoder scored at about the level of a geometry-only baseline, while within the armbar clip the same features
  reach 0.82. A quarter of the training frames did as well as all of them.
- **Order does not matter.** On the armbar clip, ordered / reversed / shuffled neighbours give 0.350 / 0.357 / 0.353
  (none passes the 10% screen); on the held-out clips, adapters beat simple baselines on many cells, but reversed and
  shuffled neighbours scored the same as ordered ones. Whatever helps is not temporal reasoning.
