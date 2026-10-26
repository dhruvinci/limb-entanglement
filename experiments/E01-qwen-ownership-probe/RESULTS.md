# E01 results: Qwen3.8 extraction campaign and ownership probe (curated record)

Curated from the private workspace reports `campaign/analysis/20260827T074622Z/CAMPAIGN_RESULT.md` and
`OWNERSHIP_PROBE_RESULT.md` (2026-08-27 to 28). Edits: cloud account, pod and credential bookkeeping removed;
physical descriptions of the athletes replaced by their roles; figure links removed (they show frames of a
third-party clip). Numbers unchanged.

> **Layout caveat (2026-10-13, see E03).** The raw vision-layer tokens behind the pooled 30x30 and full 160x90
> features below were reshaped in the wrong order: 14,080 of 14,400 grid locations were misplaced. Their absolute
> IoUs and layer rankings are void as spatial measurements. The query findings (semantic states no better than a
> matched random code; the fixed-decoder control) do not depend on that order, and E03 repeated them on corrected
> features with the same outcome.

## Part 1: extraction campaign


Run ID: `20260827T074622Z`

### Operational result

- `RUN_COMPLETE` was written at `2026-08-27T09:57:48.194066+00:00` with no integrity errors.
- The exact full-model canary passed on an NVIDIA RTX PRO 6000 Blackwell Server Edition: SM120, 97,887 MiB, driver 595.91.07, CUDA 13.0, PyTorch 2.12.1+cu130, Python 3.12.3.
- The model revision was `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, loaded in BF16 with batch size one, `use_cache=False`, and `torch.compile` disabled.
- The verified reference kernel path was used because the isolated reduced fast-kernel preflight did not complete safely.
- The successful scientific worker had zero restarts, zero OOMs, zero mandatory failures, and zero non-finite/corrupt artifacts.
- Estimated campaign GPU spend is about $5.18 from active-compute intervals, including the $1.70 preflight estimate. The deliberately conservative upper bound is $6.30, below the $30 authorization.

### Verified artifact inventory

The remote integrity sweep validated 11,204 artifacts totaling 145.96 GiB. Its checksum ledger has SHA-256 `cdf3caeeacb2e10791ee53e7bade82022cac1178a51ad3a8a5a065b6da0b529d`.

- 4,326 full-resolution spatial tensors
- 6,534 pooled spatial tensors
- 242 merged vision tensors
- 62 mandatory temporal semantic-state tensors
- 6 optional 8 fps semantic-state tensors
- 31 mandatory grounding tensors
- 3 optional 8 fps grounding tensors

The local core contains 7,856 unique files. Of these, 7,846 matched entries in the campaign checksum ledger; the remaining 10 are operational metadata files. There were no local size/hash errors and no temporary files after download.

Local tensor hierarchy:

- pooled vision layer: 900 × 1,152 BF16 (30 × 30 spatial grid)
- full vision layer: 14,400 × 1,152 BF16 (160 × 90 spatial grid)
- merged vision embedding: 3,600 × 5,120 BF16 (80 × 45 effective grid after 2 × 2 merge; source `grid_thw` is 160 × 90)
- semantic marker states: 64 × 5,120 BF16 per actor query

### Grounding observations

1. None of the eight armbar action-relational outputs explicitly identified an armbar.
2. Thinking off confidently described a suplex followed by a triangle around the neck and reported no uncertainty. The xhigh run became more cautious, described a shoulder/fireman's-carry-style throw, and correctly admitted that the final hold was unclear, but it still missed the armbar.
3. Thinking xhigh required the deterministic format-only repair. One private-clip contact-ownership answer remained invalid after its single repair because the answer was truncated. Overall grounding contract validity was 33/34.
4. The reversed 4 fps control narrated the reversed physical progression, demonstrating that frame order reaches the language reasoning path. The shuffled control nevertheless reconstructed a plausible forward sequence, so text alone does not establish robust temporal understanding.
5. With no appearance/order hints, the temporary A1/A2 assignment is not canonical across prompts. The armbar 2 fps and five-frame action-relational runs swap the physical athletes relative to most other armbar conditions. The half-guard contact-ownership run also swaps the physical mapping relative to its identity/action runs.

### Marker-state observations

Cosine similarity is used only as a sensitivity diagnostic; lower values mean a larger representation change.

| Comparison | Final-layer cosine | Last-16-layer mean |
|---|---:|---:|
| A1 vs A2, 4 fps action, thinking off | 0.9821 | 0.9827 |
| A1 vs A2, 4 fps action, xhigh | 0.9848 | 0.9860 |
| Thinking off vs xhigh, A1 | 0.9184 | 0.9305 |
| Thinking off vs xhigh, A2 | 0.9189 | 0.9353 |
| Ordered 4 fps vs shuffled, A1 | 0.9672 | 0.9776 |
| Ordered 4 fps vs reversed, A1 | 0.9766 | 0.9803 |
| Ordered 4 fps vs single frame, A1 | 0.9512 | 0.9599 |
| Ordered 4 fps vs 8 fps, A1 | 0.9875 | 0.9896 |

Relational/contact conditioning separates the two actor marker states more than identity-only conditioning in late layers. Thinking mode changes the late marker state much more than the actor swap, but xhigh slightly reduces A1/A2 separation. Single-frame versus ordered-video context creates the largest temporal-context change; 4 fps versus 8 fps is comparatively small.

These results show actor, prompt, thinking, and temporal sensitivity. They do not show that the trapped arm is assigned correctly.


## Part 2: ownership probe


Run ID: `20260827T074622Z`

### Result in one sentence

This is a scientifically useful negative result: frozen Qwen vision features support coarse athlete ownership, but the cached global actor marker states do not repair the trapped-arm assignment, do not outperform vision alone, and do not show an ordered-temporal advantage over reversed or shuffled controls.

The experiment does **not** show that Qwen lacks relational understanding. It shows that the particular interface tested here—one global marker state per actor fused with frozen patch features through a small multiplicative probe—does not expose that understanding as an actor-conditioned spatial field.

Two scope limits matter. The cloud cache contains no per-frame image-conditioned marker states (`semantic_image = 0`); the semantic probe therefore uses global native-video query states shared across every frame. The campaign also cached several breadth clips, but the ownership probe is limited to the one 32-frame armbar label set because the additional clips do not yet have matching reviewed ownership supervision. These omissions are exactly why the result should narrow the next extraction interface rather than be read as a verdict on the full model.

### Supervision and split

- `A1`: the athlete applying the final armbar.
- `A2`: the athlete whose arm is trapped.
- Thirty-two frames from the 146-frame armbar clip were used: 24 original training frames and eight held-out test frames (`18, 36, 60, 84, 96, 120, 136, 143`).
- Layer selection used 20 screen-training frames and four validation frames (`16, 48, 88, 128`). After selection, all 24 original training frames were used for the final fit; the eight test frames remained held out.
- Labels are conservative pseudo-labels, not complete human pixel ground truth. They use reviewed SAM actor cores, prevent SAM holes inside frozen actor boxes from becoming false background, and ignore uncertain regions.
- Frames `68, 72, 76, 80` contribute background supervision only because their actor masks failed visual audit.
- Frame 143 contains 21 explicitly forced A2 patches around three pre-registered trapped-arm points. No other ambiguous contact boundary was guessed.

Consequently, IoU is best interpreted comparatively across conditions using the same grid and split. The trapped-arm A2 margin is the decisive local measure.

### Probe protocol

- Qwen remained frozen.
- A balanced three-class probe predicted background, A1, or A2 from cached patch features.
- The vision-only control used a rank-32 visual projection and linear head.
- Actor-conditioned probes used the same visual projection plus the normalized A1/A2 marker-state residual and shared pair context.
- All 27 pooled vision layers were screened on validation data. Pooled layer 12 was selected under the preregistered action-query policy; vision alone independently peaked at layer 11.
- All 64 language layers were screened with pooled layer 12. Language layer 25 was selected, narrowly ahead of layer 60 and layer 63.
- Full-resolution layers `5, 11, 18, 26` were screened; layer 11 won for vision, action, and matched-random controls.
- Final pooled results use five seeds. Full-resolution results use three seeds.
- Query controls include identity, contact, action, thinking off/xhigh, single/five-frame/2 fps/4 fps/8 fps, reversed, shuffled, matched random, mean, zero, and actor swap.

### Main held-out results

Values are mean ± population standard deviation across seeds. Actor IoU is the mean of A1 and A2 IoU. Positive trapped-arm margin favors the correct owner A2; every nondegenerate condition remained negative.

#### Pooled 30 × 30 features

| Condition | Actor IoU | A1 IoU | A2 IoU | Contact accuracy | A2 contact margin |
|---|---:|---:|---:|---:|---:|
| Vision only | **0.6928 ± 0.0059** | 0.7866 | 0.5990 | 0.1333 | -0.6572 |
| Identity, ordered 4 fps | 0.6742 ± 0.0034 | 0.7578 | 0.5907 | 0.0667 | -0.7014 |
| Contact, ordered 4 fps | 0.6717 ± 0.0087 | 0.7612 | 0.5821 | 0.2667 | -0.5418 |
| Action, ordered 4 fps, thinking off | 0.6653 ± 0.0149 | 0.7551 | 0.5755 | 0.2000 | -0.6787 |
| Action, ordered 4 fps, thinking xhigh | 0.6810 ± 0.0170 | 0.7651 | 0.5969 | 0.2667 | -0.4781 |
| Action, reversed 4 fps | 0.6757 ± 0.0092 | 0.7587 | 0.5927 | 0.2667 | -0.5236 |
| Action, shuffled 4 fps | 0.6740 ± 0.0098 | 0.7650 | 0.5830 | 0.1333 | -0.6101 |
| Action, 8 fps | 0.6729 ± 0.0105 | 0.7596 | 0.5861 | 0.2667 | -0.5203 |
| Matched random actor code | 0.6687 ± 0.0117 | 0.7441 | 0.5932 | 0.2000 | -0.5549 |
| Zero query | 0.2839 ± 0.0015 | 0.5678 | 0.0000 | 0.0000 | 0.0000 |

#### Full 160 × 90 features

| Condition | Actor IoU | A1 IoU | A2 IoU | Contact accuracy | A2 contact margin |
|---|---:|---:|---:|---:|---:|
| Vision only | **0.5283 ± 0.0138** | 0.6182 | 0.4384 | **0.2593** | **-0.4048** |
| Identity, ordered 4 fps | 0.4853 ± 0.0047 | 0.5861 | 0.3844 | 0.2346 | -0.4463 |
| Contact, ordered 4 fps | 0.4834 ± 0.0126 | 0.5811 | 0.3856 | 0.1358 | -0.6420 |
| Action, ordered 4 fps, thinking off | 0.4815 ± 0.0117 | 0.5706 | 0.3924 | 0.2346 | -0.5229 |
| Action, ordered 4 fps, thinking xhigh | 0.4797 ± 0.0040 | 0.5696 | 0.3897 | 0.2099 | -0.5188 |
| Action, reversed 4 fps | 0.4791 ± 0.0038 | 0.5733 | 0.3849 | 0.1975 | -0.5828 |
| Action, shuffled 4 fps | 0.4825 ± 0.0061 | 0.5726 | 0.3923 | 0.1111 | -0.6037 |
| Action, 8 fps | 0.4849 ± 0.0143 | 0.5786 | 0.3912 | 0.1975 | -0.4925 |
| Matched random actor code | 0.4751 ± 0.0057 | 0.5696 | 0.3805 | 0.1975 | -0.4983 |
| Zero query | 0.2193 ± 0.0019 | 0.4386 | 0.0000 | 0.0000 | 0.0000 |

Pooled and full-grid absolute IoUs should not be compared as a resolution benchmark because their label densities and optimization schedules differ. Within each grid, however, vision only is consistently strongest and the semantic conditions track the matched-random control.

### Decisive fixed-decoder control

Training a separate decoder for each query can conceal whether the query itself changes the output. A final control therefore trained one decoder, froze all of its weights, and substituted only the two actor marker states at evaluation time.

With the action-trained decoder, mean actor IoU was:

| Evaluation marker pair | Actor IoU | A2 contact margin |
|---|---:|---:|
| Action 4 fps | 0.6659 | -0.6781 |
| Identity 4 fps | 0.6655 | -0.6748 |
| Contact 4 fps | 0.6648 | -0.6769 |
| Action xhigh | 0.6651 | -0.6774 |
| Reversed | 0.6651 | -0.6776 |
| Shuffled | 0.6651 | -0.6777 |
| Matched random | 0.6639 | -0.1943 |
| A1/A2 swapped | 0.0658 | +0.6781 |

The six real semantic variants span only about **0.0011 IoU** with fixed decoder weights. A1/A2 swapping flips the ownership field, but collapses whole-athlete IoU while making the trapped-arm margin positive. This is the signature of an actor-identity code, not a localized semantic correction.

The chosen layer-25 actor states are also extremely similar within each pair: action A1 versus A2 cosine is `0.998128`. Identity, contact, action, xhigh, reversed, and shuffled pairs are themselves nearly interchangeable to the trained decoder.

### Frame 143

For seed 7 at full resolution:

| Condition | A1 IoU | A2 IoU | Contact accuracy | A2 contact margin |
|---|---:|---:|---:|---:|
| Vision only | 0.732 | 0.383 | 0.222 | -0.568 |
| Identity | 0.699 | 0.278 | 0.333 | -0.383 |
| Action, thinking off | 0.680 | 0.322 | 0.296 | -0.435 |
| Action, thinking xhigh | 0.671 | 0.321 | 0.296 | -0.353 |
| Matched random | 0.681 | 0.244 | 0.333 | -0.249 |

Some semantic conditions make the negative contact margin less severe, but matched random does as well or better. None crosses zero, and all reduce overall actor IoU relative to vision only. The heatmaps show the same failure visually: the trapped forearm remains inside an A1-favoring field.

### Answers to the research questions

1. **Does semantic conditioning fix the trapped-arm ownership?** No, not with the cached global marker-state interface and this probe.
2. **Does thinking xhigh help?** Not robustly. It improves the pooled contact margin, is slightly worse than thinking off at full resolution, and becomes effectively identical when the decoder is held fixed.
3. **Does ordered temporal context beat reversed or shuffled context?** No. Ordered 4 fps is not consistently better in IoU or contact ownership. The language outputs are sensitive to order, but that sensitivity does not become a useful spatial ownership correction here.
4. **Does more spatial resolution fix it?** No. Layer 11 full-resolution features still favor A1 at the trapped arm, and semantic fusion remains worse than vision alone.
5. **Is the original VL-SEG direction invalidated?** No. The result narrows the design: a single final marker token is too global. The next model should couple actor/action queries to patch tokens before or inside the mask decoder, and it should learn that coupling across multiple labeled clips.

### Recommended next architecture experiment

Do not spend another cloud run repeating more prompt variants against the same global marker-state cache. The next decisive experiment should cache or train a genuinely spatial semantic interface:

1. Extract per-frame or short-clip query-conditioned patch states, token-to-patch cross-attention inputs, or the hidden spatial states used immediately before a VL-SEG mask decoder—not only a final global marker token.
2. Keep actor identities canonical across clips, and evaluate query substitution with one fixed decoder as a mandatory control.
3. Use several high-contact clips with human-verified ownership cores and explicit contact-point/region labels; keep entire clips held out, not only frames from the same clip.
4. Train a small cross-attention/projector-plus-decoder first. Only then consider end-to-end Qwen/VL-SEG tuning.
5. Add VideoLISA-style temporal memory after the spatial query coupling works in static frames. Require ordered context to beat reversed and shuffled controls before claiming temporal ownership reasoning.

This stays directly on the Qwen3-VL-SEG research path: the missing piece is not a better natural-language prompt but an architecture that converts relational state into a spatially varying actor query.

### Local OOM incident and recovery

The first full-resolution local probe process was killed by the operating system at about 9.5 GB resident RAM. GPU memory was not the cause. Advanced-indexed CPU subsets were being recreated for every seed and retained by the allocator. The run had already atomically checkpointed 11 completed seeds.

The runner was changed to load one train partition and one evaluation partition, keep them on the GPU when available, and perform batched indexing there. The resumed run completed all remaining seeds with roughly 1.2 GB host RSS and 1.95 GB GPU memory. No completed result was recomputed or lost.

### Data in this folder

- `data/pooled-final.json`, `data/full-final.json`: final metrics (pooled 30x30 and full 160x90 grids).
- `data/cross-query-control.json`: the fixed-decoder query substitution.
- `data/frame-metrics.json`: per-frame metrics.
- The ownership-field figures overlay frames of the third-party clip and are not published.
### Final verification

- `python -m unittest discover -s tests -v`: **136 tests passed**, zero failures, in 75.049 seconds.
- All metric files have their expected record counts: 135 pooled vision-screen records, 64 language-screen records, 70 pooled final records, 12 full layer-screen records, 30 full final records, 80 fixed-decoder records, and 80 per-frame records.
- Every saved pooled, full-resolution, and fixed-decoder logit tensor has the expected shape and contains only finite values.
- The visualization script regenerated all three figures and the per-frame metrics from the saved logits without error.
- The campaign checksum audit remains clean: 11,204 artifacts, 145.96 GiB.
