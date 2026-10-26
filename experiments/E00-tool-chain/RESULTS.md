# E00 results: tool-chain diagnostics on one armbar clip (curated record)

Curated from the private workspace ledger `Qwen3.8-VL-SEG/EXPERIMENT_LOG.md` (2026-08-26). Edits: links to local
run folders removed (the runs show frames of a third-party clip and are not published). Numbers unchanged.
Experiment names E0-E6 below are the ledger's own; in this repo they are all part of E00.

# Qwen3.8-VL-SEG Experiment Log and Research Direction

Date: 2026-08-26  
Status: initial grounding, segmentation, temporal propagation, static 3D, and temporal 3D audits complete; Qwen representation study remains the main research direction

## 1. Research question

The project asks whether a multimodal model's semantic understanding can improve visible-instance segmentation when two humans overlap so heavily that conventional segmenters confuse limb ownership.

The target is not merely to detect two people. It must answer the counterfactual query:

```text
same pixels + query for A1 -> A1's visible mask
same pixels + query for A2 -> A2's visible mask
```

The armbar clip is a useful hard case because a competent viewer can recognize the action, maintain athlete identity through inversion and rolling, and infer that the trapped forearm belongs to A2. Existing mask and body-reconstruction tools frequently assign that forearm to A1 or lose other limbs while trying to correct it.

## 2. Experiment ledger

### E0 — Five-frame reference and Qwen3.8 grounding harness

**Why:** Test the least expensive premise first: can `qwen3.8-27b` enumerate, identify, describe, and box the two athletes consistently before attempting a segmentation decoder?

**Method and tools:**

- extracted five original-resolution frames at 0.75, 1.50, 2.50, 3.50, and 5.95 seconds;
- created and approved visible-extent A1/A2 reference boxes and appearance cues;
- implemented fixed minimal, appearance-conditioned, relational, ambiguity-aware, and paraphrase prompt conditions;
- implemented strict JSON validation, reference hashing, token/spend guards, response persistence, and offline scoring;
- attempted the official Alibaba Cloud Model Studio `qwen3.8-27b` endpoint.

**Finding:** The endpoint was reached but rejected before inference with `AccessDenied.Unpurchased`. No Qwen output or billable result was produced. The harness, annotations, and gate remain usable when access is restored.

**Decision:** Do not treat the provider block as evidence against the architecture. Continue with local/tool baselines while preserving the original Qwen phase gates.


### E1 — SAM2.1 box-to-mask baseline

**Why:** Establish what a strong conventional segmenter can recover from approved actor boxes alone and identify the hard frames worth semantic conditioning.

**Method and tools:** Official `facebook/sam2.1-hiera-small`, pinned checkpoint revision, two box-prompt masks per frame, raw and logit-exclusive overlap variants.

**Finding:**

- 0.75 seconds: clean separation;
- 1.50 seconds: mostly useful, with contact overlap;
- 2.50 and 3.50 seconds: large ambiguous shared regions;
- 5.95 seconds: A2 mask unreliable;
- predicted mask confidence was not a trustworthy indicator of actor ownership under severe overlap.

**Decision:** Box prompts alone are an adequate easy-frame baseline but cannot establish limb ownership. Freeze 2.50, 3.50, and 5.95 seconds as the highest-value hard cases.


### E2 — OpenAI identity grounding → SAM3 and SAM3.1

**Why:** Replace human production boxes with a fully automated semantic front end and determine whether a frontier VLM can maintain actor identity through the sequence well enough to prompt newer SAM versions.

**Method and tools:**

- GPT-5.6 Sol identity pass over five ordered frames;
- one structured geometry pass per frame with A1/A2 boxes, two actor-positive points, and two opponent-negative points;
- exact replay through Transformers SAM3 and the official SAM3.1 tracker;
- logit-exclusive overlap resolution, prompt-adherence checks, masks, and visual contact sheets;
- no human boxes, masks, or descriptions entered the automated prompt path.

**Finding:**

- OpenAI maintained identity across all five frames and produced schema-valid geometry;
- both SAM versions were strong at 0.75 and 2.50 seconds;
- SAM3.1 recovered substantially more of the occluded carrier at 1.50 seconds;
- neither backend separated the maximally entangled 3.50-second frame reliably;
- the 5.95-second output obeyed its sparse prompts but did not resolve ownership of the intertwined limbs;
- SAM3.1 was not uniformly better than SAM3.

Human review agreed that the improved third frame was close to correct, the fourth remained anomalous, and the fifth attributed the trapped arm to the wrong athlete.

**Decision:** Semantic identity and boxes work; the semantic-to-dense-mask bridge is the bottleneck. Test signed limb prompts and time before adding model combinations.

**Cost:** Approximately $0.388 for the completed OpenAI grounding calls.


### E3 — SAM3 semantic-prompt and candidate-supply audit

**Why:** Separate three possible causes of the final trapped-arm error:

1. OpenAI understands the armbar but translates the relation to the wrong pixel;
2. SAM receives the correct sign/point but cannot supply the desired mask;
3. SAM supplies a valid candidate but the selection rule chooses the wrong one.

**Method and tools:** Box-only, identity-core, semantic-arm, sign-swapped null, and manually inspected ceiling prompts; every exposed SAM hypothesis retained; short text controls (`person`, `athlete`, `grappler`, and `person grappling on the mat`); frozen evaluation-only arm/core probes.

**Finding:**

- the saved OpenAI arm point was on A2's torso, not the trapped forearm;
- SAM changed 77,479 ownership pixels when the signed point was added, proving the interface was responsive;
- none of the three A2 hypotheses covered all three trapped-arm probes;
- the sign-swapped null reverted toward the core-only result;
- even the inspected ceiling point did not produce a clean candidate;
- long semantic text was a weak candidate-supply mechanism and did not define stable actor identity.

**Decision:** The failure is not a pure SAM bug and not a pure language-understanding bug. It is jointly a semantic-to-spatial translation error and a candidate-supply limitation.


### E4 — SAM3.1 temporal propagation pilot

**Why:** Test the easy hypothesis that more frames and tracker memory might carry clean early identity evidence through the entangled armbar position.

**Method and tools:** Official SAM3.1 tracker at source resolution; 38-frame 6 fps pilot; final-only, early-forward, pre-entanglement-forward, bidirectional, and one-correction arms; independent actor states with a shared frame/backbone cache; frozen identity/core/arm gates.

**Finding:**

- final-only retained identity cores but recovered only 1/3 trapped-limb probes and showed cross-actor speckling;
- every propagated arm lost the actor masks before the final frame;
- denser temporal context did not correct the ownership error;
- tracking confidence decayed during the high-motion lift-and-roll transition;
- no arm passed the frozen gate, so the experiment correctly stopped before an unjustified full-rate winner.

**Decision:** Frame density alone is not the missing ingredient. Temporal identity must be represented semantically or learned jointly, rather than trusting mask propagation through a severe topology change.


### E5 — Static PromptHMR prompt audit

**Why:** Test whether a learned human-body and interaction prior can recover plausible ownership even when the visible mask is locally wrong.

**Method and tools:** PromptHMR model core called directly, bypassing its detector/tracker; frozen OpenAI boxes and camera; box-only, SAM3-mask, SAM3.1-mask, and swapped-mask null arms; interaction prompt enabled; SMPL-X projection, filled-mesh IoU, trapped-arm probe distances, and prompt-preprocessing checks.

**Finding:**

- at 5.95 seconds, box-only PromptHMR assigned the arm more closely to A1;
- SAM3-mask + PromptHMR moved A2's projected forearm across the trapped hand/wrist probes (42.91 px), despite the source SAM mask assigning those pixels to A1;
- SAM3.1 also recovered relative A2 ownership, but less accurately (85.87 px);
- swapping the SAM3 identities visibly corrupted roles and worsened the arm distance, demonstrating dependence on actor-specific prompts;
- every arm collapsed the two bodies at 3.50 seconds (mesh IoU 0.764–0.869).

**Decision:** Body/interaction priors can override a bad local mask and contain useful ownership information, but single-frame PromptHMR is not robust enough. Proceed to the official temporal head.


### E6 — Full OpenAI + SAM3 + PromptHMR-Video sequence

**Why:** Run the strongest fully automated whole-sequence test: semantic identity and action understanding from OpenAI, dense masks from SAM3, per-frame body features from PromptHMR, and temporal regularization from PromptHMR-Video.

**Method and tools:**

- one GPT-5.6 Sol request over 26 ordered full-resolution frames, then interpolation to 146 A1/A2 box tracks;
- dense source-resolution SAM3 box masks for 292 actor/frame pairs;
- PromptHMR box-only and box+SAM3 image-feature caches, with both athletes processed jointly and interaction enabled;
- official PromptHMR-Video combined BEDLAM1+BEDLAM2 checkpoint;
- zero-keypoint controls, automatically mapped PromptHMR-to-COCO17 translation passes, and static-translation hybrids;
- YOLO11n-pose gate for an independent keypoint arm;
- SMPL-X reconstruction, mesh-overlap, forearm probes, temporal jitter, bone stability, rendered anchors, and a full comparison video;
- image and temporal models staged separately to fit 6 GiB VRAM.

**Finding:**

- OpenAI correctly described A1 developing an armbar on A2 with 0.94 confidence and preserved actor identity;
- PromptHMR-Video eliminated mesh-collapse frames and reduced median bone-length variation from 0.180 to about 0.010;
- raw zero-keypoint temporal output was smooth but badly registered in the camera;
- YOLO yielded valid two-person poses in only 26/146 frames and was rejected by the 90% gate;
- self-conditioned and static-translation fallbacks improved localization but did not reconstruct the final A2 forearm accurately;
- the best localized temporal arm changed relative ownership to A2, but A2's absolute final probe distance worsened from 216.43 to 254.94 px;
- box-only temporal features consistently outperformed SAM3-conditioned features;
- every temporal arm was rejected visually as a solved output.

**Decision:** More temporal information produces real regularization but cannot compensate for a weak person-specific 2D observation channel. The daisy chain has reached a ceiling on this clip. Return to the joint Qwen VL-SEG research rather than adding more independent prompt/detector combinations.

**Cost and compute:** Approximately $0.560 for the 26-image OpenAI request; 4.29 GiB PromptHMR peak allocated VRAM; 190 MiB PromptHMR-Video peak; 69/69 regression tests passing.


## 3. How each tool contributed

| Tool | Intended role | What it demonstrated | Principal limitation here |
|---|---|---|---|
| Qwen3.8 API harness | Target-model semantic grounding | Reproducible protocol and gates are ready | Provider entitlement prevented inference |
| GPT-5.6 Sol / OpenAI Responses | Identity, action semantics, boxes, sparse prompt geometry | Stable identities; correct armbar and trapped-actor interpretation | Arbitrary coordinates/points were not reliably on the intended limb |
| SAM2.1 | Frozen conventional box baseline | Easy-frame masks and hard-frame selection | Merges or confuses overlapping actors |
| SAM3 | Semantic/geometry-prompted visible masks | Prompt responsiveness and good easy-frame masks | No correct trapped-arm candidate under severe overlap |
| SAM3.1 | Newer tracker and temporal propagation | Better carrier recovery at 1.50 seconds; useful temporal controls | Not uniformly better; track decay through high motion |
| PromptHMR | Single-frame human-body and pair prior | Can locally repair mask ownership and respond to actor identity | Severe two-body collapse and unstable whole-body anatomy |
| PromptHMR-Video | Temporal 3D regularization | Smooth pose, stable body shape, fewer collapses | Needs reliable person-specific 2D observations for camera registration |
| YOLO11n-pose | Independent 17-keypoint condition | Good detections in separated frames | Only 17.8% valid two-track coverage in this clip |
| SMPL-X + OpenCV + ffmpeg | Comparable projection, metrics, and visual evidence | Exposed metric wins caused by off-image/misregistered bodies | Rendering evaluates projections, not ground-truth 3D accuracy |
| Frozen probes and null arms | Causal measurement | Distinguished relative ownership, absolute localization, prompt sign, and actor swaps | Sparse probes cannot certify the entire boundary |

## 4. What we now know

1. **Semantic recognition is ahead of dense grounding.** A VLM can understand the action and actor relationship while still placing a point on the torso instead of the trapped forearm.
2. **Prompt compliance is not correctness.** Every positive/negative click can be obeyed while a long contact boundary is assigned to the wrong person.
3. **SAM's failure is not merely selection.** In the final audit, the desired complete A2-arm candidate was absent.
4. **Human-body priors contain ownership information.** PromptHMR recovered A2's forearm locally even from a misleading SAM mask.
5. **Temporal smoothing is not temporal understanding.** PromptHMR-Video stabilized shape and motion but did not place bodies correctly without reliable observations.
6. **Independent modules compound incompatible representations.** Text semantics, masks, 2D joints, and 3D bodies each use different identity/spatial contracts; errors are lost at every handoff.
7. **The central architecture remains justified.** A joint identity-conditioned dense decoder with shared spatial evidence is more promising than additional prompt chaining.

## 7. Current decision

The tool-chain experiments were valuable diagnostics, not a diversion. They established that semantic identity/action understanding is real, dense ownership remains unresolved, body priors can sometimes repair local masks, and temporal regularization alone cannot bridge the mismatch.

The project should now continue toward Qwen3.8-VL-SEG with one added research branch: explicitly test and, if validated, distill the VLM's actor-aware visual-critic capability.


## Later caveat (added for this repo)

On 2026-10-12 a separate harness found that driving the SAM 3.1 multiplex tracker directly, without
`propagate_in_video_preflight`, makes it ignore its seeds after the prompt frame (see E04). An October note in the
workspace also judged that a tracker-only propagation path used at the time "likely invalidates" the August
temporal-audit conclusion (E4 above). Whether E4 ran through the affected path was not re-checked; treat its
negative result as unverified. The other stages (E1-E3, E5-E6) did not use that path.
