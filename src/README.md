# Code

Exported from a private research workspace for reading and re-use. The models are frozen third-party models and the
footage is not redistributed, so most scripts cannot be re-run end to end from this repository alone. What can be
re-run from here: the figures (`limb_demo.py`), the E03 tables (`qwen/final_cache/summarize_findings.py`), the token
layout audit (`qwen/final_cache/layout_audit.py`, needs only the Qwen3.8 processor files) and the Harmony4D tracker
benchmark (`tangle/tangle_bench.py`, needs Harmony4D, SAM 3.1 and a GPU). Each script states its environment and usage
in its docstring.

## `limb_demo.py`
All figures in `docs/assets/`. Every chart value is copied from an experiment's `RESULTS.md` (lines noted in the
code). The Harmony4D panel needs the `tangle_bench.py` work folder:
`python src/limb_demo.py --out docs/assets --h4d-work <work dir>` (matplotlib, numpy, OpenCV).

## `qwen/` (E01-E03)
- `extract/`: the August 2026 extraction of Qwen3.8-27B vision layers, merged embeddings and per-athlete marker
  states (`extraction.py`, `prompts.py`, `contracts.py`) and the ownership probe (`probe.py`, `probe_analysis.py`).
  `pool_spatial_tokens` in `extraction.py` carries the token-layout bug found in E03, marked in place and kept as
  run. Omitted: cloud launch, storage and watchdog modules (`launch_packet`, `artifacts`, ...), which these files
  import.
- `decoder/ownership_decoder/`: the E02 decoders: model (`model.py`), losses, metrics, training, cache readers
  (`cache.py`, same layout bug for full-resolution layers, marked), the armbar campaign definition, the
  actor-alignment and grounding audits. Omitted: label-packet builders (`breadth_labels`) and the local / cloud
  resource controllers.
- `final_cache/`: the E03 campaign: `core.py` (`restore_grid`, the layout fix, and the metrics), `layout_audit.py`,
  `cloud_data.py` (cache reader), `experiment.py` (linear and neural readouts, controls), `models.py`,
  `temporal.py`, `paired.py` (DINO / Qwen crops), `jobs.py`, `runner.py`, `audit.py`, `archive.py`, `report.py`,
  and `summarize_findings.py` (standard library only). Omitted: the cloud worker, budget guard and pod controller.
- Runtime recorded by the final campaign: Python 3.12, PyTorch 2.12.1 + CUDA 13.0, transformers 5.16.1, one NVIDIA
  RTX PRO 6000 Blackwell (96 GB).

## `tangle/` (E04-E06)
Flat modules that import each other by name; run from this folder. Paths: `paths.py` (environment variables
`LIMB_WORKSPACE`, `LIMB_H4D`, `LIMB_SAM31`, `LIMB_S3D`, `LIMB_BMP`, `LIMB_SCRATCH`).
- Tracker: `tangle_core.py` (SAM 3.1 multiplex tracker driven directly: full-mask seeds, pixel competition, memory),
  `tangle_plan.py` (anchors), `autolabel_tangle.py` (the per-interval stage, with `--corrections`, `--corr-radius`,
  `--corr-memory`), `tangle_summary.py` (hand-off QA). Its 2 fps detection scan (`tangle_scan`) is not included.
- Harmony4D benchmark: `tangle_h4d.py` (ground-truth ownership rendered from the two SMPL fits), `tangle_bench.py`,
  `tangle_bench2.py` (oracle anchors, skeleton anchors).
- Review and dots: `tangle_review*.py` (blind A/B packs), `maskfix_ab_build.py`, `maskfix_ab_serve.py`,
  `maskfix_ab.html` (the review page with red / blue / neither dots), `gold_serve.py`, `serve_review.py`,
  `tangle_ab_score.py`, `tangle_dot_eval.py` (score any variant against the dots).
- Corrections and fills (E05): `dotfix.py`, `dotenforce.py`, `holefill.py`, `bonefill.py`, `bmp_tangle.py`
  (BBoxMaskPose v2, GPL-3.0, imported at run time from `LIMB_BMP`; not distributed here), `pointvote.py`.
- Other cues (E06): `sapiens_parts.py`, `s3d_own.py`, `mesh_sam.py`, `autolabel_body3d.py`, `outfit_probe.py`,
  `appearance_probe.py`.
- Shared auto-labeller pieces the tracker uses: `autolabel_common.py`, `autolabel_detect.py`,
  `autolabel_sam31_video.py`, `autolabel_densify.py` (its VLM-labelling import, `autolabel_astra`, is not included).
- Environments: `env/venv-sam31.txt` (tracker), `venv-bmp.txt` (BBoxMaskPose, dots), `venv-s3d.txt`
  (SAM 3D Body, Sapiens), `venv-pose.txt` (review packs, fills), `venv-depth.txt` (figures).
