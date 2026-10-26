# Experiments

One folder per experiment, numbered in time order. `README.md` is the summary; `RESULTS.md` is the curated record it
cites line by line; `data/` holds result files where they exist. These experiments were not pre-registered as a
whole; where a run had gates fixed in advance, its record says so.

| ID | Question | Verdict |
|---|---|---|
| [E00](E00-tool-chain/) | Can an off-the-shelf chain (frontier VLM -> SAM -> PromptHMR) say whose forearm is trapped in an armbar? | **fail**: each hand-off loses the owner |
| [E01](E01-qwen-ownership-probe/) | Do Qwen3.8's actor states tell a probe which pixels belong to which athlete? | **negative**: an identity code, no better than random |
| [E02](E02-ownership-decoders/) | With a real decoder and controls, do the states fix the trapped arm? | **negative**: 0/27 trapped-arm cells; 6 of 7 follow-up gates fail |
| [E03](E03-audits-final-cache/) | Were the inputs right? Final corrected-layout campaign over the whole cache | **negative, final**: 14,080/14,400 tokens were misplaced; with them fixed, language states still add nothing and nothing transfers to unseen clips |
| [E04](E04-tangle-tracker/) | Can a video tracker carry ownership through a tangle? | **pass, modest**: Harmony4D 0.845 -> 0.882; blind 13-2 |
| [E05](E05-dot-corrections/) | How far do a reviewer's dots carry as tracker corrections? | **in-sample only**: held-out 75 -> 90%, blind 28-6, but no model improved |
| [E06](E06-other-ownership-cues/) | Sapiens parts, SAM 3D Body depth order, outfit colour: does any cue fix ownership? | **negative**: none adopted |
| [E07](E07-what-it-led-to/) | Why the thread stopped and where it went | engagement measured in 3D ([engage3D](https://github.com/dhruvinci/engage3d)) |
