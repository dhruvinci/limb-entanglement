# E07 record: how the thread ended (curated)

Curated from the private workspace's dated notes (2026-10-14 to 26) and the public engage3D repository. Paraphrased;
no new measurements.

## 2026-10-14: the goal restated
While approving another correction round, Dhruva said the work had lost sight of the goal: a computer-vision system
that can segment **and understand** human engagement in situations like these. Masks are a means. The asks:
- understand the configuration: who is where, which limbs are entangled with which;
- produce quantifiable measures of engagement;
- find out which representation serves that: 2D skeletons, SAM 3D Body, Sapiens heads beyond segmentation, or masks.

The work in flight was finished first (E05 round 3, E06).

## 2026-10-22: the dot rounds judged
Dhruva's assessment (also in E05): the dot rounds had polished masks with human labels on the same 11 clips and had
not improved any model or pipeline; on a clip nobody labelled, none of it does anything. Rule adopted: reviewer dots
are a test set, or training data for a model that runs on new clips; report gains on undotted clips or on ground
truth. The tangle thread was paused. The last candidate (enforced corrections, E05) was never reviewed blind.

## 2026-10-22 onward: engagement as 3D relations
Scope set by Dhruva: any close two-person interaction, not only BJJ; contact points and surfaces as well as loops.
- Engagement is measured as relations between two 3D bodies: contact between 16 body regions, distance, and
  **loop threading** (a limb or torso passing through the other person's arm or leg loop). "Whose arm is it?"
  becomes a measurable relation instead of a pixel label.
- Truth: Harmony4D multi-view 3D for standing grappling, expert position labels (ViCoS) for BJJ ground work.
- Every experiment pre-registered before it runs. Published as **engage3D**:
  https://github.com/dhruvinci/engage3d

## What carried over from this thread
- Harmony4D as ground truth for two-person ownership (`tangle_h4d.py`, rendering both SMPL bodies with the nearer
  one owning the pixel) is reused in engage3D.
- Blind A/B review and dot-level scoring: reusable ways to judge ownership without trusting one's own eye.
- Audit the coordinate contract with a fixture before trusting a spatial number (E03).
- The error that remains is limbs inside tangles. In engage3D's terms: after placement is fixed, "what remains is
  limb error inside tangles" (engage3D README, October 2026).
