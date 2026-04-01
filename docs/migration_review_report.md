# Migration Review Report

Date: 2026-03-31

Scope:
- reviewed `docs/design_doc.md`, `docs/PLAN.md`, `docs/IMPLEMENTATION.md`
- reviewed the migrated package/CLI/config layout under `cad_rl/`, `configs/`, and root CLIs
- reviewed repo rules in `AGENTS.md`
- ran the focused verification gate: `pytest tests/test_config_runtime.py tests/test_specs_and_artifacts.py tests/test_training_contracts.py tests/test_dataset_boundary.py`
- reviewed checked-in SLURM evidence under `slurm/`
- split the review into parallel tracks for architecture, rules/tests, and logs/evidence before synthesis

## Overall Judgment

I agree with the migration plan substantially more than with the claim that the implementation is fully complete.

- Plan agreement: about 85%
- Implementation agreement: about 60%

The high-level cutover is directionally right and most of the target repo shape is present. The main disagreement is with the docs claiming completion and sufficient proof. The codebase now matches the intended flat layout well enough, but there are still contract gaps, at least one direct repo-rule violation, and weak operational evidence.

## Findings

### 1. Inference does not resolve symbolic checkpoints through the runtime registry

`docs/design_doc.md` says filesystem artifacts should be the source of truth for stage handoffs. The current inference path does not do that for checkpoints.

Evidence:
- `docs/design_doc.md:31`
- `cad_rl/inference.py:54`
- `cad_rl/inference.py:142`
- `cad_rl/training.py:203`

Why this is a problem:
- The demo config uses `checkpoint: latest`, but `resolve_checkpoint_reference()` turns that string directly into `CheckpointRef(path="latest")`.
- `run_inference_from_resolved()` then passes that string straight into model loading.
- Unlike resume training, inference never calls `RunRegistry` or `select_checkpoint()`, so `latest` and `best` are not resolved from runtime artifacts.

Assessment:
- The stage-module split is correct.
- The runtime handoff is not complete.

### 2. Resume semantics are not convincingly finished, despite the docs claiming artifact-driven resume state

`docs/design_doc.md` says filesystem artifacts should be the source of truth for resume state and that train/resume should materialize run metadata and resume from checkpoint. `docs/PLAN.md` repeats that as an acceptance criterion.

Evidence:
- `docs/design_doc.md:35`
- `docs/design_doc.md:160`
- `docs/PLAN.md:30`
- `resume_train.py:18`
- `cad_rl/training.py:176`

Why this is a problem:
- `resume_train.py` still exposes an ad hoc `--checkpoint` flag instead of staying within the uniform CLI contract from the design doc.
- `train_from_resolved_config()` materializes a run directory first, then resolves the resume checkpoint from the registry for `run_id` derived from the current config. There is no end-to-end test showing resume against an existing run manifest/index actually works.
- The current tests validate checkpoint helper functions in isolation, not the real resume flow.

Evidence of limited test coverage:
- `tests/test_config_runtime.py:160`
- `tests/test_config_runtime.py:211`
- `tests/test_training_contracts.py:25`

Assessment:
- The plan is right.
- The implementation is not yet proven complete for the most important artifact-driven behavior.

### 3. The migration still violates a repo rule on imports inside `cad_rl/`

`AGENTS.md` requires absolute package imports inside `cad_rl/`. The migrated rendering module still uses a non-package import.

Evidence:
- `AGENTS.md:4`
- `cad_rl/data/render.py:6`

Why this matters:
- `from visualization_iso import Plotter as _BasePlotter` relies on import-path behavior instead of `cad_rl.data.visualization_iso`.
- This is exactly the kind of cwd-sensitive import the repo rules said to avoid.
- It also weakens confidence that the dataset-prep boundary was migrated carefully rather than just rearranged.

Assessment:
- This is a direct disagreement with the implementation quality, even though it is small and local.

### 4. The evaluation/comparison handoff is structurally weaker than the design intent

The design says stage handoffs should stay explicit on disk and comparison should consume evaluation summaries cleanly. The current summary/report contract drops important identity data.

Evidence:
- `docs/design_doc.md:17`
- `docs/design_doc.md:214`
- `cad_rl/evaluation.py:80`
- `cad_rl/comparison.py:43`

Why this matters:
- Evaluation summary JSON does not include `run_id` or checkpoint identity.
- `compare_summaries()` therefore falls back to the summary file path as the run identifier and leaves `checkpoint_refs=()`.
- That is workable for demos, but it is weaker than the runtime-record model described in the design and makes comparison outputs less trustworthy for real runs.

Assessment:
- This part is only partially migrated. The modules are split correctly, but the artifact contract is not as strong as the docs imply.

### 5. The completion claim is stronger than the available operational evidence

The docs say the migration is complete and verified. The focused pytest gate does pass now, but the only checked-in runtime log is from the legacy tree and shows inference failing.

Evidence:
- `docs/PLAN.md:3`
- `docs/IMPLEMENTATION.md:54`
- `slurm/inference.err:3`
- `slurm/inference.err:11`

Observed verification:
- Focused pytest gate run locally in this review: `12 passed, 4 skipped`

Why this matters:
- The passing tests are real, but they are mostly structural and dry-run oriented.
- Several of the more meaningful checks are skip-based, so the suite does not prove the heaviest migrated paths in a minimal environment.
- The checked-in SLURM evidence is stale and points at the old layout (`cad_rl/data/helper_visu.py`, `cad_rl/data/inference_dataset.py`), so it does not support the current “migration complete” claim.
- There is no checked-in end-to-end evidence for train -> infer -> build_meshes -> evaluate -> compare on the new architecture.

Assessment:
- I disagree with the confidence level in `docs/PLAN.md` and `docs/IMPLEMENTATION.md`.

### 6. Raw inference is still coupled to `cad_rl.data` rendering helpers

The rendering code is physically contained inside `cad_rl.data`, which is good. But the raw STL inference path still depends on that rendering stack.

Evidence:
- `docs/design_doc.md:26`
- `cad_rl/inference.py:147`
- `cad_rl/data/datasets.py:16`
- `cad_rl/data/datasets.py:196`

Why this matters:
- `load_inference_dataset()` can instantiate `RawSTLInferenceDataset`, which directly uses `Plotter1_1`.
- That keeps the code under the correct package boundary, but it also means inference still depends on rendering behavior whenever `hf_dataset` is false.
- I read that as only partially aligned with the stricter design language that dataset-prep-only rendering helpers should not leak into runtime stages.

Assessment:
- This is not as severe as the checkpoint issue.
- It is still a sign that the boundary cleanup is incomplete.

## What I Agree With

- The repo shape now matches the design doc closely: flat `cad_rl` stage modules exist, legacy package layers were removed, and config files moved to `configs/<experiment>/common.yaml` plus stage overlays.
- The execution/metrics split is materially improved. `cad_rl.execution` now owns CadQuery execution, while `cad_rl.metrics` consumes meshes/execution outputs instead of executing code itself.
- The focused tests are aligned with the migration goals and currently pass: layout, config resolution, dry-run CLI behavior, and the rendering-boundary checks are all covered at least at smoke-test level.
- Multiple independent review passes converged on the same overall assessment: the structural refactor is real, but the completion claim is ahead of the runtime proof.

Key evidence:
- `docs/design_doc.md:37`
- `cad_rl/execution.py:145`
- `cad_rl/metrics.py:118`
- `tests/test_dataset_boundary.py:51`

## Bottom Line

The weaker agent got the architecture direction mostly right and completed a meaningful amount of the mechanical flattening. I do not agree that the migration is fully done in the stronger sense implied by `docs/PLAN.md`.

If this were a code-review verdict, I would call it:
- architecture refactor: mostly successful
- production-readiness claim: overstated
- verification claim: partially supported

The clean summary is: the plan is sound, the implementation is materially improved, but the “done” claim is premature.
