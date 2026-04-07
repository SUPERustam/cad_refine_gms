# AGENTS

- Follow KISS principle
- Root Python files are public CLIs only. Keep a single root entrypoint: `cli.py`.
- Reusable code lives under `cad_rl/`. Do not add new root helper modules.
- Keep the flat stage-module layout under `cad_rl/`: `config.py`, `runtime.py`, `modeling.py`, `grpo.py`, `training.py`, `inference.py`, `execution.py`, `metrics.py`, `evaluation.py`, `comparison.py`, plus `cad_rl/data/`.
- Do not reintroduce legacy package layers or forwarding packages such as `cad_rl.specs`, `cad_rl.pipelines`, `cad_rl.models`, `cad_rl.algorithms`, or package splits that duplicate the current flat modules.
- Use absolute package imports inside `cad_rl/`; never rely on the current working directory.
- Dataset-prep-only rendering code stays under `cad_rl.data`. Optional `vis_for_norm_parts` must not leak into training, inference, evaluation, or runtime modules.
- Keep `cad_rl.data` as the only boundary that touches `Plotter1_1` and dataset-prep rendering helpers. Inference may load raw STL datasets through `cad_rl.data.datasets`, but runtime-stage modules should not import rendering helpers directly.
- Keep config-driven workflows intact around the active dataclass model in `cad_rl.config`. `RunConfig` remains the main resolved run contract, with grouped sections for task, data, model, prepare, train, infer, mesh, eval, compare, runtime, and system.
- Keep stage configs self-contained. Resolve one YAML per command invocation; do not reintroduce implicit `common.yaml` merges, `system_profile`, or a CLI `--system` overlay path.
- Keep config-driven workflows intact. Prefer updating profiles in `configs/` and package code over adding ad hoc scripts.
- Runtime artifacts under `cad_rl.runtime` are the source of truth for run directories, manifests, checkpoint records, stage handoffs on disk, and run-scoped logs. Prefer extending those records over introducing parallel state-tracking paths.
- Preserve a thin root CLI that resolves config, optionally prints a dry-run contract, and delegates to package entrypoints. Keep the workflow surface as subcommands on `cli.py`: `prepare-dataset`, `train`, `resume-train`, `infer`, `build-meshes`, `evaluate`, `compare-runs`. Keep `--config`, `--dry-run`, and `--debug`; `resume-train` keeps the user-facing `--checkpoint` selector. If resume or debug behavior changes, update the CLI and runtime contract together.
- When touching inference checkpoint handling, keep it aligned with runtime artifacts. Symbolic checkpoint names like `latest` and `best` should be resolved through the runtime registry rather than treated as literal model paths.
- When touching evaluation/comparison artifacts, prefer explicit run and checkpoint identity in summaries and reports. Avoid deepening fallback-to-summary-path behavior.
- Focused verification: `pytest tests/test_config_runtime.py tests/test_specs_and_artifacts.py tests/test_training_contracts.py tests/test_dataset_boundary.py`.
