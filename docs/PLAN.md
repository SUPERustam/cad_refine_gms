# Simplify Config/Runtime/Imports and Add Run Logging

## Summary

- Adopt `import cad_rl.x` style for project-local imports.
- Remove unnecessary wrapper/build helper layers in config, runtime, and training.
- Keep lazy imports only where they protect dry-run paths, heavy optional dependencies, or multiprocessing and inference worker startup.
- Move to one explicit YAML per command invocation with no implicit `common.yaml` merge and no separate `--system` overlay.
- Add run-scoped logging with a real logger and a debug mode.

## Key Changes

- **Config contract**
  - Keep `RunConfig` as the single resolved contract with grouped sections.
  - Require each stage config to be self-contained, including `system` and `runtime`.
  - Remove implicit sibling `common.yaml` loading, `system_profile`, and CLI `--system`.
  - Simplify `cad_rl.config` to: load one file, validate sections, build dataclasses, serialize.

- **CLI surface**
  - Keep existing subcommands on `cli.py`.
  - Keep `--config` and `--dry-run`.
  - Add `--debug` override for runtime logging.
  - Remove `--system`.
  - Dry-run should print the resolved single-file contract without importing heavy training or inference dependencies.

- **Import policy**
  - For repo-local code, prefer `import cad_rl.module` and use qualified access.
  - Move safe project imports to file top.
  - Keep local imports for heavy or optional deps such as `torch`, `transformers`, `trl`, `datasets`, `cadquery`, `trimesh`, and for worker-only or spawn-sensitive paths.
  - Do not force third-party `from x import y` rewrites unless already touched.

- **Runtime and training simplification**
  - Collapse duplicate run materialization paths in `cad_rl.runtime` into one source of truth.
  - Keep checkpoint selection in runtime and use it consistently for both resume-train and inference.
  - Trim thin builder wrappers in `cad_rl.training` where they only rename fields or forward dataclasses.
  - Preserve runtime artifact ownership: run dir, manifest, resolved config, checkpoints, logs.

- **Logging and debuggability**
  - Use Python `logging` as the base logger.
  - Initialize logging once per command with console output plus a run-scoped file under `logs/`.
  - Support debug level from CLI and config; CLI flag wins.
  - Replace ad hoc `print` paths with logger calls.
  - Keep Comet optional through existing training and reporting settings; do not make it required for debugging.

## Public Interface Changes

- `cli.py`
  - Remove `--system`.
  - Add `--debug`.
- Config files
  - Inline `system` instead of external system profile references.
  - No implicit `common.yaml`; every file is standalone.
- `cad_rl.config.resolve_*`
  - Stop accepting `system=...`.
  - Resolve from one file only.
- `cad_rl.runtime`
  - Add a logging setup entrypoint and keep checkpoint resolution APIs as the single path for symbolic checkpoint names.

## Test Plan

- Update config tests to assert single-file resolution only, with no `common_config_path` or `system_config_path`.
- Keep CLI dry-run tests for every subcommand using only `--config`.
- Add or adjust tests that inference resolves `latest` and `best` through `RunRegistry`, not as literal paths.
- Add tests that runtime materialization creates the log directory and log file path predictably.
- Add a debug-mode test that logger level changes without changing the resolved config shape unexpectedly.
- Re-run the focused gate:
  - `pytest tests/test_config_runtime.py tests/test_specs_and_artifacts.py tests/test_training_contracts.py tests/test_dataset_boundary.py`

## Assumptions

- Keep the existing subcommand names and overall workflow unchanged.
- "One YAML per run" means one self-contained YAML per command invocation, not one global pipeline file for all stages.
- Logging stays local-first; Comet remains optional secondary reporting.
- Strict top-level imports are not applied to heavy optional dependencies or spawn-sensitive code.
