---
name: Fix inference outputs
overview: Explain why the current inference run produced empty generations and no mesh artifacts, then outline the targeted code changes needed to restore usable predictions and mesh export.
todos:
  - id: fix-assistant-extraction
    content: Replace the broken assistant-span extraction in `cad_rl/pipelines/inference.py` with the working logic from `examples/inference_cad_model.py`.
    status: completed
  - id: rerun-inference
    content: Rerun `infer.py` to regenerate `outputs/inference_fusion360_test_mesh_1000.jsonl` with non-empty model outputs.
    status: completed
  - id: choose-output-shape
    content: Decide whether to keep the current two-step JSONL -> mesh workflow or also emit per-sample prediction text files under `predictions/` like the DeepCAD example.
    status: completed
  - id: materialize-meshes
    content: Run `build_meshes.py` on the regenerated JSONL and write STL outputs to the desired `predictions/...` directory.
    status: completed
isProject: false
---

# Fix Inference Outputs

## What is happening

- `[/scratch/498rustam/cad_refine_m/infer.py](/scratch/498rustam/cad_refine_m/infer.py)` only runs model inference and writes JSONL records to the `--output` path.
- `[/scratch/498rustam/cad_refine_m/cad_rl/pipelines/inference.py](/scratch/498rustam/cad_refine_m/cad_rl/pipelines/inference.py)` hardcodes `"execution": {"status": "not_run"}`, so no generated CAD code is executed in this step.
- Mesh files are only materialized by `[/scratch/498rustam/cad_refine_m/build_meshes.py](/scratch/498rustam/cad_refine_m/build_meshes.py)`, which calls `[/scratch/498rustam/cad_refine_m/cad_rl/pipelines/mesh.py](/scratch/498rustam/cad_refine_m/cad_rl/pipelines/mesh.py)` to write `output_dir/meshes/sample_*.stl`.
- `[/scratch/498rustam/cad_refine_m/predictions/deepcad_test_mesh_1000/00003166.txt](/scratch/498rustam/cad_refine_m/predictions/deepcad_test_mesh_1000/00003166.txt)` is an example of prediction text output, not a generated STL mesh.
- That `deepcad` layout is produced by `[/scratch/498rustam/cad_refine_m/examples/inference_cad_model.py](/scratch/498rustam/cad_refine_m/examples/inference_cad_model.py)`, which writes one `.txt` or `.dsl.py` file per sample into `--out_dir`.

## Root cause to fix

- `[/scratch/498rustam/cad_refine_m/cad_rl/pipelines/inference.py](/scratch/498rustam/cad_refine_m/cad_rl/pipelines/inference.py)` has a bug in `_extract_assistant_text()`.
- It currently loops with `for end_tag in part:`, which iterates characters from the decoded text instead of checking for the real `"<|im_end|>"` marker.
- That can turn valid model output into `""`, which matches the empty `raw_generation` and `wrapped_code` values in `[/scratch/498rustam/cad_refine_m/outputs/inference_fusion360_test_mesh_1000.jsonl](/scratch/498rustam/cad_refine_m/outputs/inference_fusion360_test_mesh_1000.jsonl)`.
- The correct implementation already exists in `[/scratch/498rustam/cad_refine_m/examples/inference_cad_model.py](/scratch/498rustam/cad_refine_m/examples/inference_cad_model.py)`.

## Proposed implementation

- Update `_extract_assistant_text()` in `[/scratch/498rustam/cad_refine_m/cad_rl/pipelines/inference.py](/scratch/498rustam/cad_refine_m/cad_rl/pipelines/inference.py)` to match the working logic from the example script.
- Re-run `infer.py` so `outputs/inference_fusion360_test_mesh_1000.jsonl` contains actual generated CadQuery code.
- Keep the existing package workflow as the default:
`infer.py` writes JSONL first, then `build_meshes.py` materializes STL files.
- If the goal is to mirror the `deepcad` folder layout exactly, add an optional output mode later that writes per-sample `.txt` predictions under `predictions/...` in addition to JSONL.
- Run `build_meshes.py --input outputs/inference_fusion360_test_mesh_1000.jsonl --output-dir predictions/fusion360_test_mesh_1000` if you want STL files under `predictions/` specifically.
- Verify that the regenerated JSONL contains non-empty `raw_generation` values and that `predictions/fusion360_test_mesh_1000/meshes/sample_*.stl` is created.

## Notes

- The `vtkXOpenGLRenderWindow` warning in `[/scratch/498rustam/cad_refine_m/slurm/inference.err.log](/scratch/498rustam/cad_refine_m/slurm/inference.err.log)` does not explain the empty JSONL rows.
- The current run likely completed inference, but the buggy post-decode extraction discarded the assistant content.
- The current `fusion360` job did not fail to save meshes because of Slurm; it used a pipeline that does not export meshes during `infer.py`.

