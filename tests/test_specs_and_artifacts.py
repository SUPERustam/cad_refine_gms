from __future__ import annotations

import json
from pathlib import Path

import pytest

from cad_rl.data import fingerprint_prepared_dataset, load_prepared_hf_dataset
from cad_rl.data import PreparedDatasetFingerprint, PreparedDatasetManifest
from cad_rl.specs.task import (
    GenerationDefaults,
    ModelSpec,
    PromptRenderSpec,
    TaskSpec,
    default_task_spec,
)


class _DummyProcessor:
    def __init__(self) -> None:
        self.tokenizer = object()

    def apply_chat_template(self, message, tokenize=False, add_generation_prompt=True):
        return json.dumps(
            {
                "message": message,
                "tokenize": tokenize,
                "add_generation_prompt": add_generation_prompt,
            },
            sort_keys=True,
        )


def test_task_spec_round_trip_and_prompt_render():
    spec = default_task_spec()
    payload = spec.to_dict()
    restored = TaskSpec.from_dict(payload)
    processor = _DummyProcessor()
    prompt = restored.render_prompt(processor, image="img")

    assert restored.task_id == spec.task_id
    assert restored.output_var_name == "result"
    assert '"type": "image"' in prompt
    assert restored.default_eval_tiers == ("quick", "standard", "full")


def test_model_spec_round_trip():
    spec = ModelSpec(
        generation_defaults=GenerationDefaults(max_new_tokens=123, temperature=0.5),
        processor_render=PromptRenderSpec(resized_width=32, resized_height=64),
    )
    restored = ModelSpec.from_dict(spec.to_dict())
    assert restored.generation_defaults.max_new_tokens == 123
    assert restored.processor_render.resized_height == 64


def test_prepared_dataset_fingerprint_and_loader(tmp_path: Path):
    datasets = pytest.importorskip("datasets")

    root = tmp_path / "dataset"
    dataset = datasets.Dataset.from_dict({"value": [1, 2, 3]})
    dataset.save_to_disk(str(root))

    fp = fingerprint_prepared_dataset(root)
    loaded = load_prepared_hf_dataset(root)

    assert fp.file_count > 0
    assert fp.byte_size > 0
    assert fp.dataset_path == str(root)
    assert fp.fingerprint
    assert len(loaded) == 3

    restored_fp = PreparedDatasetFingerprint.from_dict(fp.to_dict())
    manifest = PreparedDatasetManifest(
        task_id="task",
        split="train",
        source_root=str(root),
        output_path=str(root),
        row_count=3,
        model_id="model",
        prompt_template_id="template",
        render_profile_id="render",
        fingerprint=fp,
    )
    restored_manifest = PreparedDatasetManifest.from_dict(manifest.to_dict())

    assert restored_fp.dataset_path == str(root)
    assert restored_manifest.fingerprint.fingerprint == fp.fingerprint
