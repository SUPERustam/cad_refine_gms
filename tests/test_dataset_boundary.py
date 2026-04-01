from __future__ import annotations

import builtins
import importlib
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_dataset_root_modules_are_gone() -> None:
    assert not (ROOT / "multiview_dataset.py").exists()
    assert not (ROOT / "helper_visu.py").exists()
    assert (ROOT / "prepare_dataset.py").exists()


def test_flat_legacy_config_profiles_are_gone() -> None:
    legacy = list((ROOT / "configs").glob("*.yaml"))
    assert not legacy, f"legacy flat config profiles remain: {legacy}"


def test_dataset_package_import_smoke() -> None:
    import cad_rl.data.render  # noqa: F401
    import cad_rl.data  # noqa: F401
    import cad_rl.config  # noqa: F401
    import cad_rl.runtime  # noqa: F401
    import prepare_dataset  # noqa: F401

    pytest.importorskip("torch")
    import cad_rl.data.prepare  # noqa: F401
    import cad_rl.data.datasets  # noqa: F401


def test_missing_vis_for_norm_parts_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vis_for_norm_parts":
            raise ImportError("simulated missing vis_for_norm_parts")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("cad_rl.data.render", None)
    mod = importlib.import_module("cad_rl.data.render")

    with pytest.raises(RuntimeError, match="vis_for_norm_parts"):
        mod.Plotter1_1()


def test_rendering_helpers_stay_under_data_boundary() -> None:
    offenders: list[str] = []
    for path in (ROOT / "cad_rl").rglob("*.py"):
        if "/data/" in str(path).replace("\\", "/"):
            continue
        text = path.read_text(encoding="utf-8")
        if "cad_rl.data.render" in text or "vis_for_norm_parts" in text:
            offenders.append(str(path))
    assert not offenders, f"rendering helpers leaked outside cad_rl.data: {offenders}"


def test_evaluation_does_not_execute_cad_code() -> None:
    evaluation_text = (ROOT / "cad_rl" / "evaluation.py").read_text(encoding="utf-8")
    build_meshes_text = (ROOT / "build_meshes.py").read_text(encoding="utf-8")

    assert "execute_code_to_mesh" not in evaluation_text
    assert "cadquery" not in evaluation_text
    assert "build_meshes_from_resolved" in build_meshes_text
