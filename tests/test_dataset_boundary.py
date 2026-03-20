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


def test_dataset_package_import_smoke() -> None:
    import cad_rl.data.helper_visu  # noqa: F401
    import cad_rl.data.multiview_dataset  # noqa: F401
    import cad_rl.data.prepare_dataset  # noqa: F401
    import cad_rl.data  # noqa: F401
    import cad_rl.algorithms  # noqa: F401
    import cad_rl.pipelines.comparison  # noqa: F401
    import cad_rl.pipelines.inference  # noqa: F401
    import cad_rl.models  # noqa: F401
    import cad_rl.config  # noqa: F401
    import cad_rl.runtime  # noqa: F401
    import cad_rl.specs  # noqa: F401
    import prepare_dataset  # noqa: F401


def test_missing_vis_for_norm_parts_raises_clear_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "vis_for_norm_parts":
            raise ImportError("simulated missing vis_for_norm_parts")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    sys.modules.pop("cad_rl.data.helper_visu", None)
    mod = importlib.import_module("cad_rl.data.helper_visu")

    with pytest.raises(RuntimeError, match="vis_for_norm_parts"):
        mod.Plotter1_1()


def test_no_former_root_module_imports_under_cad_rl() -> None:
    forbidden = (
        "from multiview_dataset",
        "import multiview_dataset",
        "from helper_visu",
        "import helper_visu",
        "from utils",
        "import utils",
    )
    scan_roots = [
        ROOT / "cad_rl" / "data",
        ROOT / "cad_rl" / "utils.py",
        ROOT / "prepare_dataset.py",
    ]
    offenders: list[str] = []
    for scan_root in scan_roots:
        if scan_root.is_file():
            paths = [scan_root]
        else:
            paths = list(scan_root.rglob("*.py"))
        for path in paths:
            text = path.read_text(encoding="utf-8")
            if any(token in text for token in forbidden):
                offenders.append(str(path))
    assert not offenders, f"root-module imports leaked into: {offenders}"
