#!/usr/bin/env python3
"""Move preds/ and preds/renders/ into predictions/ by filename stem."""

from pathlib import Path
import shutil

# --- config ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PREDS = PROJECT_ROOT / "preds"
RENDERS = PREDS / "renders"
DEST = PROJECT_ROOT / "predictions"

MCB = DEST / "mcb_a_batch_groundtruth_1000"
F360 = DEST / "fusion360_test_mesh_1000"
MCB_RENDERS = MCB / "renders"
F360_RENDERS = F360 / "renders"


def kind(stem: str) -> str | None:
    if stem.isdigit():
        return "mcb"
    parts = stem.split("_")
    if len(parts) >= 3 and all(parts):
        return "f360"
    return None


def relocate(
    src_dir: Path,
    mcb_out: Path,
    f360_out: Path,
    *,
    txt_only: bool,
) -> None:
    if not src_dir.is_dir():
        return
    for path in sorted(src_dir.iterdir()):
        if not path.is_file():
            continue
        if txt_only and path.suffix.lower() != ".txt":
            continue
        k = kind(path.stem)
        if k is None:
            continue
        out = mcb_out if k == "mcb" else f360_out
        out.mkdir(parents=True, exist_ok=True)
        dest = out / path.name
        if dest.exists():
            raise FileExistsError(dest)
        shutil.copy(str(path), str(dest))


def main() -> None:
    relocate(PREDS, MCB, F360, txt_only=True)
    relocate(RENDERS, MCB_RENDERS, F360_RENDERS, txt_only=False)


if __name__ == "__main__":
    main()
