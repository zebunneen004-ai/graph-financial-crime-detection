from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive" / "original_v1"
SENTINEL = ARCHIVE / "ARCHIVED.txt"

LEGACY_FILES = [
    "05_modeling_enhancement.py",
    "05_modeling_enhancement_FIXED.py",
    "07_aml_alert_quality.py",
    "README_backup.md",
    "run.bat.legacy",
]
LEGACY_DIRECTORIES = ["notebooks", "figures", "results", "models"]


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> None:
    if SENTINEL.exists():
        print(f"Legacy archive already prepared: {ARCHIVE}")
        return
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    # Copy only known legacy artifacts. Raw data, .git, .venv/venv and corrected source are excluded.
    for name in (
        "05_modeling_enhancement.py",
        "05_modeling_enhancement_FIXED.py",
        "07_aml_alert_quality.py",
        "README_backup.md",
        "report/paper_draft.md",
        "employer_brief/employer_brief.md",
    ):
        copy_if_exists(ROOT / name, ARCHIVE / name)

    for directory in ("notebooks",):
        copy_if_exists(ROOT / directory, ARCHIVE / directory)

    for directory in ("figures", "results", "models"):
        source = ROOT / directory
        if source.exists():
            destination = ARCHIVE / directory
            destination.mkdir(parents=True, exist_ok=True)
            for child in source.iterdir():
                if child.name in {"final", "archive"}:
                    continue
                copy_if_exists(child, destination / child.name)

    SENTINEL.write_text(
        "Legacy source, notebooks and historical outputs copied before corrected execution.\n"
        "Raw data, virtual environments and Git internals were intentionally excluded.\n",
        encoding="utf-8",
    )
    print(f"Prepared legacy archive: {ARCHIVE}")


if __name__ == "__main__":
    main()
