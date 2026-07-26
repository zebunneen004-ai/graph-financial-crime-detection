from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunContext:
    run_id: str
    timestamp_utc: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")


def get_run_context() -> RunContext:
    run_id = os.environ.get("PROJECT_RUN_ID", default_run_id())
    timestamp = os.environ.get("PROJECT_RUN_TIMESTAMP", utc_now())
    return RunContext(run_id=run_id, timestamp_utc=timestamp)


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    return config


def project_path(config: dict[str, Any], key: str) -> Path:
    value = config["paths"][key]
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def ensure_directories(config: dict[str, Any]) -> None:
    for key in (
        "processed_data",
        "results_final",
        "results_archive",
        "figures_final",
        "figures_archive",
        "models_final",
        "models_archive",
        "logs",
        "report",
        "employer_brief",
    ):
        project_path(config, key).mkdir(parents=True, exist_ok=True)


def resolve_raw_file(config: dict[str, Any], filename: str) -> Path:
    candidates = [
        project_path(config, "raw_data") / filename,
        project_path(config, "legacy_raw_data") / filename,
        ROOT / "data" / "raw" / filename,
        ROOT / "data" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    searched = "\n".join(f"  - {p}" for p in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Searched:\n{searched}")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def setup_logging(config: dict[str, Any], stage: str) -> logging.Logger:
    ensure_directories(config)
    context = get_run_context()
    logger = logging.getLogger(f"{context.run_id}.{stage}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    log_path = project_path(config, "logs") / f"{context.run_id}_{stage}.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", delete=False, dir=path.parent, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=str)
        temp_name = handle.name
    Path(temp_name).replace(path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        frame.to_parquet(path, index=False, engine="pyarrow", compression="zstd")
    except ImportError as exc:
        raise RuntimeError(
            "Parquet output requires pyarrow. Install dependencies with "
            "`python -m pip install -r requirements.txt`."
        ) from exc


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", delete=False, dir=path.parent, suffix=".tmp"
    ) as handle:
        frame.to_csv(handle, index=False)
        temp_name = handle.name
    Path(temp_name).replace(path)


def relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def record_provenance(
    config: dict[str, Any],
    outputs: Iterable[Path],
    *,
    stage: str,
    source: str,
    output_type: str,
    model_family: str = "",
    feature_set: str = "",
    hyperparameters: dict[str, Any] | None = None,
    calibration_status: str = "",
    data_split: str = "",
    prediction_file: str = "",
    notes: str = "",
) -> None:
    context = get_run_context()
    path = project_path(config, "results_final") / "provenance_manifest.csv"
    rows: list[dict[str, Any]] = []
    for output in outputs:
        rows.append(
            {
                "output_path": relative_to_root(output),
                "output_type": output_type,
                "run_id": context.run_id,
                "timestamp_utc": utc_now(),
                "source_script_function": source,
                "stage": stage,
                "model_family": model_family,
                "feature_set": feature_set,
                "hyperparameters": json.dumps(hyperparameters or {}, sort_keys=True),
                "calibration_status": calibration_status,
                "data_split": data_split,
                "prediction_file": relative_to_root(Path(prediction_file)) if prediction_file else "",
                "seed": config["project"]["seed"],
                "git_commit": git_commit(),
                "notes": notes,
            }
        )

    existing = pd.read_csv(path) if path.exists() else pd.DataFrame()
    combined = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    atomic_csv(combined, path)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unavailable"


def package_versions(packages: Iterable[str]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def runtime_metadata(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": get_run_context().__dict__,
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": git_commit(),
        "config_path": relative_to_root(Path(config["_config_path"])) if config.get("_config_path") else "",
        "packages": package_versions(
            [
                "pandas",
                "numpy",
                "networkx",
                "scipy",
                "scikit-learn",
                "xgboost",
                "shap",
                "statsmodels",
                "matplotlib",
                "joblib",
                "PyYAML",
                "pyarrow",
                "tabulate",
            ]
        ),
    }


def validate_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
