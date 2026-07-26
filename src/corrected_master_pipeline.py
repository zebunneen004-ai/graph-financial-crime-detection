from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import (
    ROOT,
    ensure_directories,
    load_config,
    project_path,
    runtime_metadata,
    setup_logging,
    utc_now,
    write_json,
)

STAGES = [
    "data_loading",
    "graph_construction",
    "graph_features",
    "statistics",
    "modeling",
    "tuning",
    "calibration",
    "alert_budget",
    "explainability",
    "reporting",
]


def archive_existing_final_outputs(config: dict, run_id: str) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    mappings = [
        ("results_final", "results_archive"),
        ("figures_final", "figures_archive"),
        ("models_final", "models_archive"),
    ]
    for final_key, archive_key in mappings:
        source = project_path(config, final_key)
        if not source.exists() or not any(source.iterdir()):
            source.mkdir(parents=True, exist_ok=True)
            continue
        destination = project_path(config, archive_key) / f"{timestamp}_before_{run_id}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        for child in source.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()


def run_stage(stage: str, config_path: Path, environment: dict[str, str], logger) -> None:
    command = [sys.executable, "-m", f"src.{stage}", "--config", str(config_path)]
    logger.info("Starting stage: %s", stage)
    started = time.perf_counter()
    result = subprocess.run(command, cwd=ROOT, env=environment)
    elapsed = time.perf_counter() - started
    if result.returncode != 0:
        raise RuntimeError(f"Stage {stage} failed with exit code {result.returncode}.")
    logger.info("Completed stage %s in %.2f seconds", stage, elapsed)


def run_pipeline(config_path: str | Path, overwrite: bool, selected_stages: list[str] | None) -> None:
    path = Path(config_path)
    if not path.is_absolute():
        path = ROOT / path
    config = load_config(path)
    ensure_directories(config)

    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    timestamp = utc_now()
    environment = os.environ.copy()
    environment["PROJECT_RUN_ID"] = run_id
    environment["PROJECT_RUN_TIMESTAMP"] = timestamp
    environment.setdefault("PYTHONHASHSEED", str(config["project"]["seed"]))
    environment.setdefault("OMP_NUM_THREADS", str(config["models"]["threads"]))
    environment.setdefault("MKL_NUM_THREADS", str(config["models"]["threads"]))

    os.environ.update(
        {
            "PROJECT_RUN_ID": run_id,
            "PROJECT_RUN_TIMESTAMP": timestamp,
        }
    )
    logger = setup_logging(config, "master_pipeline")

    final_directories = [
        project_path(config, "results_final"),
        project_path(config, "figures_final"),
        project_path(config, "models_final"),
    ]
    outputs_exist = any(directory.exists() and any(directory.iterdir()) for directory in final_directories)
    if outputs_exist and not overwrite:
        raise FileExistsError(
            "Final output directories are not empty. Re-run with --overwrite to archive them before a clean run."
        )
    if overwrite:
        archive_existing_final_outputs(config, run_id)

    metadata_path = project_path(config, "results_final") / "run_metadata.json"
    write_json(runtime_metadata(config), metadata_path)

    stages = selected_stages or STAGES
    invalid = sorted(set(stages) - set(STAGES))
    if invalid:
        raise ValueError(f"Unknown stages: {invalid}. Valid stages: {STAGES}")

    started = time.perf_counter()
    logger.info("Run ID: %s", run_id)
    logger.info("Stages: %s", ", ".join(stages))
    for stage in stages:
        run_stage(stage, path, environment, logger)
    elapsed = time.perf_counter() - started

    completion = {
        "run_id": run_id,
        "started_utc": timestamp,
        "completed_utc": utc_now(),
        "runtime_seconds": elapsed,
        "stages": stages,
        "status": "success",
    }
    write_json(completion, project_path(config, "results_final") / "run_completion.json")
    logger.info("Corrected pipeline completed successfully in %.2f seconds", elapsed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the corrected memory-isolated project pipeline.")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Archive existing final outputs and run from clean final directories.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=STAGES,
        help="Run only selected stages. Dependencies must already exist.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run_pipeline(arguments.config, arguments.overwrite, arguments.stages)
