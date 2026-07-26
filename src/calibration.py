from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.common import (
    load_config,
    project_path,
    read_json,
    record_provenance,
    save_parquet,
    setup_logging,
    write_json,
)


def logit_transform(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_sigmoid_calibrator(validation_scores: np.ndarray, validation_labels: np.ndarray, seed: int):
    model = LogisticRegression(random_state=seed, max_iter=2000)
    model.fit(logit_transform(validation_scores), validation_labels)
    return model


def apply_calibrator(model, scores: np.ndarray) -> np.ndarray:
    return model.predict_proba(logit_transform(scores))[:, 1]


def run_calibration(config: dict) -> None:
    logger = setup_logging(config, "calibration")
    results = project_path(config, "results_final")
    models = project_path(config, "models_final")
    figures = project_path(config, "figures_final")

    predictions = pd.read_parquet(results / "operational_raw_predictions.parquet")
    calibration = predictions.loc[predictions["split"] == "calibration"].copy()
    operating = predictions.loc[predictions["split"] == "operating"].copy()
    test = predictions.loc[predictions["split"] == "test"].copy()
    if calibration.empty or operating.empty or test.empty:
        raise AssertionError("Operational predictions must include calibration, operating, and final-test rows.")

    method = str(config["calibration"]["method"]).lower()
    if method != "sigmoid":
        raise ValueError("This implementation currently supports dedicated calibration-block sigmoid calibration only.")

    seed = int(config["project"]["seed"])
    calibrator = fit_sigmoid_calibrator(
        calibration["raw_score"].to_numpy(), calibration["true_label"].to_numpy(), seed
    )
    calibration["calibrated_probability"] = apply_calibrator(
        calibrator, calibration["raw_score"].to_numpy()
    )
    operating["calibrated_probability"] = apply_calibrator(
        calibrator, operating["raw_score"].to_numpy()
    )
    test["calibrated_probability"] = apply_calibrator(calibrator, test["raw_score"].to_numpy())
    calibrated = pd.concat([calibration, operating, test], ignore_index=True)

    metrics = {
        "method": "sigmoid / Platt-style logistic calibration",
        "fit_population": "labeled calibration time steps 31-35 only",
        "evaluation_population": "labeled final-test time steps 41-49",
        "raw_test_brier_score": float(brier_score_loss(test["true_label"], test["raw_score"])),
        "calibrated_test_brier_score": float(
            brier_score_loss(test["true_label"], test["calibrated_probability"])
        ),
        "test_prevalence": float(test["true_label"].mean()),
        "discrimination_note": (
            "Calibration evaluates probability agreement and is separate from ranking discrimination. "
            "It is not expected to improve PR-AUC."
        ),
    }

    output_path = results / "calibrated_operational_predictions.parquet"
    metrics_path = results / "calibration_metrics.json"
    model_path = models / "calibration_block_sigmoid_calibrator.joblib"
    save_parquet(calibrated, output_path)
    write_json(metrics, metrics_path)
    joblib.dump(
        {
            "calibrator": calibrator,
            "input_transform": "logit(clipped raw score)",
            "fit_split": "calibration 31-35",
            "seed": seed,
        },
        model_path,
    )

    bins = int(config["calibration"]["bins"])
    raw_fraction, raw_mean = calibration_curve(
        test["true_label"], test["raw_score"], n_bins=bins, strategy="quantile"
    )
    calibrated_fraction, calibrated_mean = calibration_curve(
        test["true_label"], test["calibrated_probability"], n_bins=bins, strategy="quantile"
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.plot(raw_mean, raw_fraction, marker="o", label="Raw XGBoost score")
    ax.plot(calibrated_mean, calibrated_fraction, marker="o", label="Calibration-block-fitted sigmoid")
    ax.set_xlabel("Mean predicted value")
    ax.set_ylabel("Observed illicit-labeled fraction")
    ax.set_title("Calibration on the final labeled holdout")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    figure_path = figures / "calibration_curve.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    record_provenance(
        config,
        [output_path, metrics_path, model_path, figure_path],
        stage="calibration",
        source="src.calibration.run_calibration",
        output_type="calibration model, predictions, metrics, and figure",
        model_family="XGBoost + sigmoid calibrator",
        calibration_status="calibrator fitted on dedicated calibration block only",
        data_split="fit 31-35; operating selection 36-40; evaluate 41-49",
        prediction_file=str(output_path),
        notes="Raw outputs remain model scores; calibrated values are reported as probabilities.",
    )
    logger.info("Calibration complete. Raw Brier %.5f; calibrated Brier %.5f", metrics["raw_test_brier_score"], metrics["calibrated_test_brier_score"])


def run(config_path: str | Path) -> None:
    run_calibration(load_config(config_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit probability calibration on the dedicated calibration block.")
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().config)
