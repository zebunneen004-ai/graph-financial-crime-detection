from __future__ import annotations

import argparse
import gc
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

from src.common import load_config, project_path, record_provenance, save_parquet, setup_logging
from src.modeling_utils import (
    chronological_split,
    feature_sets,
    make_default_model,
    metric_row,
    prediction_frame,
    select_f1_threshold,
    split_summary,
)

MODELS = ["logistic_regression", "random_forest", "xgboost"]
FEATURE_SET_ORDER = ["dataset_only", "graph_only", "combined"]


def load_modeling_frame(config: dict) -> pd.DataFrame:
    processed = project_path(config, "processed_data")
    labeled = pd.read_parquet(processed / "labeled_features.parquet")
    graph = pd.read_parquet(processed / "graph_features.parquet")
    frame = labeled.merge(graph, on=["txId", "time_step"], how="inner", validate="one_to_one")
    if len(frame) != int(config["expected"]["labeled"]):
        raise AssertionError("Modeling merge did not retain every labeled observation.")
    return frame


def create_model_comparison_figure(config: dict, metrics: pd.DataFrame) -> Path:
    test_metrics = metrics.loc[metrics["split"] == "test"].copy()
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(MODELS))
    width = 0.24
    for offset, feature_set in zip((-width, 0, width), FEATURE_SET_ORDER):
        subset = (
            test_metrics.loc[test_metrics["feature_set"] == feature_set]
            .set_index("model_family")
            .reindex(MODELS)
        )
        ax.bar(x + offset, subset["pr_auc"], width, label=feature_set.replace("_", " ").title())
    ax.set_xticks(x, [name.replace("_", " ").title() for name in MODELS])
    ax.set_ylabel("Final-test PR-AUC")
    ax.set_title("Default model ablation: all nine model/feature combinations")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output = project_path(config, "figures_final") / "default_model_ablation.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def run_default_ablation(config: dict) -> None:
    logger = setup_logging(config, "modeling")
    frame = load_modeling_frame(config)
    sets = feature_sets(frame)
    splits = chronological_split(frame, config)

    summary = split_summary(splits)
    split_summary_path = project_path(config, "results_final") / "split_summary.csv"
    summary.to_csv(split_summary_path, index=False)

    manifest_rows = []
    for split_name, split_frame in (
        ("train", splits.train),
        ("calibration", splits.calibration),
        ("operating", splits.operating),
        ("test", splits.test),
    ):
        manifest_rows.append(
            split_frame[["txId", "time_step", "label"]].assign(split=split_name)
        )
    split_manifest = pd.concat(manifest_rows, ignore_index=True)
    split_manifest_path = project_path(config, "results_final") / "split_manifest.parquet"
    save_parquet(split_manifest, split_manifest_path)

    metric_rows: list[dict] = []
    prediction_rows: list[pd.DataFrame] = []
    model_paths: list[Path] = []

    for model_name in MODELS:
        for feature_set_name in FEATURE_SET_ORDER:
            columns = sets[feature_set_name]
            logger.info("Fitting default %s with %s", model_name, feature_set_name)
            model = make_default_model(model_name, config, splits.train["label"])
            model.fit(splits.train[columns], splits.train["label"])

            operating_scores = model.predict_proba(splits.operating[columns])[:, 1]
            threshold, operating_best_f1 = select_f1_threshold(
                splits.operating["label"], operating_scores
            )
            test_scores = model.predict_proba(splits.test[columns])[:, 1]

            for split_name, split_frame, scores in (
                ("operating", splits.operating, operating_scores),
                ("test", splits.test, test_scores),
            ):
                metrics = metric_row(split_frame["label"], scores, threshold)
                metric_rows.append(
                    {
                        "model_family": model_name,
                        "feature_set": feature_set_name,
                        "split": split_name,
                        "operating_selected_threshold": threshold,
                        "operating_best_f1": operating_best_f1,
                        **metrics,
                    }
                )
                prediction_rows.append(
                    prediction_frame(
                        split_frame,
                        scores,
                        threshold,
                        model_family=model_name,
                        feature_set=feature_set_name,
                        split=split_name,
                    )
                )

            model_path = (
                project_path(config, "models_final")
                / f"default_{model_name}_{feature_set_name}.joblib"
            )
            joblib.dump(
                {
                    "model": model,
                    "feature_names": columns,
                    "model_family": model_name,
                    "feature_set": feature_set_name,
                    "threshold": threshold,
                    "training_period": [
                        int(splits.train["time_step"].min()),
                        int(splits.train["time_step"].max()),
                    ],
                },
                model_path,
            )
            model_paths.append(model_path)
            del model, operating_scores, test_scores
            gc.collect()

    metrics_frame = pd.DataFrame(metric_rows)
    predictions_frame = pd.concat(prediction_rows, ignore_index=True)
    metrics_path = project_path(config, "results_final") / "default_model_metrics.csv"
    predictions_path = project_path(config, "results_final") / "default_model_predictions.parquet"
    metrics_frame.to_csv(metrics_path, index=False)
    save_parquet(predictions_frame, predictions_path)
    figure_path = create_model_comparison_figure(config, metrics_frame)

    record_provenance(
        config,
        [split_summary_path, split_manifest_path, metrics_path, predictions_path, figure_path, *model_paths],
        stage="modeling",
        source="src.modeling.run_default_ablation",
        output_type="default model ablation",
        data_split="train 1-30; calibration 31-35 unused here; threshold selection 36-40; final test 41-49",
        prediction_file=str(predictions_path),
        notes=(
            "All nine default comparisons. Logistic regression preprocessing is train-fitted; "
            "tree models are unscaled; thresholds selected on the operating-selection block only."
        ),
    )
    logger.info("Default nine-model ablation complete.")


def run(config_path: str | Path) -> None:
    run_default_ablation(load_config(config_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete default model ablation.")
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().config)
