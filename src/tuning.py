from __future__ import annotations

import argparse
import gc
import json
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
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler

from src.common import (
    get_run_context,
    load_config,
    project_path,
    record_provenance,
    save_parquet,
    setup_logging,
    write_json,
)
from src.modeling import load_modeling_frame
from src.modeling_utils import (
    chronological_split,
    feature_sets,
    make_xgboost,
    metric_row,
    prediction_frame,
    select_f1_threshold,
    temporal_cv_indices,
)

PRIMARY_FEATURE_SETS = ["dataset_only", "combined"]
SENSITIVITY_FEATURE_SET = "combined_with_snapshot_size"


def evaluate_temporal_cv_candidate(
    config: dict,
    train_frame: pd.DataFrame,
    feature_names: list[str],
    feature_set_name: str,
    candidate_name: str,
    params: dict,
    trial: int,
) -> tuple[pd.DataFrame, dict]:
    folds = temporal_cv_indices(train_frame.reset_index(drop=True), config)
    X = train_frame[feature_names].reset_index(drop=True)
    y = train_frame["label"].reset_index(drop=True)
    rows: list[dict] = []
    scores: list[float] = []

    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        model = make_xgboost(params, config, y.iloc[train_indices])
        model.fit(X.iloc[train_indices], y.iloc[train_indices])
        fold_scores = model.predict_proba(X.iloc[validation_indices])[:, 1]
        fold_pr_auc = float(average_precision_score(y.iloc[validation_indices], fold_scores))
        scores.append(fold_pr_auc)
        rows.append(
            {
                "candidate_name": candidate_name,
                "feature_set": feature_set_name,
                "trial": int(trial),
                "fold": int(fold_number),
                "fold_pr_auc": fold_pr_auc,
                "train_min_time": int(train_frame.iloc[train_indices]["time_step"].min()),
                "train_max_time": int(train_frame.iloc[train_indices]["time_step"].max()),
                "validation_min_time": int(train_frame.iloc[validation_indices]["time_step"].min()),
                "validation_max_time": int(train_frame.iloc[validation_indices]["time_step"].max()),
                "parameters": json.dumps(params, sort_keys=True),
            }
        )
        del model, fold_scores
        gc.collect()

    summary = {
        "candidate_name": candidate_name,
        "feature_set": feature_set_name,
        "trial": int(trial),
        "mean_cv_pr_auc": float(np.mean(scores)),
        "std_cv_pr_auc": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "parameters": params,
    }
    return pd.DataFrame(rows), summary


def tune_feature_set(
    config: dict,
    train_frame: pd.DataFrame,
    feature_names: list[str],
    feature_set_name: str,
) -> tuple[dict, pd.DataFrame, dict]:
    logger = setup_logging(config, "tuning")
    tuning_cfg = config["models"]["tuning"]
    candidates = list(
        ParameterSampler(
            tuning_cfg["parameter_space"],
            n_iter=int(tuning_cfg["n_trials"]),
            random_state=int(config["project"]["seed"]),
        )
    )
    fold_tables: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for trial_number, params in enumerate(candidates, start=1):
        table, summary = evaluate_temporal_cv_candidate(
            config,
            train_frame,
            feature_names,
            feature_set_name,
            f"tuned_{feature_set_name}",
            params,
            trial_number,
        )
        fold_tables.append(table)
        summaries.append(summary)
        logger.info(
            "%s tuning trial %d/%d: mean temporal CV PR-AUC %.4f",
            feature_set_name,
            trial_number,
            len(candidates),
            summary["mean_cv_pr_auc"],
        )

    summary_frame = pd.DataFrame(
        [
            {
                **{key: value for key, value in row.items() if key != "parameters"},
                "parameters": json.dumps(row["parameters"], sort_keys=True),
            }
            for row in summaries
        ]
    ).sort_values(
        ["mean_cv_pr_auc", "std_cv_pr_auc", "trial"],
        ascending=[False, True, True],
    )
    best = summary_frame.iloc[0]
    best_params = json.loads(best["parameters"])
    best_summary = {
        "candidate_name": f"tuned_{feature_set_name}",
        "feature_set": feature_set_name,
        "trial": int(best["trial"]),
        "mean_cv_pr_auc": float(best["mean_cv_pr_auc"]),
        "std_cv_pr_auc": float(best["std_cv_pr_auc"]),
        "parameters": best_params,
    }
    return best_params, pd.concat(fold_tables, ignore_index=True), best_summary


def choose_default_or_tuned(
    default_summary: dict,
    tuned_summary: dict,
    minimum_gain: float,
) -> tuple[dict, str]:
    gain = float(tuned_summary["mean_cv_pr_auc"] - default_summary["mean_cv_pr_auc"])
    if gain >= minimum_gain:
        return tuned_summary, (
            f"Tuned candidate selected because temporal-CV PR-AUC improved by {gain:+.4f}, "
            f"meeting the configured {minimum_gain:.4f} minimum tuning gain."
        )
    return default_summary, (
        f"Default candidate retained for parsimony because tuning improved temporal-CV PR-AUC by "
        f"only {gain:+.4f}, below the configured {minimum_gain:.4f} minimum gain."
    )


def paired_bootstrap_delta(
    labels: np.ndarray,
    dataset_scores: np.ndarray,
    combined_scores: np.ndarray,
    repetitions: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    if not (len(labels) == len(dataset_scores) == len(combined_scores)):
        raise ValueError("Paired bootstrap arrays are not aligned.")
    rng = np.random.default_rng(seed)
    point = average_precision_score(labels, combined_scores) - average_precision_score(
        labels, dataset_scores
    )
    deltas: list[float] = []
    for _ in range(repetitions):
        indices = rng.integers(0, len(labels), size=len(labels))
        sampled_labels = labels[indices]
        if np.unique(sampled_labels).size < 2:
            continue
        deltas.append(
            average_precision_score(sampled_labels, combined_scores[indices])
            - average_precision_score(sampled_labels, dataset_scores[indices])
        )
    return float(point), np.asarray(deltas, dtype=float)


def time_block_bootstrap_delta(
    predictions: pd.DataFrame,
    repetitions: int,
    seed: int,
) -> np.ndarray:
    required = {"time_step", "true_label", "dataset_score", "combined_score"}
    if not required.issubset(predictions.columns):
        raise ValueError(f"Time-block bootstrap requires {sorted(required)}")
    rng = np.random.default_rng(seed)
    time_steps = np.sort(predictions["time_step"].unique())
    blocks = {time: predictions.loc[predictions["time_step"] == time] for time in time_steps}
    deltas: list[float] = []
    for _ in range(repetitions):
        sampled_steps = rng.choice(time_steps, size=len(time_steps), replace=True)
        sampled = pd.concat([blocks[int(time)] for time in sampled_steps], ignore_index=True)
        if sampled["true_label"].nunique() < 2:
            continue
        deltas.append(
            average_precision_score(sampled["true_label"], sampled["combined_score"])
            - average_precision_score(sampled["true_label"], sampled["dataset_score"])
        )
    return np.asarray(deltas, dtype=float)


def confidence_interval(values: np.ndarray, level: float) -> tuple[float, float]:
    if len(values) == 0:
        raise ValueError("No valid bootstrap repetitions were retained.")
    alpha = 1.0 - level
    return (
        float(np.quantile(values, alpha / 2)),
        float(np.quantile(values, 1 - alpha / 2)),
    )


def create_candidate_figures(
    config: dict,
    candidate_summary: pd.DataFrame,
    candidate_metrics: pd.DataFrame,
    bootstrap_values: np.ndarray,
    point_delta: float,
) -> list[Path]:
    figures = project_path(config, "figures_final")
    paths: list[Path] = []

    primary_candidates = [
        "default_dataset_only",
        "tuned_dataset_only",
        "default_combined",
        "tuned_combined",
    ]
    cv = candidate_summary.set_index("candidate_name").loc[primary_candidates]
    test = (
        candidate_metrics.loc[candidate_metrics["split"] == "test"]
        .set_index("candidate_name")
        .loc[primary_candidates]
    )
    labels = [name.replace("_", " ").title() for name in primary_candidates]
    x = np.arange(len(primary_candidates))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, cv["mean_cv_pr_auc"], width, label="Mean temporal CV")
    ax.bar(x + width / 2, test["pr_auc"], width, label="Final test (reported after selection)")
    ax.errorbar(
        x - width / 2,
        cv["mean_cv_pr_auc"],
        yerr=cv["std_cv_pr_auc"].fillna(0),
        fmt="none",
        capsize=3,
    )
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("PR-AUC")
    ax.set_title("Default and tuned XGBoost candidates")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    comparison_path = figures / "xgboost_candidate_comparison.png"
    fig.savefig(comparison_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(comparison_path)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(bootstrap_values, bins=45)
    ax.axvline(0, linestyle="--", linewidth=1.5, label="No difference")
    ax.axvline(point_delta, linewidth=1.5, label=f"Observed Δ={point_delta:.4f}")
    ax.set_xlabel("Paired bootstrap ΔPR-AUC (best combined − best dataset only)")
    ax.set_ylabel("Bootstrap samples")
    ax.set_title("Incremental value of added topology features")
    ax.legend()
    fig.tight_layout()
    bootstrap_path = figures / "paired_delta_bootstrap.png"
    fig.savefig(bootstrap_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    paths.append(bootstrap_path)
    return paths


def run_tuning(config: dict) -> None:
    logger = setup_logging(config, "tuning")
    frame = load_modeling_frame(config)
    sets = feature_sets(frame)
    sets[SENSITIVITY_FEATURE_SET] = [*sets["combined"], "weak_component_size"]
    splits = chronological_split(frame, config)
    results_dir = project_path(config, "results_final")
    models_dir = project_path(config, "models_final")

    default_params = dict(config["models"]["default_xgboost"])
    all_fold_results: list[pd.DataFrame] = []
    candidate_summaries: dict[str, dict] = {}
    best_parameters: dict[str, dict] = {}

    # Evaluate fixed default candidates on exactly the same temporal folds used for tuning.
    for feature_set_name in PRIMARY_FEATURE_SETS:
        candidate_name = f"default_{feature_set_name}"
        fold_table, summary = evaluate_temporal_cv_candidate(
            config,
            splits.train,
            sets[feature_set_name],
            feature_set_name,
            candidate_name,
            default_params,
            trial=0,
        )
        all_fold_results.append(fold_table)
        candidate_summaries[candidate_name] = summary

    # Tune both primary feature sets and the snapshot-size sensitivity model identically.
    for feature_set_name in [*PRIMARY_FEATURE_SETS, SENSITIVITY_FEATURE_SET]:
        best_params, fold_table, summary = tune_feature_set(
            config, splits.train, sets[feature_set_name], feature_set_name
        )
        all_fold_results.append(fold_table)
        candidate_summaries[f"tuned_{feature_set_name}"] = summary
        best_parameters[feature_set_name] = best_params

    minimum_tuning_gain = float(config["models"]["tuning"]["minimum_tuning_gain_pr_auc"])
    best_dataset, dataset_variant_reason = choose_default_or_tuned(
        candidate_summaries["default_dataset_only"],
        candidate_summaries["tuned_dataset_only"],
        minimum_tuning_gain,
    )
    best_combined, combined_variant_reason = choose_default_or_tuned(
        candidate_summaries["default_combined"],
        candidate_summaries["tuned_combined"],
        minimum_tuning_gain,
    )

    practical_margin = float(config["models"]["tuning"]["practical_margin_pr_auc"])
    feature_gain = float(best_combined["mean_cv_pr_auc"] - best_dataset["mean_cv_pr_auc"])
    if feature_gain >= practical_margin:
        selected_summary = best_combined
        feature_selection_reason = (
            f"Combined selected because its best candidate exceeded the best dataset-only candidate by "
            f"{feature_gain:+.4f} mean temporal-CV PR-AUC, meeting the {practical_margin:.4f} margin."
        )
    else:
        selected_summary = best_dataset
        feature_selection_reason = (
            f"Dataset-only selected for parsimony because the best combined candidate improved mean "
            f"temporal-CV PR-AUC by only {feature_gain:+.4f}, below the {practical_margin:.4f} margin."
        )

    candidate_definitions = {
        "default_dataset_only": ("dataset_only", default_params),
        "tuned_dataset_only": ("dataset_only", best_parameters["dataset_only"]),
        "default_combined": ("combined", default_params),
        "tuned_combined": ("combined", best_parameters["combined"]),
        "tuned_combined_with_snapshot_size": (
            SENSITIVITY_FEATURE_SET,
            best_parameters[SENSITIVITY_FEATURE_SET],
        ),
    }

    metric_rows: list[dict] = []
    prediction_tables: dict[str, dict[str, pd.DataFrame]] = {}
    model_paths: dict[str, Path] = {}

    for candidate_name, (feature_set_name, params) in candidate_definitions.items():
        columns = sets[feature_set_name]
        model = make_xgboost(params, config, splits.train["label"])
        model.fit(splits.train[columns], splits.train["label"])

        scores_by_split = {
            "calibration": model.predict_proba(splits.calibration[columns])[:, 1],
            "operating": model.predict_proba(splits.operating[columns])[:, 1],
            "test": model.predict_proba(splits.test[columns])[:, 1],
        }
        threshold, operating_best_f1 = select_f1_threshold(
            splits.operating["label"], scores_by_split["operating"]
        )
        prediction_tables[candidate_name] = {}
        for split_name, split_frame in (
            ("calibration", splits.calibration),
            ("operating", splits.operating),
            ("test", splits.test),
        ):
            scores = scores_by_split[split_name]
            prediction_tables[candidate_name][split_name] = prediction_frame(
                split_frame,
                scores,
                threshold,
                model_family=candidate_name,
                feature_set=feature_set_name,
                split=split_name,
            )
            metric_rows.append(
                {
                    "candidate_name": candidate_name,
                    "model_variant": "tuned" if candidate_name.startswith("tuned_") else "default",
                    "feature_set": feature_set_name,
                    "split": split_name,
                    "operating_best_f1": operating_best_f1,
                    **metric_row(split_frame["label"], scores, threshold),
                }
            )

        model_path = models_dir / f"xgboost_{candidate_name}.joblib"
        joblib.dump(
            {
                "model": model,
                "feature_names": columns,
                "candidate_name": candidate_name,
                "feature_set": feature_set_name,
                "parameters": params,
                "threshold": threshold,
                "training_period": [1, 30],
                "model_selection_basis": "expanding-window temporal CV within time steps 1-30",
                "calibration_period": [31, 35],
                "operating_selection_period": [36, 40],
                "final_test_period": [41, 49],
            },
            model_path,
        )
        model_paths[candidate_name] = model_path
        del model, scores_by_split
        gc.collect()

    metrics = pd.DataFrame(metric_rows)
    summary_frame = pd.DataFrame(
        [
            {
                "candidate_name": name,
                "feature_set": summary["feature_set"],
                "mean_cv_pr_auc": summary["mean_cv_pr_auc"],
                "std_cv_pr_auc": summary["std_cv_pr_auc"],
                "trial": summary["trial"],
                "parameters": json.dumps(summary["parameters"], sort_keys=True),
            }
            for name, summary in candidate_summaries.items()
        ]
    )

    best_dataset_name = str(best_dataset["candidate_name"])
    best_combined_name = str(best_combined["candidate_name"])
    selected_candidate_name = str(selected_summary["candidate_name"])
    selected_feature_set = str(selected_summary["feature_set"])

    aligned_test = prediction_tables[best_dataset_name]["test"][[
        "txId", "time_step", "true_label", "raw_score"
    ]].rename(columns={"raw_score": "dataset_score"})
    aligned_test = aligned_test.merge(
        prediction_tables[best_combined_name]["test"][["txId", "raw_score"]].rename(
            columns={"raw_score": "combined_score"}
        ),
        on="txId",
        how="inner",
        validate="one_to_one",
    )

    repetitions = int(config["bootstrap"]["repetitions"])
    level = float(config["bootstrap"]["confidence_level"])
    seed = int(config["project"]["seed"])
    point_delta, paired_values = paired_bootstrap_delta(
        aligned_test["true_label"].to_numpy(),
        aligned_test["dataset_score"].to_numpy(),
        aligned_test["combined_score"].to_numpy(),
        repetitions,
        seed,
    )
    paired_ci = confidence_interval(paired_values, level)
    block_values = time_block_bootstrap_delta(aligned_test, repetitions, seed + 1)
    block_ci = confidence_interval(block_values, level)

    inference = {
        "comparison": f"{best_combined_name} minus {best_dataset_name}",
        "dataset_candidate": best_dataset_name,
        "combined_candidate": best_combined_name,
        "point_delta_pr_auc": point_delta,
        "paired_bootstrap_repetitions_retained": int(len(paired_values)),
        "paired_bootstrap_confidence_interval": list(paired_ci),
        "time_step_block_bootstrap_repetitions_retained": int(len(block_values)),
        "time_step_block_bootstrap_confidence_interval": list(block_ci),
        "confidence_level": level,
        "practical_margin_pr_auc": practical_margin,
        "interpretation": (
            "A confidence interval containing zero means the paired analysis did not detect a statistically "
            "reliable improvement. It does not prove the models are identical or that graph features can never help."
        ),
    }

    sensitivity_validation = metrics.loc[
        (metrics["split"] == "operating")
        & (metrics["candidate_name"] == "tuned_combined_with_snapshot_size")
    ].iloc[0]
    tuned_combined_operating = metrics.loc[
        (metrics["split"] == "operating")
        & (metrics["candidate_name"] == "tuned_combined")
    ].iloc[0]
    sensitivity_test = metrics.loc[
        (metrics["split"] == "test")
        & (metrics["candidate_name"] == "tuned_combined_with_snapshot_size")
    ].iloc[0]
    tuned_combined_test = metrics.loc[
        (metrics["split"] == "test")
        & (metrics["candidate_name"] == "tuned_combined")
    ].iloc[0]
    snapshot_sensitivity = {
        "feature_set": "dataset predictors + five primary topology variables + weak_component_size",
        "role": "sensitivity analysis only; not eligible for primary operational selection",
        "operating_pr_auc": float(sensitivity_validation["pr_auc"]),
        "final_test_pr_auc": float(sensitivity_test["pr_auc"]),
        "operating_delta_vs_tuned_primary_combined": float(
            sensitivity_validation["pr_auc"] - tuned_combined_operating["pr_auc"]
        ),
        "final_test_delta_vs_tuned_primary_combined": float(
            sensitivity_test["pr_auc"] - tuned_combined_test["pr_auc"]
        ),
        "interpretation": (
            "weak_component_size is treated as a snapshot-level time-step proxy, not transaction-specific isolation."
        ),
    }

    fold_path = results_dir / "temporal_tuning_fold_results.csv"
    summary_path = results_dir / "xgboost_candidate_cv_summary.csv"
    metrics_path = results_dir / "xgboost_candidate_metrics.csv"
    # Compatibility alias retained for downstream users of the first corrected run.
    tuned_metrics_path = results_dir / "tuned_xgboost_metrics.csv"
    aligned_path = results_dir / "aligned_incremental_test_predictions.parquet"
    inference_path = results_dir / "incremental_value_inference.json"
    sensitivity_path = results_dir / "snapshot_size_sensitivity.json"
    selection_path = results_dir / "operational_model_selection.json"
    raw_predictions_path = results_dir / "operational_raw_predictions.parquet"

    pd.concat(all_fold_results, ignore_index=True).to_csv(fold_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    metrics.to_csv(tuned_metrics_path, index=False)
    save_parquet(aligned_test, aligned_path)
    write_json(inference, inference_path)
    write_json(snapshot_sensitivity, sensitivity_path)

    selection_payload = {
        "selected_candidate_name": selected_candidate_name,
        "selected_model_family": "xgboost",
        "selected_model_variant": "tuned" if selected_candidate_name.startswith("tuned_") else "default",
        "selected_feature_set": selected_feature_set,
        "selected_model_artifact": str(model_paths[selected_candidate_name].relative_to(Path(__file__).resolve().parents[1])),
        "best_dataset_candidate": best_dataset_name,
        "best_dataset_model_artifact": str(model_paths[best_dataset_name].relative_to(Path(__file__).resolve().parents[1])),
        "best_combined_candidate": best_combined_name,
        "best_combined_model_artifact": str(model_paths[best_combined_name].relative_to(Path(__file__).resolve().parents[1])),
        "model_selection_population": "expanding-window temporal cross-validation within labeled time steps 1-30",
        "calibration_population": "labeled time steps 31-35",
        "operating_selection_population": "labeled time steps 36-40",
        "final_test_population": "labeled time steps 41-49",
        "default_vs_tuned_rule": f"tuning must improve mean temporal-CV PR-AUC by at least {minimum_tuning_gain}",
        "feature_set_rule": f"combined must exceed dataset-only mean temporal-CV PR-AUC by at least {practical_margin}",
        "dataset_variant_reason": dataset_variant_reason,
        "combined_variant_reason": combined_variant_reason,
        "feature_selection_reason": feature_selection_reason,
        "final_test_not_used_for_selection": True,
        "candidate_cv_results": {
            name: {
                "mean_cv_pr_auc": float(summary["mean_cv_pr_auc"]),
                "std_cv_pr_auc": float(summary["std_cv_pr_auc"]),
            }
            for name, summary in candidate_summaries.items()
        },
        "best_parameters": best_parameters,
    }
    write_json(selection_payload, selection_path)

    operational_predictions = pd.concat(
        [
            prediction_tables[selected_candidate_name]["calibration"],
            prediction_tables[selected_candidate_name]["operating"],
            prediction_tables[selected_candidate_name]["test"],
        ],
        ignore_index=True,
    )
    operational_predictions["model_version"] = get_run_context().run_id
    operational_predictions["seed"] = seed
    operational_predictions["candidate_name"] = selected_candidate_name
    save_parquet(operational_predictions, raw_predictions_path)

    figure_paths = create_candidate_figures(
        config, summary_frame, metrics, paired_values, point_delta
    )
    record_provenance(
        config,
        [
            fold_path,
            summary_path,
            metrics_path,
            tuned_metrics_path,
            aligned_path,
            inference_path,
            sensitivity_path,
            selection_path,
            raw_predictions_path,
            *model_paths.values(),
            *figure_paths,
        ],
        stage="tuning",
        source="src.tuning.run_tuning",
        output_type="temporal candidate selection and incremental-value inference",
        model_family="XGBoost",
        data_split=(
            "temporal CV/model selection within 1-30; calibration 31-35; "
            "operating selection 36-40; final test 41-49"
        ),
        prediction_file=str(raw_predictions_path),
        notes=(
            "Default and tuned dataset-only/combined candidates were evaluated on identical temporal folds; "
            "the final test was not used for selection."
        ),
    )
    logger.info("Candidate selection complete. Operational candidate: %s", selected_candidate_name)


def run(config_path: str | Path) -> None:
    run_tuning(load_config(config_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run temporal XGBoost tuning, candidate selection, and incremental inference."
    )
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().config)
