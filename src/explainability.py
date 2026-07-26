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
from sklearn.inspection import permutation_importance
import shap

from src.common import (
    load_config,
    project_path,
    read_json,
    record_provenance,
    setup_logging,
    write_json,
)
from src.graph_features import PRIMARY_GRAPH_FEATURES
from src.modeling import load_modeling_frame
from src.modeling_utils import chronological_split, feature_sets


def stratified_sample(frame: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if len(frame) <= sample_size:
        return frame.copy()
    prevalence = frame["label"].value_counts(normalize=True)
    pieces = []
    remaining = sample_size
    classes = sorted(frame["label"].unique())
    for index, label in enumerate(classes):
        group = frame.loc[frame["label"] == label]
        if index == len(classes) - 1:
            count = min(len(group), remaining)
        else:
            count = min(len(group), max(1, int(round(sample_size * prevalence[label]))))
        pieces.append(group.sample(n=count, random_state=seed + int(label)))
        remaining -= count
    sample = pd.concat(pieces, ignore_index=True)
    if len(sample) < sample_size:
        remaining_pool = frame.loc[~frame["txId"].isin(sample["txId"])]
        sample = pd.concat(
            [sample, remaining_pool.sample(n=sample_size - len(sample), random_state=seed + 99)],
            ignore_index=True,
        )
    return sample.sample(frac=1, random_state=seed).reset_index(drop=True)


def save_shap_plots(
    config: dict,
    model,
    sample: pd.DataFrame,
    feature_names: list[str],
    cases: pd.DataFrame,
) -> tuple[list[Path], pd.DataFrame, dict]:
    logger = setup_logging(config, "explainability")
    figures = project_path(config, "figures_final")
    X_sample = sample[feature_names]

    logger.info("Computing TreeSHAP values on %d final-test observations.", len(sample))
    explainer = shap.TreeExplainer(model, model_output="raw")
    explanation = explainer(X_sample)
    mean_abs = np.abs(explanation.values).mean(axis=0)
    importance = pd.DataFrame(
        {"feature": feature_names, "mean_absolute_shap": mean_abs}
    ).sort_values("mean_absolute_shap", ascending=False, ignore_index=True)
    top_feature = str(importance.iloc[0]["feature"])

    paths: list[Path] = []
    plt.figure()
    shap.plots.beeswarm(explanation, max_display=int(config["explainability"]["max_display"]), show=False)
    plt.title("SHAP summary for the authoritative operational model")
    summary_path = figures / "shap_summary.png"
    plt.gcf().savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close(plt.gcf())
    paths.append(summary_path)

    top_index = feature_names.index(top_feature)
    plt.figure()
    shap.plots.scatter(explanation[:, top_index], show=False)
    plt.title(f"SHAP dependence: {top_feature}")
    dependence_path = figures / "shap_dependence_top.png"
    plt.gcf().savefig(dependence_path, dpi=300, bbox_inches="tight")
    plt.close(plt.gcf())
    paths.append(dependence_path)

    metadata = {
        "model_output_space": "raw XGBoost margin / log-odds-like model output",
        "sample_size": int(len(sample)),
        "top_feature": top_feature,
        "interpretation_guardrail": (
            "SHAP describes the fitted model's output decomposition. It does not establish criminal intent, "
            "prove suspicious activity, or independently justify a suspicious transaction report."
        ),
    }
    return paths, importance, metadata


def save_case_waterfalls(
    config: dict,
    model,
    full_test: pd.DataFrame,
    feature_names: list[str],
    cases: pd.DataFrame,
) -> list[Path]:
    figures = project_path(config, "figures_final")
    explainer = shap.TreeExplainer(model, model_output="raw")
    paths: list[Path] = []
    for _, case in cases.iterrows():
        tx_id = int(case["txId"])
        outcome = str(case["outcome"])
        row = full_test.loc[full_test["txId"] == tx_id]
        if len(row) != 1:
            raise AssertionError(f"Could not uniquely align SHAP case txId={tx_id}.")
        explanation = explainer(row[feature_names])
        plt.figure()
        shap.plots.waterfall(
            explanation[0], max_display=int(config["explainability"]["max_display"]), show=False
        )
        plt.title(f"SHAP waterfall: {outcome.replace('_', ' ')} | txId {tx_id}")
        output = figures / f"case_{outcome}_waterfall.png"
        plt.gcf().savefig(output, dpi=300, bbox_inches="tight")
        plt.close(plt.gcf())
        paths.append(output)
    return paths


def run_permutation_importance(
    config: dict,
    combined_model,
    test: pd.DataFrame,
    combined_features: list[str],
) -> tuple[pd.DataFrame, dict]:
    repeats = int(config["explainability"]["permutation_repeats"])
    seed = int(config["project"]["seed"])
    result = permutation_importance(
        combined_model,
        test[combined_features],
        test["label"],
        scoring="average_precision",
        n_repeats=repeats,
        random_state=seed,
        n_jobs=1,
    )
    importance = pd.DataFrame(
        {
            "feature": combined_features,
            "mean_pr_auc_decrease": result.importances_mean,
            "standard_deviation": result.importances_std,
        }
    ).sort_values("mean_pr_auc_decrease", ascending=False, ignore_index=True)
    importance["rank"] = np.arange(1, len(importance) + 1)
    graph_rows = importance.loc[importance["feature"].isin(PRIMARY_GRAPH_FEATURES)]
    highest_graph = graph_rows.iloc[0].to_dict() if not graph_rows.empty else None
    metadata = {
        "scoring_metric": "average_precision / PR-AUC",
        "number_of_repeats": repeats,
        "seed": seed,
        "model": "best combined XGBoost candidate selected by temporal CV",
        "feature_set": "165 supplied predictors plus five primary topology features",
        "evaluation_split": "labeled final test time steps 41-49",
        "highest_ranked_graph_feature": highest_graph,
        "comparison_note": (
            "Permutation importance measures test-performance degradation after shuffling. SHAP decomposes "
            "model outputs. The methods answer different questions and need not rank variables identically."
        ),
    }
    return importance, metadata


def create_permutation_figure(config: dict, importance: pd.DataFrame) -> Path:
    top = importance.head(15).sort_values("mean_pr_auc_decrease")
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["mean_pr_auc_decrease"], xerr=top["standard_deviation"])
    ax.set_xlabel("Mean decrease in final-test PR-AUC after permutation")
    ax.set_title("Permutation importance: best combined XGBoost candidate")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output = project_path(config, "figures_final") / "permutation_importance_top15.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def run_explainability(config: dict) -> None:
    logger = setup_logging(config, "explainability")
    results = project_path(config, "results_final")
    models = project_path(config, "models_final")
    selection = read_json(results / "operational_model_selection.json")
    selected_feature_set = selection["selected_feature_set"]
    selected_candidate_name = selection["selected_candidate_name"]

    operational_bundle = joblib.load(Path(__file__).resolve().parents[1] / selection["selected_model_artifact"])
    operational_model = operational_bundle["model"]
    operational_features = operational_bundle["feature_names"]
    combined_bundle = joblib.load(Path(__file__).resolve().parents[1] / selection["best_combined_model_artifact"])

    frame = load_modeling_frame(config)
    split = chronological_split(frame, config)
    test = split.test.copy()
    cases = pd.read_csv(results / "case_studies.csv")
    sample = stratified_sample(
        test,
        int(config["explainability"]["shap_sample_size"]),
        int(config["project"]["seed"]),
    )

    shap_paths, shap_importance, shap_metadata = save_shap_plots(
        config, operational_model, sample, operational_features, cases
    )
    waterfall_paths = save_case_waterfalls(
        config, operational_model, test, operational_features, cases
    )
    shap_importance_path = results / "shap_feature_importance.csv"
    shap_metadata_path = results / "shap_metadata.json"
    shap_importance.to_csv(shap_importance_path, index=False)
    write_json(shap_metadata, shap_metadata_path)

    sets = feature_sets(frame)
    permutation, permutation_metadata = run_permutation_importance(
        config,
        combined_bundle["model"],
        test,
        sets["combined"],
    )
    permutation_path = results / "permutation_importance.csv"
    permutation_metadata_path = results / "permutation_importance_metadata.json"
    permutation.to_csv(permutation_path, index=False)
    write_json(permutation_metadata, permutation_metadata_path)
    permutation_figure = create_permutation_figure(config, permutation)

    record_provenance(
        config,
        [
            shap_importance_path,
            shap_metadata_path,
            permutation_path,
            permutation_metadata_path,
            permutation_figure,
            *shap_paths,
            *waterfall_paths,
        ],
        stage="explainability",
        source="src.explainability.run_explainability",
        output_type="SHAP and permutation-importance outputs",
        model_family=f"authoritative {selected_candidate_name} for SHAP; best combined candidate for permutation importance",
        feature_set=selected_feature_set,
        data_split="labeled final test 41-49",
        notes="Anonymized supplied features are not given behavioral interpretations.",
    )
    logger.info("Explainability and importance outputs complete.")


def run(config_path: str | Path) -> None:
    run_explainability(load_config(config_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate model explainability and importance outputs.")
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().config)
