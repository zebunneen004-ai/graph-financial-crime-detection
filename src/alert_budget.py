from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.common import (
    get_run_context,
    load_config,
    project_path,
    read_json,
    record_provenance,
    save_parquet,
    setup_logging,
    write_json,
)


def evaluate_budget(frame: pd.DataFrame, budget: float) -> dict:
    if not 0 < budget <= 1:
        raise ValueError("Alert budget must be in (0, 1].")
    ranked = frame.sort_values(["raw_score", "txId"], ascending=[False, True]).reset_index(drop=True)
    alerts = max(1, int(math.ceil(len(ranked) * budget)))
    alerted = ranked.iloc[:alerts]
    total_positive = int(ranked["true_label"].sum())
    true_positive = int(alerted["true_label"].sum())
    false_positive = alerts - true_positive
    false_negative = total_positive - true_positive
    precision = true_positive / alerts if alerts else 0.0
    recall = true_positive / total_positive if total_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    prevalence = float(ranked["true_label"].mean())
    return {
        "budget": float(budget),
        "labeled_population": int(len(ranked)),
        "alerts": int(alerts),
        "true_positives": true_positive,
        "false_positives": false_positive,
        "false_negatives": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "false_positives_per_true_positive": float(false_positive / true_positive) if true_positive else float("inf"),
        "lift_over_labeled_prevalence": float(precision / prevalence) if prevalence else 0.0,
        "labeled_prevalence": prevalence,
    }


def budget_table(frame: pd.DataFrame, budgets: list[float]) -> pd.DataFrame:
    result = pd.DataFrame([evaluate_budget(frame, budget) for budget in budgets]).sort_values("budget")
    result["marginal_alerts_per_additional_true_positive"] = np.nan
    for current_index in range(1, len(result)):
        alert_delta = result.iloc[current_index]["alerts"] - result.iloc[current_index - 1]["alerts"]
        tp_delta = result.iloc[current_index]["true_positives"] - result.iloc[current_index - 1]["true_positives"]
        result.iloc[current_index, result.columns.get_loc("marginal_alerts_per_additional_true_positive")] = (
            alert_delta / tp_delta if tp_delta > 0 else float("inf")
        )
    return result


def select_budget(validation_table: pd.DataFrame) -> float:
    ordered = validation_table.sort_values(["f1", "budget"], ascending=[False, True])
    return float(ordered.iloc[0]["budget"])


def mark_selected_alerts(frame: pd.DataFrame, budget: float) -> pd.DataFrame:
    result = frame.copy()
    result["rank"] = result.groupby("split")["raw_score"].rank(method="first", ascending=False).astype("int32")
    result["selected_alert"] = False
    for split_name, indices in result.groupby("split").groups.items():
        alert_count = max(1, int(math.ceil(len(indices) * budget)))
        ranked_indices = result.loc[indices].sort_values(["raw_score", "txId"], ascending=[False, True]).index[:alert_count]
        result.loc[ranked_indices, "selected_alert"] = True
    return result


def select_case_studies(test: pd.DataFrame) -> pd.DataFrame:
    result = test.copy()
    result["outcome"] = np.select(
        [
            result["selected_alert"] & result["true_label"].eq(1),
            result["selected_alert"] & result["true_label"].eq(0),
            ~result["selected_alert"] & result["true_label"].eq(1),
            ~result["selected_alert"] & result["true_label"].eq(0),
        ],
        ["true_positive", "false_positive", "false_negative", "true_negative"],
        default="unknown",
    )
    cases = []
    for outcome in ("true_positive", "false_positive", "false_negative", "true_negative"):
        candidates = result.loc[result["outcome"] == outcome]
        if candidates.empty:
            continue
        if outcome == "true_negative":
            chosen = candidates.sort_values("raw_score").iloc[0]
        else:
            chosen = candidates.sort_values("raw_score", ascending=False).iloc[0]
        cases.append(chosen)
    return pd.DataFrame(cases)


def cautious_network_description(row: pd.Series) -> str:
    return (
        f"Within the completed Elliptic snapshot, this transaction has in-degree {int(row['in_degree'])}, "
        f"out-degree {int(row['out_degree'])}, snapshot-normalized PageRank {float(row['pagerank_relative']):.3f}, "
        f"clustering {float(row['clustering']):.3f}, and k-core {int(row['k_core'])}. "
        "These values provide descriptive network context only and do not establish layering, evasion, "
        "criminal intent, or a reporting decision."
    )


def create_alert_figure(config: dict, table: pd.DataFrame, split_name: str, selected_budget: float) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    x = table["budget"] * 100
    for axis, metric, title in (
        (axes[0], "precision", "Precision"),
        (axes[1], "recall", "Recall"),
        (axes[2], "f1", "F1"),
    ):
        axis.plot(x, table[metric], marker="o")
        axis.axvline(selected_budget * 100, linestyle="--", label="Operating-block-selected budget")
        axis.set_xlabel("Alert budget (%)")
        axis.set_ylabel(metric.title())
        axis.set_title(title)
        axis.grid(alpha=0.25)
    axes[0].legend()
    fig.suptitle(f"Alert-prioritization trade-offs: labeled {split_name} set")
    fig.tight_layout()
    output = project_path(config, "figures_final") / f"alert_budget_{split_name}.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def run_alert_analysis(config: dict) -> None:
    logger = setup_logging(config, "alert_budget")
    results = project_path(config, "results_final")
    predictions = pd.read_parquet(results / "calibrated_operational_predictions.parquet")
    operating = predictions.loc[predictions["split"] == "operating"].copy()
    test = predictions.loc[predictions["split"] == "test"].copy()
    if operating.empty or test.empty:
        raise AssertionError("Calibrated predictions must include operating and final-test rows.")
    budgets = [float(value) for value in config["alerts"]["candidate_budgets"]]

    operating_table = budget_table(operating, budgets)
    selected_budget = select_budget(operating_table)
    test_table = budget_table(test, budgets)
    operating_table["selected_on_operating_block"] = operating_table["budget"].eq(selected_budget)
    test_table["selected_on_operating_block"] = test_table["budget"].eq(selected_budget)

    authoritative = mark_selected_alerts(predictions, selected_budget)
    authoritative["run_id"] = get_run_context().run_id
    authoritative["timestamp_utc"] = get_run_context().timestamp_utc
    authoritative = authoritative.rename(columns={"true_label": "true_label"})

    graph = pd.read_parquet(project_path(config, "processed_data") / "graph_features.parquet")
    cases = select_case_studies(authoritative.loc[authoritative["split"] == "test"])
    cases = cases.merge(graph, on=["txId", "time_step"], how="left", validate="one_to_one")
    cases["network_description"] = cases.apply(cautious_network_description, axis=1)

    operating_path = results / "operating_alert_budget_analysis.csv"
    test_path = results / "final_test_alert_budget_analysis.csv"
    authoritative_path = results / "authoritative_test_predictions.parquet"
    all_predictions_path = results / "authoritative_operating_and_test_predictions.parquet"
    cases_path = results / "case_studies.csv"
    selection_path = results / "alert_budget_selection.json"

    operating_table.to_csv(operating_path, index=False)
    test_table.to_csv(test_path, index=False)
    save_parquet(authoritative.loc[authoritative["split"] == "test"], authoritative_path)
    save_parquet(authoritative, all_predictions_path)
    cases.to_csv(cases_path, index=False)
    selected_test_row = test_table.loc[test_table["budget"].eq(selected_budget)].iloc[0].to_dict()
    write_json(
        {
            "candidate_budgets": budgets,
            "selection_rule": config["alerts"]["selection_rule"],
            "selection_population": "labeled operating-selection time steps 36-40",
            "selected_budget": selected_budget,
            "application_population": "labeled final test time steps 41-49",
            "selected_budget_final_test_metrics": selected_test_row,
            "warning": (
                "The selected percentage is an experiment-specific operating point, not a universally optimal "
                "budget for AML institutions."
            ),
        },
        selection_path,
    )

    figure_paths = [
        create_alert_figure(config, operating_table, "operating", selected_budget),
        create_alert_figure(config, test_table, "final_test", selected_budget),
    ]
    record_provenance(
        config,
        [
            operating_path,
            test_path,
            authoritative_path,
            all_predictions_path,
            cases_path,
            selection_path,
            *figure_paths,
        ],
        stage="alert_budget",
        source="src.alert_budget.run_alert_analysis",
        output_type="alert-budget results, authoritative predictions, and case selection",
        calibration_status="calibrated probabilities available; ranking uses raw model score",
        data_split="calibration fit 31-35; budget selected 36-40 and frozen before 41-49 application",
        prediction_file=str(authoritative_path),
        notes="All final operational figures must derive from the authoritative prediction file.",
    )
    logger.info("Selected %.1f%% alert budget on the operating block and applied it unchanged to final test.", selected_budget * 100)


def run(config_path: str | Path) -> None:
    run_alert_analysis(load_config(config_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select and evaluate an operating-block-frozen alert budget.")
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().config)
