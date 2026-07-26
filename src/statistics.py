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
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from src.common import load_config, project_path, record_provenance, setup_logging
from src.graph_features import PRIMARY_GRAPH_FEATURES


def rank_biserial_from_u(u_statistic: float, n_illicit: int, n_licit: int) -> float:
    """Positive values mean illicit-labeled observations tend to have larger values."""
    denominator = n_illicit * n_licit
    return float((2.0 * u_statistic / denominator) - 1.0) if denominator else float("nan")


def effect_label(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 0.10:
        return "negligible"
    if magnitude < 0.30:
        return "small"
    if magnitude < 0.50:
        return "moderate"
    return "large"


def summarize_feature(frame: pd.DataFrame, feature: str) -> dict:
    illicit = frame.loc[frame["label"] == 1, feature].dropna().to_numpy()
    licit = frame.loc[frame["label"] == 0, feature].dropna().to_numpy()
    test = mannwhitneyu(illicit, licit, alternative="two-sided", method="asymptotic")
    effect = rank_biserial_from_u(test.statistic, len(illicit), len(licit))
    return {
        "feature": feature,
        "n_illicit": int(len(illicit)),
        "n_licit": int(len(licit)),
        "illicit_median": float(np.median(illicit)),
        "illicit_q1": float(np.quantile(illicit, 0.25)),
        "illicit_q3": float(np.quantile(illicit, 0.75)),
        "licit_median": float(np.median(licit)),
        "licit_q1": float(np.quantile(licit, 0.25)),
        "licit_q3": float(np.quantile(licit, 0.75)),
        "mann_whitney_u": float(test.statistic),
        "raw_p_value": float(test.pvalue),
        "rank_biserial_correlation": effect,
        "effect_magnitude": effect_label(effect),
        "direction": "higher among illicit-labeled" if effect > 0 else "lower among illicit-labeled",
    }


def pooled_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    results = pd.DataFrame([summarize_feature(frame, feature) for feature in PRIMARY_GRAPH_FEATURES])
    _, adjusted, _, _ = multipletests(results["raw_p_value"], method="holm")
    results["holm_adjusted_p_value"] = adjusted
    results["interpretation"] = results.apply(
        lambda row: (
            f"{row['effect_magnitude'].capitalize()} pooled difference; {row['direction']}. "
            "This pooled comparison is descriptive and may be temporally confounded."
        ),
        axis=1,
    )
    return results


def within_time_analysis(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for time_step, group in frame.groupby("time_step", sort=True):
        if group["label"].nunique() < 2:
            continue
        for feature in PRIMARY_GRAPH_FEATURES:
            summary = summarize_feature(group, feature)
            summary["time_step"] = int(time_step)
            rows.append(summary)
    result = pd.DataFrame(rows)
    if result.empty:
        raise AssertionError("No time step contained both classes for within-time analysis.")
    result["holm_adjusted_p_value_within_time_step"] = np.nan
    for time_step, indices in result.groupby("time_step").groups.items():
        _, adjusted, _, _ = multipletests(result.loc[indices, "raw_p_value"], method="holm")
        result.loc[indices, "holm_adjusted_p_value_within_time_step"] = adjusted
    return result


def create_statistical_figures(
    config: dict, frame: pd.DataFrame, pooled: pd.DataFrame, within_time: pd.DataFrame
) -> list[Path]:
    figures = project_path(config, "figures_final")
    output_paths: list[Path] = []

    transformations = {
        "in_degree": (lambda x: np.log1p(x), "log1p(in-degree)"),
        "out_degree": (lambda x: np.log1p(x), "log1p(out-degree)"),
        "pagerank_relative": (lambda x: np.log1p(x), "log1p(snapshot-normalized PageRank)"),
        "clustering": (lambda x: x, "clustering coefficient"),
        "k_core": (lambda x: np.log1p(x), "log1p(k-core)"),
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()
    for axis, feature in zip(axes, PRIMARY_GRAPH_FEATURES):
        transform, label = transformations[feature]
        licit = transform(frame.loc[frame["label"] == 0, feature].to_numpy())
        illicit = transform(frame.loc[frame["label"] == 1, feature].to_numpy())
        axis.boxplot([licit, illicit], tick_labels=["Licit-labeled", "Illicit-labeled"], showfliers=False)
        row = pooled.loc[pooled["feature"] == feature].iloc[0]
        axis.set_title(
            f"{feature}\nHolm p={row['holm_adjusted_p_value']:.2e}; r_rb={row['rank_biserial_correlation']:.3f}"
        )
        axis.set_ylabel(label)
    axes[-1].axis("off")
    fig.suptitle("Primary topology features: pooled descriptive comparisons", fontsize=14)
    fig.tight_layout()
    pooled_path = figures / "graph_feature_distributions.png"
    fig.savefig(pooled_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(pooled_path)

    pivot = within_time.pivot(index="feature", columns="time_step", values="rank_biserial_correlation")
    pivot = pivot.reindex(PRIMARY_GRAPH_FEATURES)
    fig, ax = plt.subplots(figsize=(14, 4.5))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", vmin=-0.5, vmax=0.5, cmap="coolwarm")
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns, rotation=90)
    ax.set_xlabel("Time step")
    ax.set_title("Within-time-step rank-biserial effects")
    fig.colorbar(image, ax=ax, label="Rank-biserial correlation")
    fig.tight_layout()
    within_path = figures / "within_time_effects.png"
    fig.savefig(within_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(within_path)

    context = (
        frame.groupby("time_step")
        .agg(
            snapshot_size=("weak_component_size", "first"),
            weak_component_unique=("weak_component_size", "nunique"),
            labeled_count=("label", "count"),
            illicit_prevalence=("label", "mean"),
        )
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(context["snapshot_size"], context["illicit_prevalence"])
    for _, row in context.iterrows():
        ax.annotate(str(int(row["time_step"])), (row["snapshot_size"], row["illicit_prevalence"]), fontsize=7)
    ax.set_xlabel("Snapshot size / weak_component_size")
    ax.set_ylabel("Illicit prevalence among labeled observations")
    ax.set_title("Snapshot-level context: size and labeled prevalence")
    fig.tight_layout()
    context_path = figures / "weak_component_snapshot_context.png"
    fig.savefig(context_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(context_path)
    return output_paths


def run_analysis(config: dict) -> None:
    logger = setup_logging(config, "statistics")
    processed = project_path(config, "processed_data")
    labeled = pd.read_parquet(processed / "labeled_features.parquet", columns=["txId", "time_step", "label"])
    graph = pd.read_parquet(processed / "graph_features.parquet")
    frame = labeled.merge(graph, on=["txId", "time_step"], how="inner", validate="one_to_one")
    if len(frame) != int(config["expected"]["labeled"]):
        raise AssertionError("Statistical analysis did not retain all labeled observations.")

    pooled = pooled_analysis(frame)
    within_time = within_time_analysis(frame)
    snapshot_context = (
        frame.groupby("time_step")
        .agg(
            snapshot_size=("weak_component_size", "first"),
            weak_component_size_unique_values=("weak_component_size", "nunique"),
            labeled_count=("label", "count"),
            illicit_count=("label", "sum"),
            illicit_prevalence=("label", "mean"),
        )
        .reset_index()
    )

    results = project_path(config, "results_final")
    pooled_path = results / "graph_feature_pooled_statistics.csv"
    within_path = results / "graph_feature_within_time_statistics.csv"
    context_path = results / "weak_component_snapshot_context.csv"
    pooled.to_csv(pooled_path, index=False)
    within_time.to_csv(within_path, index=False)
    snapshot_context.to_csv(context_path, index=False)
    figure_paths = create_statistical_figures(config, frame, pooled, within_time)

    record_provenance(
        config,
        [pooled_path, within_path, context_path, *figure_paths],
        stage="statistics",
        source="src.statistics.run_analysis",
        output_type="statistical result and figure",
        data_split="labeled observations; pooled and within-time-step descriptive analyses",
        notes="Holm-adjusted tests and correctly named rank-biserial correlation.",
    )
    logger.info("Corrected statistical analysis complete.")


def run(config_path: str | Path) -> None:
    run_analysis(load_config(config_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run corrected graph-feature statistical analyses.")
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().config)
