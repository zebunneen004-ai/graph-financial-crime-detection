from __future__ import annotations

import argparse
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
    load_config,
    project_path,
    record_provenance,
    relative_to_root,
    resolve_raw_file,
    sha256_file,
    save_parquet,
    setup_logging,
    validate_columns,
    write_json,
)

FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"


def normalize_classes(classes: pd.DataFrame) -> pd.DataFrame:
    validate_columns(classes, ["txId", "class"], "classes")
    result = classes[["txId", "class"]].copy()
    result["txId"] = pd.to_numeric(result["txId"], errors="raise").astype("int64")
    normalized = result["class"].astype("string").str.strip().str.lower()
    mapping = {"1": 1, "2": 0, "unknown": pd.NA}
    unexpected = sorted(set(normalized.dropna().unique()) - set(mapping))
    if unexpected:
        raise ValueError(f"Unexpected class values: {unexpected}")
    result["label"] = normalized.map(mapping).astype("Int8")
    result["label_status"] = np.where(result["label"].isna(), "unknown", "labeled")
    return result


def feature_names(column_count: int) -> list[str]:
    if column_count < 3:
        raise ValueError(f"Feature file has only {column_count} columns; expected at least 3.")
    return ["txId", "time_step"] + [f"feature_{i}" for i in range(1, column_count - 1)]


def validate_expected_counts(classes: pd.DataFrame, config: dict) -> dict[str, int]:
    expected = config["expected"]
    counts = {
        "nodes": int(len(classes)),
        "illicit": int((classes["label"] == 1).sum()),
        "licit": int((classes["label"] == 0).sum()),
        "unknown": int(classes["label"].isna().sum()),
        "labeled": int(classes["label"].notna().sum()),
    }
    for key, actual in counts.items():
        configured = int(expected[key])
        if actual != configured:
            raise AssertionError(f"Expected {key}={configured:,}, observed {actual:,}.")
    if not classes["txId"].is_unique:
        raise AssertionError("Class file contains duplicate txIds.")
    return counts


def build_processed_tables(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    logger = setup_logging(config, "data_loading")
    feature_path = resolve_raw_file(config, FEATURES_FILE)
    class_path = resolve_raw_file(config, CLASSES_FILE)
    edge_path = resolve_raw_file(config, EDGES_FILE)

    logger.info("Reading class labels from %s", class_path)
    classes_raw = pd.read_csv(class_path, dtype={"txId": "int64", "class": "string"})
    classes = normalize_classes(classes_raw)
    count_summary = validate_expected_counts(classes, config)

    sample = pd.read_csv(feature_path, header=None, nrows=2)
    names = feature_names(sample.shape[1])
    supplied_features = names[2:]
    if len(supplied_features) != int(config["expected"]["supplied_predictors"]):
        raise AssertionError(
            f"Expected {config['expected']['supplied_predictors']} supplied predictors, "
            f"observed {len(supplied_features)}."
        )

    label_lookup = classes.set_index("txId")["label"]
    all_node_chunks: list[pd.DataFrame] = []
    labeled_chunks: list[pd.DataFrame] = []
    logger.info("Streaming the raw feature file in chunks: %s", feature_path)

    for chunk_number, chunk in enumerate(
        pd.read_csv(feature_path, header=None, names=names, chunksize=25_000), start=1
    ):
        chunk["txId"] = pd.to_numeric(chunk["txId"], errors="raise").astype("int64")
        chunk["time_step"] = pd.to_numeric(chunk["time_step"], errors="raise").astype("int16")
        all_node_chunks.append(chunk[["txId", "time_step"]].copy())

        chunk["label"] = chunk["txId"].map(label_lookup)
        labeled = chunk.loc[chunk["label"].notna()].copy()
        labeled["label"] = labeled["label"].astype("int8")
        labeled[supplied_features] = labeled[supplied_features].astype("float32")
        labeled_chunks.append(labeled[["txId", "time_step", *supplied_features, "label"]])
        logger.info("Processed feature chunk %d", chunk_number)

    node_index = pd.concat(all_node_chunks, ignore_index=True)
    labeled_features = pd.concat(labeled_chunks, ignore_index=True)

    expected = config["expected"]
    if len(node_index) != int(expected["nodes"]):
        raise AssertionError(f"Feature file row count mismatch: {len(node_index):,}")
    if len(labeled_features) != int(expected["labeled"]):
        raise AssertionError(f"Labeled feature row count mismatch: {len(labeled_features):,}")
    if not node_index["txId"].is_unique:
        raise AssertionError("Feature file contains duplicate txIds.")
    if set(node_index["txId"]) != set(classes["txId"]):
        raise AssertionError("Feature and class files do not contain identical txId sets.")
    if node_index["time_step"].nunique() != int(expected["time_steps"]):
        raise AssertionError("Unexpected number of time steps.")

    processed = project_path(config, "processed_data")
    node_path = processed / "node_index.parquet"
    labels_path = processed / "labels.parquet"
    labeled_path = processed / "labeled_features.parquet"
    save_parquet(node_index, node_path)
    save_parquet(classes, labels_path)
    save_parquet(labeled_features, labeled_path)

    edge_rows = sum(1 for _ in edge_path.open("rb")) - 1
    if edge_rows != int(expected["edges"]):
        raise AssertionError(f"Expected {expected['edges']:,} edges, observed {edge_rows:,}.")

    local_count = int(config["expected"]["local_supplied_predictors_excluding_time_step"])
    aggregate_count = int(config["expected"]["one_hop_aggregated_predictors"])
    if local_count + aggregate_count != len(supplied_features):
        raise AssertionError("Configured local and one-hop feature groups do not sum to supplied predictor count.")

    manifest = {
        "input_files": {
            "features": {"path": relative_to_root(feature_path), "sha256": sha256_file(feature_path)},
            "classes": {"path": relative_to_root(class_path), "sha256": sha256_file(class_path)},
            "edges": {"path": relative_to_root(edge_path), "sha256": sha256_file(edge_path)},
        },
        "counts": count_summary | {"edges": edge_rows},
        "feature_schema": {
            "txId": "identifier; excluded from predictors",
            "time_step": "chronological split variable; excluded from primary predictors",
            "supplied_predictor_count": len(supplied_features),
            "local_supplied_predictor_count_excluding_time_step": int(config["expected"]["local_supplied_predictors_excluding_time_step"]),
            "one_hop_aggregated_predictor_count": int(config["expected"]["one_hop_aggregated_predictors"]),
            "local_supplied_predictors": supplied_features[: int(config["expected"]["local_supplied_predictors_excluding_time_step"])],
            "one_hop_aggregated_predictors": supplied_features[int(config["expected"]["local_supplied_predictors_excluding_time_step"]):],
            "supplied_predictors": supplied_features,
        },
    }
    manifest_path = project_path(config, "results_final") / "data_manifest.json"
    write_json(manifest, manifest_path)

    record_provenance(
        config,
        [node_path, labels_path, labeled_path, manifest_path],
        stage="data_loading",
        source="src.data_loading.build_processed_tables",
        output_type="processed data and manifest",
        notes="Unknown labels retained in graph node index and excluded from supervised table.",
    )
    logger.info("Processed data tables saved successfully.")
    return node_index, classes, labeled_features, supplied_features


def create_eda_outputs(
    config: dict,
    node_index: pd.DataFrame,
    classes: pd.DataFrame,
    labeled_features: pd.DataFrame,
) -> None:
    logger = setup_logging(config, "data_loading")
    figures = project_path(config, "figures_final")
    results = project_path(config, "results_final")

    label_counts = pd.DataFrame(
        {
            "label": ["Unknown", "Licit-labeled", "Illicit-labeled"],
            "count": [
                int(classes["label"].isna().sum()),
                int((classes["label"] == 0).sum()),
                int((classes["label"] == 1).sum()),
            ],
        }
    )
    label_counts["percentage_of_all_nodes"] = label_counts["count"] / len(classes)
    label_table_path = results / "label_distribution.csv"
    label_counts.to_csv(label_table_path, index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(label_counts["label"], label_counts["count"])
    ax.set_ylabel("Transactions")
    ax.set_title("Elliptic label availability")
    ax.ticklabel_format(style="plain", axis="y")
    for bar, value in zip(bars, label_counts["count"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    label_figure_path = figures / "label_distribution.png"
    fig.savefig(label_figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    all_counts = node_index.groupby("time_step").size().rename("all_transactions")
    labeled_time = (
        labeled_features.groupby("time_step")["label"]
        .agg(labeled_transactions="count", illicit_labeled="sum")
    )
    timeline = all_counts.to_frame().join(labeled_time, how="left").fillna(0).reset_index()
    timeline["illicit_prevalence_among_labeled"] = (
        timeline["illicit_labeled"] / timeline["labeled_transactions"].replace(0, np.nan)
    )
    timeline_path = results / "transactions_by_time_step.csv"
    timeline.to_csv(timeline_path, index=False)

    overall_prevalence = labeled_features["label"].mean()
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.bar(timeline["time_step"], timeline["all_transactions"], alpha=0.75)
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("All transactions in processed snapshot")
    ax2 = ax1.twinx()
    ax2.plot(
        timeline["time_step"],
        timeline["illicit_prevalence_among_labeled"],
        marker="o",
        linewidth=1.5,
    )
    ax2.axhline(overall_prevalence, linestyle="--", linewidth=1.5, label=f"Overall labeled prevalence = {overall_prevalence:.1%}")
    ax2.set_ylabel("Illicit prevalence among labeled transactions")
    ax2.legend(loc="upper right")
    ax1.set_title("Transactions and labeled illicit prevalence by time step")
    fig.tight_layout()
    timeline_figure_path = figures / "transactions_by_timestep.png"
    fig.savefig(timeline_figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    record_provenance(
        config,
        [label_table_path, label_figure_path, timeline_path, timeline_figure_path],
        stage="data_loading",
        source="src.data_loading.create_eda_outputs",
        output_type="EDA table and figure",
        data_split="all nodes for counts; labeled observations for prevalence",
        notes="Unknown nodes remain in graph population but are excluded from supervised prevalence.",
    )
    logger.info("EDA outputs created.")


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    node_index, classes, labeled_features, _ = build_processed_tables(config)
    create_eda_outputs(config, node_index, classes, labeled_features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare memory-efficient Elliptic data tables.")
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config)
