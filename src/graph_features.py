from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import numpy as np
import pandas as pd

from src.common import (
    load_config,
    project_path,
    record_provenance,
    save_parquet,
    setup_logging,
    write_json,
)
from src.graph_construction import load_edges

# PageRank is computed independently within each completed time-step snapshot and
# divided by the uniform baseline (1 / snapshot_size). A value of 1 therefore
# equals the snapshot mean, making the feature comparable across snapshots.
PRIMARY_GRAPH_FEATURES = [
    "in_degree",
    "out_degree",
    "pagerank_relative",
    "clustering",
    "k_core",
]
DESCRIPTIVE_GRAPH_FEATURES = ["total_degree", "pagerank_raw_snapshot"]
SENSITIVITY_GRAPH_FEATURES = ["weak_component_size"]


def compute_degree_features(nodes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes["txId"].to_numpy())
    graph.add_edges_from(edges[["txId1", "txId2"]].itertuples(index=False, name=None))
    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())

    result = nodes[["txId", "time_step"]].copy()
    result["in_degree"] = result["txId"].map(in_degree).fillna(0).astype("int32")
    result["out_degree"] = result["txId"].map(out_degree).fillna(0).astype("int32")
    result["total_degree"] = result["in_degree"] + result["out_degree"]
    del graph, in_degree, out_degree
    gc.collect()
    return result


def compute_snapshot_features(
    config: dict, nodes: pd.DataFrame, edges: pd.DataFrame
) -> tuple[pd.DataFrame, dict, dict]:
    logger = setup_logging(config, "graph_features")
    time_lookup = nodes.set_index("txId")["time_step"]
    edges = edges.copy()
    edges["time_step_source"] = edges["txId1"].map(time_lookup).astype("int16")
    edges["time_step_target"] = edges["txId2"].map(time_lookup).astype("int16")
    if not edges["time_step_source"].eq(edges["time_step_target"]).all():
        raise AssertionError("Snapshot feature calculation found cross-time-step edges.")

    graph_cfg = config["graph"]
    alpha = float(graph_cfg["pagerank_alpha"])
    tolerance = float(graph_cfg["pagerank_tolerance"])
    maximum_iterations = int(graph_cfg["pagerank_max_iterations"])

    rows: list[pd.DataFrame] = []
    validation_rows: list[dict] = []
    pagerank_rows: list[dict] = []

    for time_step, snapshot_nodes in nodes.groupby("time_step", sort=True):
        node_ids = snapshot_nodes["txId"].to_numpy()
        snapshot_edges = edges.loc[
            edges["time_step_source"].eq(time_step), ["txId1", "txId2"]
        ]

        directed = nx.DiGraph()
        directed.add_nodes_from(node_ids)
        directed.add_edges_from(snapshot_edges.itertuples(index=False, name=None))
        try:
            raw_pagerank = nx.pagerank(
                directed,
                alpha=alpha,
                tol=tolerance,
                max_iter=maximum_iterations,
            )
        except nx.PowerIterationFailedConvergence as exc:
            raise RuntimeError(
                f"PageRank failed to converge for time step {time_step}."
            ) from exc

        snapshot_size = len(snapshot_nodes)
        uniform_baseline = 1.0 / snapshot_size

        undirected = nx.Graph()
        undirected.add_nodes_from(node_ids)
        undirected.add_edges_from(snapshot_edges.itertuples(index=False, name=None))
        clustering = nx.clustering(undirected)
        core_number = nx.core_number(undirected) if undirected.number_of_nodes() else {}
        components = list(nx.connected_components(undirected))
        component_sizes: dict[int, int] = {}
        for component in components:
            size = len(component)
            component_sizes.update({int(node): size for node in component})

        snapshot_result = snapshot_nodes[["txId", "time_step"]].copy()
        snapshot_result["pagerank_raw_snapshot"] = (
            snapshot_result["txId"].map(raw_pagerank).fillna(0.0).astype("float64")
        )
        snapshot_result["pagerank_relative"] = (
            snapshot_result["pagerank_raw_snapshot"] / uniform_baseline
        ).astype("float32")
        snapshot_result["clustering"] = (
            snapshot_result["txId"].map(clustering).fillna(0.0).astype("float32")
        )
        snapshot_result["k_core"] = (
            snapshot_result["txId"].map(core_number).fillna(0).astype("int16")
        )
        snapshot_result["weak_component_size"] = (
            snapshot_result["txId"].map(component_sizes).fillna(1).astype("int32")
        )
        rows.append(snapshot_result)

        raw_values = snapshot_result["pagerank_raw_snapshot"].to_numpy()
        relative_values = snapshot_result["pagerank_relative"].to_numpy()
        pagerank_rows.append(
            {
                "time_step": int(time_step),
                "snapshot_node_count": int(snapshot_size),
                "raw_score_sum": float(raw_values.sum()),
                "raw_score_mean": float(raw_values.mean()),
                "uniform_baseline": float(uniform_baseline),
                "relative_score_mean": float(relative_values.mean()),
                "relative_score_min": float(relative_values.min()),
                "relative_score_max": float(relative_values.max()),
                "completed_without_convergence_error": True,
            }
        )
        validation_rows.append(
            {
                "time_step": int(time_step),
                "snapshot_node_count": int(snapshot_size),
                "snapshot_edge_count": int(len(snapshot_edges)),
                "connected_component_count": int(len(components)),
                "weak_component_size_unique_values": int(
                    snapshot_result["weak_component_size"].nunique()
                ),
                "weak_component_size_equals_snapshot_size_for_all_nodes": bool(
                    snapshot_result["weak_component_size"].eq(snapshot_size).all()
                ),
            }
        )
        logger.info("Computed directed and undirected features for time step %s", time_step)
        del directed, undirected, raw_pagerank, clustering, core_number, components, component_sizes
        gc.collect()

    result = pd.concat(rows, ignore_index=True)
    validation = pd.DataFrame(validation_rows)
    pagerank_validation = pd.DataFrame(pagerank_rows)

    weak_component_summary = {
        "all_time_steps_have_one_weak_component_size_value": bool(
            validation["weak_component_size_unique_values"].eq(1).all()
        ),
        "weak_component_size_equals_snapshot_size_everywhere": bool(
            validation["weak_component_size_equals_snapshot_size_for_all_nodes"].all()
        ),
        "interpretation": (
            "When weak_component_size equals snapshot size for every node, it is a snapshot-level "
            "context variable and time-step proxy, not transaction-specific isolation."
        ),
        "per_time_step": validation.to_dict(orient="records"),
    }
    pagerank_summary = {
        "scope": "computed independently within each completed time-step snapshot",
        "primary_feature": "pagerank_relative = raw snapshot PageRank / (1 / snapshot node count)",
        "descriptive_feature": "pagerank_raw_snapshot",
        "alpha": alpha,
        "tolerance": tolerance,
        "maximum_iterations": maximum_iterations,
        "all_snapshot_raw_sums_close_to_one": bool(
            np.allclose(pagerank_validation["raw_score_sum"], 1.0, atol=1e-7)
        ),
        "all_snapshot_relative_means_close_to_one": bool(
            np.allclose(pagerank_validation["relative_score_mean"], 1.0, atol=1e-6)
        ),
        "interpretation": (
            "Snapshot normalization prevents the pooled model from using PageRank's 1/N scaling as "
            "an indirect snapshot-size signal. A relative value of 1 equals the snapshot mean."
        ),
        "per_time_step": pagerank_validation.to_dict(orient="records"),
    }
    return result, weak_component_summary, pagerank_summary


def build_graph_features(config: dict) -> pd.DataFrame:
    logger = setup_logging(config, "graph_features")
    processed = project_path(config, "processed_data")
    nodes = pd.read_parquet(processed / "node_index.parquet")
    edges = load_edges(config)

    directed = compute_degree_features(nodes, edges)
    snapshot, weak_component_validation, pagerank_metadata = compute_snapshot_features(
        config, nodes, edges
    )
    final = directed.merge(
        snapshot,
        on=["txId", "time_step"],
        how="inner",
        validate="one_to_one",
    )

    required = [
        "txId",
        "time_step",
        *PRIMARY_GRAPH_FEATURES,
        *DESCRIPTIVE_GRAPH_FEATURES,
        *SENSITIVITY_GRAPH_FEATURES,
    ]
    if final[required].isna().any().any():
        raise AssertionError("Graph feature output contains missing values.")
    if not final["total_degree"].eq(final["in_degree"] + final["out_degree"]).all():
        raise AssertionError("total_degree does not equal in_degree + out_degree.")
    if len(final) != int(config["expected"]["nodes"]):
        raise AssertionError("Graph feature row count does not match node count.")

    final_path = processed / "graph_features.parquet"
    final_csv_path = processed / "graph_features.csv"
    save_parquet(final, final_path)
    final.to_csv(final_csv_path, index=False)

    metadata = {
        "primary_model_features": PRIMARY_GRAPH_FEATURES,
        "descriptive_only_features": DESCRIPTIVE_GRAPH_FEATURES,
        "sensitivity_context_features": SENSITIVITY_GRAPH_FEATURES,
        "total_degree_exclusion_reason": "Exact deterministic sum of in_degree and out_degree.",
        "feature_availability_limitation": (
            "Features are calculated over completed Elliptic graph snapshots. Out-degree and downstream "
            "topology may not be causally available when a transaction first appears. The project is a "
            "retrospective batch classification study, not a live-streaming deployment."
        ),
        "pagerank": pagerank_metadata,
        "weak_component_validation": weak_component_validation,
    }
    metadata_path = project_path(config, "results_final") / "graph_feature_metadata.json"
    write_json(metadata, metadata_path)

    record_provenance(
        config,
        [final_path, final_csv_path, metadata_path],
        stage="graph_features",
        source="src.graph_features.build_graph_features",
        output_type="processed graph features",
        data_split="all nodes and completed graph snapshots",
        notes=(
            "Primary models use five node-level features, including snapshot-normalized PageRank; "
            "total_degree excluded; weak_component_size separate."
        ),
    )
    logger.info("Graph features written to %s", final_path)
    return final


def run(config_path: str | Path) -> None:
    build_graph_features(load_config(config_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute corrected graph features.")
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().config)
