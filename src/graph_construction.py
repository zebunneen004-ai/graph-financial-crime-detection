from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import networkx as nx
import pandas as pd

from src.common import (
    load_config,
    project_path,
    record_provenance,
    resolve_raw_file,
    setup_logging,
    write_json,
)


def load_edges(config: dict) -> pd.DataFrame:
    edge_path = resolve_raw_file(config, "elliptic_txs_edgelist.csv")
    edges = pd.read_csv(edge_path, dtype={"txId1": "int64", "txId2": "int64"})
    required = {"txId1", "txId2"}
    if not required.issubset(edges.columns):
        raise ValueError(f"Edge file must contain {sorted(required)}; got {list(edges.columns)}")
    return edges


def audit_graph(config: dict) -> dict:
    logger = setup_logging(config, "graph_construction")
    processed = project_path(config, "processed_data")
    node_index = pd.read_parquet(processed / "node_index.parquet")
    edges = load_edges(config)
    expected = config["expected"]

    if len(node_index) != int(expected["nodes"]):
        raise AssertionError("Unexpected node count before graph construction.")
    if len(edges) != int(expected["edges"]):
        raise AssertionError("Unexpected edge count before graph construction.")

    node_ids = pd.Index(node_index["txId"], dtype="int64")
    missing_sources = int((~edges["txId1"].isin(node_ids)).sum())
    missing_targets = int((~edges["txId2"].isin(node_ids)).sum())
    if missing_sources or missing_targets:
        raise AssertionError(
            f"Edges reference missing nodes: sources={missing_sources}, targets={missing_targets}."
        )

    time_lookup = node_index.set_index("txId")["time_step"]
    source_time = edges["txId1"].map(time_lookup)
    target_time = edges["txId2"].map(time_lookup)
    cross_time_mask = source_time.ne(target_time)
    self_loop_mask = edges["txId1"].eq(edges["txId2"])

    edge_time_summary = pd.DataFrame(
        {
            "edge_type": ["same_time_step", "cross_time_step", "self_loop"],
            "count": [
                int((~cross_time_mask).sum()),
                int(cross_time_mask.sum()),
                int(self_loop_mask.sum()),
            ],
        }
    )
    edge_time_summary["percentage_of_edges"] = edge_time_summary["count"] / len(edges)
    edge_time_path = project_path(config, "results_final") / "edge_time_consistency.csv"
    edge_time_summary.to_csv(edge_time_path, index=False)

    if int(cross_time_mask.sum()) != 0:
        raise AssertionError("Cross-time-step edges were detected; snapshot assumptions are invalid.")
    if int(self_loop_mask.sum()) != 0:
        raise AssertionError("Self-loops were detected; DAG claim must not be made.")

    logger.info("Constructing directed graph with all labeled and unknown nodes.")
    graph = nx.DiGraph()
    graph.add_nodes_from(node_index["txId"].to_numpy())
    graph.add_edges_from(edges[["txId1", "txId2"]].itertuples(index=False, name=None))

    node_count = graph.number_of_nodes()
    edge_count = graph.number_of_edges()
    weak_components = list(nx.weakly_connected_components(graph))
    weak_sizes = sorted((len(component) for component in weak_components), reverse=True)
    is_dag = nx.is_directed_acyclic_graph(graph)
    strong_component_count = nx.number_strongly_connected_components(graph)

    in_degrees = dict(graph.in_degree())
    out_degrees = dict(graph.out_degree())
    observed_sources = sum(value == 0 for value in in_degrees.values())
    observed_sinks = sum(value == 0 for value in out_degrees.values())

    snapshot_counts = node_index.groupby("time_step").size().sort_index()
    component_size_counts = pd.Series(weak_sizes).value_counts().to_dict()

    audit = {
        "node_count": node_count,
        "edge_count": edge_count,
        "density": nx.density(graph),
        "mean_in_degree": sum(in_degrees.values()) / node_count,
        "mean_out_degree": sum(out_degrees.values()) / node_count,
        "weak_component_count": len(weak_components),
        "strong_component_count": strong_component_count,
        "largest_weak_component_size": weak_sizes[0],
        "weak_component_sizes_descending": weak_sizes,
        "observed_sources_in_subgraph": observed_sources,
        "observed_sinks_in_subgraph": observed_sinks,
        "self_loop_count": int(self_loop_mask.sum()),
        "same_time_step_edge_count": int((~cross_time_mask).sum()),
        "cross_time_step_edge_count": int(cross_time_mask.sum()),
        "same_time_step_edge_percentage": float((~cross_time_mask).mean()),
        "is_directed_acyclic_graph": bool(is_dag),
        "time_step_count": int(node_index["time_step"].nunique()),
        "snapshot_sizes": {str(k): int(v) for k, v in snapshot_counts.items()},
        "component_size_frequency": {str(k): int(v) for k, v in component_size_counts.items()},
        "interpretation_guardrail": (
            "The processed graph consists of temporally separated benchmark snapshots. "
            "Observed sources and sinks refer only to the Elliptic subgraph and must not be "
            "automatically interpreted as mining origins, cold storage, or final endpoints."
        ),
    }

    if node_count != int(expected["nodes"]) or edge_count != int(expected["edges"]):
        raise AssertionError("NetworkX graph counts do not match the raw data contract.")
    if not is_dag:
        raise AssertionError("The processed graph is not a DAG; revise all DAG-related language.")

    audit_path = project_path(config, "results_final") / "graph_audit.json"
    write_json(audit, audit_path)
    record_provenance(
        config,
        [edge_time_path, audit_path],
        stage="graph_construction",
        source="src.graph_construction.audit_graph",
        output_type="graph audit",
        data_split="all 203,769 nodes, including unknown labels",
        notes="Every edge checked; no sampling used.",
    )
    logger.info("Graph audit complete. DAG=%s, weak components=%d", is_dag, len(weak_components))
    return audit


def run(config_path: str | Path) -> None:
    audit_graph(load_config(config_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit the complete Elliptic directed graph.")
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args().config)
