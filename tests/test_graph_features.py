import networkx as nx
import numpy as np
import pandas as pd

from src.graph_features import PRIMARY_GRAPH_FEATURES


def test_toy_graph_degree_and_core_contract():
    graph = nx.DiGraph([(1, 2), (2, 3), (2, 4)])
    assert graph.in_degree(2) == 1
    assert graph.out_degree(2) == 2
    undirected = graph.to_undirected()
    cores = nx.core_number(undirected)
    assert cores[2] >= 1


def test_primary_features_exclude_redundant_and_snapshot_context_variables():
    assert "total_degree" not in PRIMARY_GRAPH_FEATURES
    assert "weak_component_size" not in PRIMARY_GRAPH_FEATURES
    assert "pagerank_raw_snapshot" not in PRIMARY_GRAPH_FEATURES
    assert PRIMARY_GRAPH_FEATURES == [
        "in_degree",
        "out_degree",
        "pagerank_relative",
        "clustering",
        "k_core",
    ]


def test_total_degree_is_exactly_redundant():
    frame = pd.DataFrame({"in_degree": [0, 1, 3], "out_degree": [2, 1, 4]})
    frame["total_degree"] = frame["in_degree"] + frame["out_degree"]
    assert frame["total_degree"].equals(frame["in_degree"] + frame["out_degree"])


def test_snapshot_normalized_pagerank_has_mean_one():
    graph = nx.DiGraph([(1, 2), (2, 3)])
    graph.add_node(4)
    raw = nx.pagerank(graph)
    relative = np.array([raw[node] * graph.number_of_nodes() for node in graph.nodes])
    assert np.isclose(relative.mean(), 1.0)
