import numpy as np
import pandas as pd

from src.modeling_utils import feature_sets, select_f1_threshold


def test_feature_sets_never_include_ids_time_or_corrected_exclusions():
    frame = pd.DataFrame(
        {
            "txId": [1],
            "time_step": [1],
            "feature_1": [0.1],
            "feature_2": [0.2],
            "in_degree": [1],
            "out_degree": [1],
            "pagerank_relative": [1.2],
            "pagerank_raw_snapshot": [0.01],
            "clustering": [0.0],
            "k_core": [1],
            "total_degree": [2],
            "weak_component_size": [10],
            "label": [0],
        }
    )
    sets = feature_sets(frame)
    for columns in sets.values():
        assert "txId" not in columns
        assert "time_step" not in columns
        assert "total_degree" not in columns
        assert "pagerank_raw_snapshot" not in columns
        assert "weak_component_size" not in columns


def test_threshold_is_selected_from_operating_scores():
    labels = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.7, 0.9])
    threshold, f1 = select_f1_threshold(labels, scores)
    assert 0.2 < threshold <= 0.7
    assert f1 == 1.0
