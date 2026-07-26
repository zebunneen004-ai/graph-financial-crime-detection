import numpy as np
import pandas as pd

from src.tuning import paired_bootstrap_delta, time_block_bootstrap_delta


def test_paired_bootstrap_preserves_alignment_and_is_reproducible():
    labels = np.array([0, 0, 0, 1, 1, 1])
    dataset = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    combined = np.array([0.1, 0.2, 0.25, 0.65, 0.75, 0.85])
    point_a, values_a = paired_bootstrap_delta(labels, dataset, combined, 100, 42)
    point_b, values_b = paired_bootstrap_delta(labels, dataset, combined, 100, 42)
    assert point_a == point_b
    assert np.array_equal(values_a, values_b)


def test_time_block_bootstrap_runs_on_aligned_prediction_table():
    frame = pd.DataFrame(
        {
            "time_step": [41, 41, 42, 42, 43, 43],
            "true_label": [0, 1, 0, 1, 0, 1],
            "dataset_score": [0.1, 0.8, 0.2, 0.7, 0.3, 0.6],
            "combined_score": [0.1, 0.85, 0.2, 0.75, 0.25, 0.65],
        }
    )
    values = time_block_bootstrap_delta(frame, 50, 42)
    assert len(values) > 0
