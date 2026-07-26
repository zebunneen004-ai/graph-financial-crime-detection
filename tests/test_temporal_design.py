import pandas as pd

from src.modeling_utils import chronological_split, temporal_cv_indices


CONFIG = {
    "splits": {
        "train_start": 1,
        "train_end": 30,
        "calibration_start": 31,
        "calibration_end": 35,
        "operating_start": 36,
        "operating_end": 40,
        "test_start": 41,
        "test_end": 49,
    },
    "cv_folds": [
        {"train": [1, 10], "validation": [11, 15]},
        {"train": [1, 15], "validation": [16, 20]},
        {"train": [1, 20], "validation": [21, 25]},
        {"train": [1, 25], "validation": [26, 30]},
    ],
}


def synthetic_frame():
    rows = []
    tx_id = 1
    for time_step in range(1, 50):
        rows.append({"txId": tx_id, "time_step": time_step, "label": 0})
        tx_id += 1
        rows.append({"txId": tx_id, "time_step": time_step, "label": 1})
        tx_id += 1
    return pd.DataFrame(rows)


def test_chronological_split_is_non_overlapping_and_ordered():
    split = chronological_split(synthetic_frame(), CONFIG)
    assert split.train["time_step"].min() == 1
    assert split.train["time_step"].max() == 30
    assert split.calibration["time_step"].min() == 31
    assert split.calibration["time_step"].max() == 35
    assert split.operating["time_step"].min() == 36
    assert split.operating["time_step"].max() == 40
    assert split.test["time_step"].min() == 41
    assert split.test["time_step"].max() == 49
    assert set(split.train["txId"]).isdisjoint(split.test["txId"])
    assert set(split.calibration["txId"]).isdisjoint(split.operating["txId"])


def test_expanding_window_folds_never_train_on_future():
    train = synthetic_frame().query("time_step <= 30").reset_index(drop=True)
    folds = temporal_cv_indices(train, CONFIG)
    assert len(folds) == 4
    for train_indices, validation_indices in folds:
        assert (
            train.iloc[train_indices]["time_step"].max()
            < train.iloc[validation_indices]["time_step"].min()
        )
