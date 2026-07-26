import pandas as pd

from src.data_loading import feature_names, normalize_classes


def test_feature_names_exclude_identifier_and_time_from_supplied_count():
    names = feature_names(167)
    assert names[0] == "txId"
    assert names[1] == "time_step"
    assert names[-1] == "feature_165"
    assert len(names[2:]) == 165


def test_label_mapping_preserves_unknown_as_missing_supervised_label():
    classes = pd.DataFrame(
        {"txId": [1, 2, 3], "class": ["1", "2", "unknown"]}
    )
    result = normalize_classes(classes)
    assert result.loc[result["txId"] == 1, "label"].iloc[0] == 1
    assert result.loc[result["txId"] == 2, "label"].iloc[0] == 0
    assert result.loc[result["txId"] == 3, "label"].isna().all()
