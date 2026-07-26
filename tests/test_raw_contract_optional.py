from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.integration
def test_raw_class_and_edge_contract_when_files_are_present():
    root = Path(__file__).resolve().parents[1]
    data_candidates = [root / "data" / "raw", root / "data"]
    data_dir = next((path for path in data_candidates if (path / "elliptic_txs_classes.csv").exists()), None)
    if data_dir is None:
        pytest.skip("Raw Elliptic files are not included in the code-only rebuild kit.")
    classes = pd.read_csv(data_dir / "elliptic_txs_classes.csv")
    edges = pd.read_csv(data_dir / "elliptic_txs_edgelist.csv")
    assert len(classes) == 203_769
    assert len(edges) == 234_355
    assert classes["txId"].is_unique
