import pandas as pd

from src.alert_budget import budget_table, evaluate_budget, select_budget


def sample_predictions():
    return pd.DataFrame(
        {
            "txId": range(1, 11),
            "raw_score": [0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20],
            "true_label": [1, 1, 0, 1, 0, 0, 0, 0, 0, 0],
        }
    )


def test_budget_counts_are_consistent():
    result = evaluate_budget(sample_predictions(), 0.20)
    assert result["alerts"] == 2
    assert result["true_positives"] == 2
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 1


def test_budget_selection_uses_f1_then_smaller_budget():
    table = budget_table(sample_predictions(), [0.1, 0.2, 0.4])
    selected = select_budget(table)
    assert selected in {0.1, 0.2, 0.4}
    best_f1 = table["f1"].max()
    expected = table.loc[table["f1"].eq(best_f1), "budget"].min()
    assert selected == expected
