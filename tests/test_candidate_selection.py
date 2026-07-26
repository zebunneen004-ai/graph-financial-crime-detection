from src.tuning import choose_default_or_tuned


def summary(name: str, score: float):
    return {
        "candidate_name": name,
        "feature_set": "dataset_only",
        "mean_cv_pr_auc": score,
        "std_cv_pr_auc": 0.01,
        "trial": 0,
        "parameters": {},
    }


def test_default_retained_when_tuning_gain_is_too_small():
    chosen, reason = choose_default_or_tuned(
        summary("default_dataset_only", 0.60),
        summary("tuned_dataset_only", 0.601),
        minimum_gain=0.002,
    )
    assert chosen["candidate_name"] == "default_dataset_only"
    assert "parsimony" in reason.lower()


def test_tuned_selected_when_gain_meets_margin():
    chosen, _ = choose_default_or_tuned(
        summary("default_dataset_only", 0.60),
        summary("tuned_dataset_only", 0.603),
        minimum_gain=0.002,
    )
    assert chosen["candidate_name"] == "tuned_dataset_only"
