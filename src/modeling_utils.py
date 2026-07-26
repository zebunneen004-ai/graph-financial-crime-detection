from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.graph_features import PRIMARY_GRAPH_FEATURES


@dataclass(frozen=True)
class SplitFrames:
    train: pd.DataFrame
    calibration: pd.DataFrame
    operating: pd.DataFrame
    test: pd.DataFrame


def supplied_feature_names(frame: pd.DataFrame) -> list[str]:
    features = [column for column in frame.columns if column.startswith("feature_")]
    return sorted(features, key=lambda value: int(value.split("_")[1]))


def feature_sets(frame: pd.DataFrame) -> dict[str, list[str]]:
    dataset = supplied_feature_names(frame)
    combined = [*dataset, *PRIMARY_GRAPH_FEATURES]
    forbidden = {"txId", "time_step", "total_degree", "pagerank_raw_snapshot", "weak_component_size"}
    for name, columns in {"dataset_only": dataset, "graph_only": PRIMARY_GRAPH_FEATURES, "combined": combined}.items():
        overlap = forbidden.intersection(columns)
        if overlap:
            raise AssertionError(f"Forbidden predictors in {name}: {sorted(overlap)}")
    return {
        "dataset_only": dataset,
        "graph_only": list(PRIMARY_GRAPH_FEATURES),
        "combined": combined,
    }


def chronological_split(frame: pd.DataFrame, config: dict) -> SplitFrames:
    split = config["splits"]
    train = frame.loc[frame["time_step"].between(split["train_start"], split["train_end"])].copy()
    calibration = frame.loc[
        frame["time_step"].between(split["calibration_start"], split["calibration_end"])
    ].copy()
    operating = frame.loc[
        frame["time_step"].between(split["operating_start"], split["operating_end"])
    ].copy()
    test = frame.loc[frame["time_step"].between(split["test_start"], split["test_end"])].copy()

    frames = {
        "train": train,
        "calibration": calibration,
        "operating": operating,
        "test": test,
    }
    for name, subset in frames.items():
        if subset.empty:
            raise AssertionError(f"Chronological split {name} is empty.")

    id_sets = {name: set(subset["txId"]) for name, subset in frames.items()}
    names = list(id_sets)
    for index, left in enumerate(names):
        for right in names[index + 1:]:
            if id_sets[left] & id_sets[right]:
                raise AssertionError(f"Chronological splits {left} and {right} overlap.")

    if train["time_step"].max() >= calibration["time_step"].min():
        raise AssertionError("Calibration does not occur after training.")
    if calibration["time_step"].max() >= operating["time_step"].min():
        raise AssertionError("Operating selection does not occur after calibration.")
    if operating["time_step"].max() >= test["time_step"].min():
        raise AssertionError("Final test does not occur after operating selection.")
    return SplitFrames(train=train, calibration=calibration, operating=operating, test=test)


def split_summary(splits: SplitFrames) -> pd.DataFrame:
    rows = []
    for name, frame in (
        ("train", splits.train),
        ("calibration", splits.calibration),
        ("operating", splits.operating),
        ("test", splits.test),
    ):
        rows.append(
            {
                "split": name,
                "rows": len(frame),
                "licit": int((frame["label"] == 0).sum()),
                "illicit": int((frame["label"] == 1).sum()),
                "illicit_prevalence": float(frame["label"].mean()),
                "minimum_time_step": int(frame["time_step"].min()),
                "maximum_time_step": int(frame["time_step"].max()),
            }
        )
    return pd.DataFrame(rows)


def temporal_cv_indices(train_frame: pd.DataFrame, config: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for specification in config["cv_folds"]:
        train_start, train_end = specification["train"]
        validation_start, validation_end = specification["validation"]
        train_indices = np.flatnonzero(train_frame["time_step"].between(train_start, train_end).to_numpy())
        validation_indices = np.flatnonzero(
            train_frame["time_step"].between(validation_start, validation_end).to_numpy()
        )
        if len(train_indices) == 0 or len(validation_indices) == 0:
            raise AssertionError(f"Empty temporal fold: {specification}")
        if train_frame.iloc[train_indices]["time_step"].max() >= train_frame.iloc[validation_indices]["time_step"].min():
            raise AssertionError(f"Temporal ordering failure: {specification}")
        if train_frame.iloc[validation_indices]["label"].nunique() < 2:
            raise AssertionError(f"Validation fold has only one class: {specification}")
        folds.append((train_indices, validation_indices))
    return folds


def class_ratio(y: pd.Series | np.ndarray) -> float:
    y_array = np.asarray(y)
    positives = int((y_array == 1).sum())
    negatives = int((y_array == 0).sum())
    if positives == 0:
        raise ValueError("Training target contains no illicit-labeled observations.")
    return negatives / positives


def make_default_model(name: str, config: dict, y_train: pd.Series):
    seed = int(config["project"]["seed"])
    threads = int(config["models"]["threads"])
    if name == "logistic_regression":
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=seed,
                    ),
                ),
            ]
        )
    if name == "random_forest":
        params = config["models"]["random_forest"]
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(params["n_estimators"]),
                        min_samples_leaf=int(params["min_samples_leaf"]),
                        max_features=params["max_features"],
                        class_weight="balanced_subsample",
                        random_state=seed,
                        n_jobs=threads,
                    ),
                ),
            ]
        )
    if name == "xgboost":
        params = dict(config["models"]["default_xgboost"])
        return XGBClassifier(
            **params,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=threads,
            scale_pos_weight=class_ratio(y_train),
            missing=np.nan,
        )
    raise ValueError(f"Unknown model: {name}")


def make_xgboost(params: dict, config: dict, y_train: pd.Series) -> XGBClassifier:
    return XGBClassifier(
        **params,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=int(config["project"]["seed"]),
        n_jobs=int(config["models"]["threads"]),
        scale_pos_weight=class_ratio(y_train),
        missing=np.nan,
    )


def select_f1_threshold(y_true: Iterable[int], scores: Iterable[float]) -> tuple[float, float]:
    y_array = np.asarray(y_true)
    score_array = np.asarray(scores)
    precision, recall, thresholds = precision_recall_curve(y_array, score_array)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-15)
    best_f1 = np.nanmax(f1_values)
    candidate_indices = np.flatnonzero(np.isclose(f1_values, best_f1))
    best_index = candidate_indices[-1]
    return float(thresholds[best_index]), float(f1_values[best_index])


def metric_row(y_true: Iterable[int], scores: Iterable[float], threshold: float) -> dict[str, float]:
    y_array = np.asarray(y_true)
    scores_array = np.asarray(scores)
    predictions = (scores_array >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_array, scores_array)),
        "roc_auc": float(roc_auc_score(y_array, scores_array)),
        "precision": float(precision_score(y_array, predictions, zero_division=0)),
        "recall": float(recall_score(y_array, predictions, zero_division=0)),
        "f1": float(f1_score(y_array, predictions, zero_division=0)),
        "threshold": float(threshold),
        "prevalence": float(y_array.mean()),
        "rows": int(len(y_array)),
        "illicit": int(y_array.sum()),
        "licit": int((y_array == 0).sum()),
    }


def prediction_frame(
    base: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    *,
    model_family: str,
    feature_set: str,
    split: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "txId": base["txId"].to_numpy(),
            "time_step": base["time_step"].to_numpy(),
            "true_label": base["label"].to_numpy(),
            "raw_score": scores,
            "hard_prediction": (scores >= threshold).astype("int8"),
            "threshold": threshold,
            "model_family": model_family,
            "feature_set": feature_set,
            "split": split,
        }
    )
