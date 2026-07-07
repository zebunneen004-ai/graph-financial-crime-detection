import pandas as pd
import numpy as np
from pathlib import Path


def load_elliptic_data(data_dir="data"):
    data_path = Path(data_dir)
    features = pd.read_csv(data_path / "elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(data_path / "elliptic_txs_classes.csv")
    edges = pd.read_csv(data_path / "elliptic_txs_edgelist.csv")
    return features, classes, edges


def clean_and_merge(features, classes):
    feature_cols = ["txId", "time_step"] + [f"feature_{i}" for i in range(1, features.shape[1] - 1)]
    features.columns = feature_cols
    df = features.merge(classes, on="txId", how="left")
    df = df.copy()
    df["label"] = df["class"].map({"1": 1, "2": 0, "unknown": "unknown"})
    return df

def split_labeled_unlabeled(df):
    df_all = df.copy()
    df_labeled = df[df["label"].isin([0, 1])].copy()
    df_labeled["label"] = df_labeled["label"].astype(int)
    return df_all, df_labeled


def save_processed(df_all, df_labeled, output_dir="data/processed"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_path / "clean_nodes_all.csv", index=False)
    df_labeled.to_csv(out_path / "clean_nodes_labeled.csv", index=False)
    print(f"Saved processed data to {output_dir}")
