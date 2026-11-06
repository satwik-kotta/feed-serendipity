"""Loaders for KuaiRec. Every function returns a tidy pandas DataFrame."""
import ast
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("KUAIREC_DIR", ROOT / "data" / "KuaiRec" / "data"))

MATRIX_COLS = ["user_id", "video_id", "play_duration", "video_duration",
               "timestamp", "watch_ratio"]


def load_matrix(name: str = "big") -> pd.DataFrame:
    """Load big_matrix or small_matrix. Prefers parquet if download.py made it."""
    stem = f"{name}_matrix"
    pq = DATA / f"{stem}.parquet"
    if pq.exists():
        df = pd.read_parquet(pq, columns=MATRIX_COLS)
    else:
        df = pd.read_csv(DATA / f"{stem}.csv", usecols=MATRIX_COLS)
    df["user_id"] = df["user_id"].astype("int32")
    df["video_id"] = df["video_id"].astype("int32")
    df["watch_ratio"] = df["watch_ratio"].astype("float32")
    return df


def _parse_tags(s) -> list:
    if isinstance(s, (list, tuple)):
        return list(s)
    if pd.isna(s):
        return []
    return list(ast.literal_eval(str(s)))


def load_tags() -> pd.DataFrame:
    """One row per video with a python list of tag ids in `tags`."""
    df = pd.read_csv(DATA / "item_categories.csv")
    df["tags"] = df["feat"].map(_parse_tags)
    df["n_tags"] = df["tags"].str.len()
    return df[["video_id", "tags", "n_tags"]]


def load_tags_long() -> pd.DataFrame:
    """One row per (video_id, tag). A video with 3 tags appears 3 times."""
    return load_tags().explode("tags").dropna(subset=["tags"]).rename(
        columns={"tags": "tag"}).astype({"tag": "int16"})[["video_id", "tag"]]


def load_authors() -> pd.DataFrame:
    """video_id to author_id, taken from daily item features (most frequent value)."""
    df = pd.read_csv(DATA / "item_daily_features.csv",
                     usecols=["video_id", "author_id"])
    df = df.dropna()
    return (df.groupby("video_id")["author_id"]
              .agg(lambda s: s.mode().iat[0])
              .astype("int64").reset_index())


def load_user_features() -> pd.DataFrame:
    return pd.read_csv(DATA / "user_features.csv")
