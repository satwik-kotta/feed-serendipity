"""Build the per-interaction feature table for the fatigue analysis.

    python -m src.features            # writes data/features_big.parquet

Processes users in chunks so it fits in a few GB of RAM.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.load import DATA, load_matrix, load_tags
from src.narrowness import primary_tag, same_tag_streak, tag_matrix, window_features
from src.sessions import add_sessions

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "features_big.parquet"
WR_CLIP = 5.0


def build(chunk_users: int = 800, out_path: Path = OUT, verbose: bool = True) -> Path:
    big = load_matrix("big").sort_values(["user_id", "timestamp"], kind="stable")
    big = big.reset_index(drop=True)
    tags = load_tags()
    n_videos = int(max(big.video_id.max(), tags.video_id.max())) + 1
    n_tags = int(tags.tags.explode().max()) + 1
    M = tag_matrix(tags, n_videos, n_tags)
    ptag_v = primary_tag(tags, n_videos)

    users = big.user_id.unique()
    writer = None
    for k in range(0, len(users), chunk_users):
        uu = users[k:k + chunk_users]
        d = big[big.user_id.isin(uu)].reset_index(drop=True)
        d = add_sessions(d)
        u, v = d.user_id.to_numpy(), d.video_id.to_numpy()
        feats = window_features(u, v, M)
        for name, arr in feats.items():
            d[name] = arr
        d["ptag"] = ptag_v[v]
        d["streak"] = same_tag_streak(u, d["ptag"].to_numpy())
        d["wr"] = d.watch_ratio.clip(upper=WR_CLIP).astype("float32")
        d["full"] = (d.watch_ratio >= 1).astype("int8")
        d["log_pos"] = np.log(d["pos"]).astype("float32")
        hour = pd.to_datetime(d.timestamp, unit="s", utc=True).dt.tz_convert("Asia/Shanghai").dt.hour
        d["hour_block"] = (hour // 4).astype("int8")
        d["day"] = ((d.timestamp - d.timestamp.min()) // 86400).astype("int16")
        keep = ["user_id", "video_id", "timestamp", "session_id", "pos", "sess_len", "exit",
                "last_sess", "wr", "full", "log_pos", "hour_block", "day", "ptag", "streak",
                "repeat_share"] + [c for c in d.columns if c.startswith("narrow_")]
        tbl = pa.Table.from_pandas(d[keep], preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out_path, tbl.schema)
        writer.write_table(tbl)
        if verbose:
            print(f"  users {k + len(uu):>5,}/{len(users):,}  rows {len(d):>9,}", flush=True)
    writer.close()
    return out_path


if __name__ == "__main__":
    p = build()
    print("wrote", p)
