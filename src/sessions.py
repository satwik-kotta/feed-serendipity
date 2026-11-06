"""Split each user's viewing log into sessions."""
import numpy as np
import pandas as pd

GAP_SECONDS = 30 * 60


def add_sessions(df: pd.DataFrame, gap: int = GAP_SECONDS) -> pd.DataFrame:
    """df must be sorted by user_id, timestamp.

    Adds
        session_id   global integer id
        pos          1-based position of the video inside its session
        sess_len     number of videos in the session
        exit         1 if this is the last video of the session
        last_sess    1 if the session is the user's final one (its exit is censored)
    """
    u = df["user_id"].to_numpy()
    t = df["timestamp"].to_numpy()
    new_user = np.r_[True, u[1:] != u[:-1]]
    gap_break = np.r_[True, np.diff(t) > gap]
    new_sess = new_user | gap_break
    sid = np.cumsum(new_sess) - 1

    out = df.copy()
    out["session_id"] = sid.astype("int32")
    out["pos"] = (out.groupby("session_id").cumcount() + 1).astype("int32")
    sess_len = np.bincount(sid)
    out["sess_len"] = sess_len[sid].astype("int32")
    out["exit"] = (out["pos"] == out["sess_len"]).astype("int8")

    last_sid = out.groupby("user_id")["session_id"].transform("max").to_numpy()
    out["last_sess"] = (sid == last_sid).astype("int8")
    return out
