"""Rolling Narrowness Index and related "content diet" features.

For every interaction i of a user, look at the N videos that user watched just
before it (not including i). Each video spreads a weight of 1 across its tags
(a video with 2 tags gives 0.5 to each). The window's tag shares p give

    narrowness = 1 - H(p) / log(T)

where H is Shannon entropy and T is the number of tags in the catalog.
0 means a perfectly even mix of every tag, 1 means one tag only.
"""
import numpy as np
import pandas as pd


def tag_matrix(tags: pd.DataFrame, n_videos: int, n_tags: int) -> np.ndarray:
    """Dense video x tag weight matrix. Row sums are 1 (or 0 for untagged videos)."""
    M = np.zeros((n_videos, n_tags), dtype=np.float32)
    for vid, tl in zip(tags["video_id"].to_numpy(), tags["tags"]):
        if len(tl):
            M[vid, tl] = 1.0 / len(tl)
    return M


def primary_tag(tags: pd.DataFrame, n_videos: int) -> np.ndarray:
    """First listed tag per video, -1 if none."""
    out = np.full(n_videos, -1, dtype=np.int16)
    for vid, tl in zip(tags["video_id"].to_numpy(), tags["tags"]):
        if len(tl):
            out[vid] = tl[0]
    return out


def _entropy_narrowness(S: np.ndarray, n_tags: int) -> np.ndarray:
    tot = S.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tot > 0, S / tot, 0)
        h = -(np.where(p > 0, p * np.log(p), 0)).sum(1)
    return (1 - h / np.log(n_tags)).astype(np.float32)


def window_features(user: np.ndarray, video: np.ndarray, M: np.ndarray,
                    windows=(10, 20, 50), repeat_window: int = 20) -> dict:
    """Rows must be sorted by user then time. Works on one chunk of whole users.

    Returns arrays aligned with the input rows
        narrow_{N}       narrowness of the previous N videos, NaN if fewer than N exist
        repeat_share     share of the previous `repeat_window` videos' tag weight that
                         lands on the current video's tags (how "more of the same" it is)
    """
    n, T = len(user), M.shape[1]
    W = M[video]                                       # n x T
    C0 = np.zeros((n + 1, T), dtype=np.float64)
    np.cumsum(W, axis=0, out=C0[1:])

    new_user = np.r_[True, user[1:] != user[:-1]]
    ustart = np.maximum.accumulate(np.where(new_user, np.arange(n), 0))
    idx = np.arange(n)
    hist = idx - ustart                               # videos seen before this one

    out = {}
    for N in windows:
        a = np.maximum(idx - N, ustart)
        S = C0[idx] - C0[a]
        nar = _entropy_narrowness(S, T)
        nar[hist < N] = np.nan
        out[f"narrow_{N}"] = nar
        if N == repeat_window:
            tot = S.sum(1)
            with np.errstate(divide="ignore", invalid="ignore"):
                rep = np.where(tot > 0, (S / tot[:, None] * (W > 0)).sum(1), np.nan)
            rep[hist < N] = np.nan
            out["repeat_share"] = rep.astype(np.float32)
    return out


def same_tag_streak(user: np.ndarray, ptag: np.ndarray) -> np.ndarray:
    """How many videos immediately before this one had the same primary tag."""
    n = len(user)
    same = np.r_[False, (user[1:] == user[:-1]) & (ptag[1:] == ptag[:-1]) & (ptag[1:] >= 0)]
    # streak resets wherever `same` is False
    run_id = np.cumsum(~same)
    first = np.r_[0, np.flatnonzero(np.diff(run_id)) + 1]
    start_of_run = np.repeat(first, np.diff(np.r_[first, n]))
    return (np.arange(n) - start_of_run).astype(np.int32)
