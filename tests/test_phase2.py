"""Check the fast vectorised features against slow, obviously correct versions.

    python -m pytest tests -q
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.fe import feols
from src.narrowness import same_tag_streak, window_features
from src.sessions import add_sessions

rng = np.random.default_rng(1)
T, V = 6, 40
M = np.zeros((V, T), dtype=np.float32)
for v in range(V):
    tl = rng.choice(T, rng.integers(1, 3), replace=False)
    M[v, tl] = 1 / len(tl)
user = np.repeat([0, 1, 2], [30, 5, 25])
video = rng.integers(0, V, len(user))


def slow_narrow(i, N):
    start = np.flatnonzero(user == user[i])[0]
    if i - start < N:
        return np.nan, np.nan
    S = M[video[i - N:i]].sum(0)
    p = S / S.sum()
    h = -sum(x * np.log(x) for x in p if x > 0)
    rep = p[M[video[i]] > 0].sum()
    return 1 - h / np.log(T), rep


def test_window_features():
    f = window_features(user, video, M, windows=(4,), repeat_window=4)
    for i in range(len(user)):
        nar, rep = slow_narrow(i, 4)
        assert np.isclose(f["narrow_4"][i], nar, equal_nan=True, atol=1e-5)
        assert np.isclose(f["repeat_share"][i], rep, equal_nan=True, atol=1e-5)


def test_streak():
    u = np.array([0, 0, 0, 0, 1, 1, 1])
    t = np.array([3, 3, 3, 5, 5, 5, 2])
    assert same_tag_streak(u, t).tolist() == [0, 1, 2, 0, 0, 1, 0]


def test_sessions():
    df = pd.DataFrame({"user_id": [0, 0, 0, 1, 1],
                       "timestamp": [0, 60, 4000, 10, 20]})
    s = add_sessions(df)
    assert s.session_id.tolist() == [0, 0, 1, 2, 2]
    assert s.pos.tolist() == [1, 2, 1, 1, 2]
    assert s.exit.tolist() == [0, 1, 1, 0, 1]
    assert s.last_sess.tolist() == [0, 0, 1, 1, 1]


def test_feols_matches_dummy_ols():
    n = 600
    d = pd.DataFrame({"user_id": rng.integers(0, 20, n), "video_id": rng.integers(0, 30, n),
                      "x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    d["y"] = 0.5 * d.x1 - 0.3 * d.x2 + d.user_id * 0.1 + (d.video_id % 5) * 0.2 + rng.normal(size=n)
    res = feols(d, "y", ["x1", "x2"])
    X = pd.get_dummies(d[["user_id", "video_id"]].astype(str), drop_first=True).astype(float)
    X = np.column_stack([np.ones(n), d[["x1", "x2"]], X])
    b = np.linalg.lstsq(X, d.y.to_numpy(), rcond=None)[0]
    assert np.allclose(res.coef.values, b[1:3], atol=1e-6)
