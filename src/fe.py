"""Linear regression with user and video fixed effects, clustered by user.

The two sets of fixed effects are absorbed with alternating projections
(subtract user means, then video means, repeat until stable). This gives the
same slope estimates as putting a dummy for every user and video into OLS,
without building a 7K + 10K column matrix.
"""
import numpy as np
import pandas as pd
from scipy import stats


def _demean(A: np.ndarray, groups: list, tol: float = 1e-7, max_iter: int = 200) -> np.ndarray:
    """Works in place on a float64 array to keep memory down."""
    A = A.astype(np.float64, copy=False)
    counts = [np.bincount(g).astype(np.float64) for g in groups]
    counts = [np.where(c == 0, 1, c) for c in counts]
    for _ in range(max_iter):
        delta = 0.0
        for g, c in zip(groups, counts):
            for j in range(A.shape[1]):
                m = np.bincount(g, weights=A[:, j]) / c
                A[:, j] -= m[g]
                delta = max(delta, np.abs(m).max())
        if delta < tol:
            break
    return A


def feols(df: pd.DataFrame, y: str, x: list, fe=("user_id", "video_id"),
          cluster: str = "user_id") -> pd.DataFrame:
    """Return a coefficient table with cluster-robust standard errors."""
    ids = list(dict.fromkeys(list(fe) + [cluster]))
    ok = df[[y] + x].notna().all(axis=1).to_numpy()
    groups = [pd.factorize(df[f].to_numpy()[ok])[0] for f in fe]
    cl = pd.factorize(df[cluster].to_numpy()[ok])[0]
    y_mean = float(df[y].to_numpy()[ok].mean())
    A = np.empty((ok.sum(), 1 + len(x)), dtype=np.float64)
    for j, c in enumerate([y] + x):
        A[:, j] = df[c].to_numpy()[ok]
    A = _demean(A, groups)
    yd, Xd = A[:, 0], A[:, 1:]

    # Drop regressors with no variation left after removing the fixed effects
    keep = Xd.std(0) > 1e-10
    if not keep.all():
        print("feols: dropped (no within variation):", [c for c, k in zip(x, keep) if not k])
    x_all, x = x, [c for c, k in zip(x, keep) if k]
    Xd = Xd[:, keep]

    XtX = Xd.T @ Xd
    beta = np.linalg.solve(XtX, Xd.T @ yd)
    e = yd - Xd @ beta

    del A
    G = cl.max() + 1
    S = np.zeros((G, Xd.shape[1]))
    np.add.at(S, cl, Xd * e[:, None])
    n, k = Xd.shape
    k_fe = sum(len(np.unique(g)) for g in groups)
    adj = G / (G - 1) * (n - 1) / max(n - k - k_fe, 1)
    bread = np.linalg.inv(XtX)
    V = adj * bread @ (S.T @ S) @ bread
    se = np.sqrt(np.diag(V))
    t = beta / se
    p = 2 * stats.t.sf(np.abs(t), G - 1)
    crit = stats.t.ppf(0.975, G - 1)
    out = pd.DataFrame({"coef": beta, "se": se, "t": t, "p": p,
                        "lo": beta - crit * se, "hi": beta + crit * se}, index=x).reindex(x_all)
    out.attrs.update(n=n, clusters=G, y_mean=y_mean)
    return out
