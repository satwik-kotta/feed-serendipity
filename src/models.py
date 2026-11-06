"""Baseline recommender: ALS + LightGBM reranker.

Two targets are trained side by side
    raw      watch ratio. Mostly measures how short a video is (item mean vs log duration: r = -0.91),
             so a model trained on it shows everyone the same short clips.
    debiased how far a play's watch ratio sits above the typical value for a video of that length
             (a smooth curve fitted on the big matrix), as a percentile from 0 to 1. The same idea as
             Kuaishou's duration-debiased watch-time work (D2Q, KDD 2022), without D2Q's bins, which
             leave a saw-tooth length bias at every bin edge.
             This is the main baseline.

Training never sees the small matrix. The setup mirrors production:

    history  = big matrix before the time cut   -> features + ALS
    labels   = big matrix after the time cut    -> LightGBM target (watch ratio)

For the final evaluation, history = the whole big matrix, and every
(small-matrix user, small-matrix video) pair is scored. The small matrix holds
the true watch ratio for almost every one of those pairs.

Run in steps (each fits in a few minutes and ~3 GB RAM):

    python -m src.models als      # fit ALS on history and on the full big matrix
    python -m src.models train    # build features on the label period, fit LightGBM
    python -m src.models score    # score all small-matrix pairs with every baseline
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.load import load_authors, load_matrix, load_tags
from src.narrowness import primary_tag

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data" / "models"
WR_CLIP = 5.0
CUT_Q = 0.8                     # last 20% of time in the big matrix is the label period
ALS_PARAMS = dict(factors=64, regularization=0.05, iterations=15, random_state=0)
SMOOTH = 5                      # pseudo-count for user x tag and author means

FEATURES = ["als", "item_mean_wr", "item_log_cnt", "item_full_rate", "item_replay_rate",
            "user_mean_wr", "user_log_cnt", "user_full_rate",
            "ut_mean_wr", "ut_share", "author_mean_wr", "author_log_cnt",
            "log_duration", "n_tags", "ptag"]


# ---------------------------------------------------------------- data

def load_big():
    df = load_matrix("big").drop(columns=["play_duration"])
    df["video_duration"] = df.video_duration.astype("float32")
    df["wr"] = df.watch_ratio.clip(upper=WR_CLIP).astype("float32")
    return df


def duration_curve(big, n_points: int = 100):
    """Typical log(1 + watch ratio) as a smooth function of log video length.

    Means of log1p(watch ratio) in 100 equal-count bins of plays, used as knots for
    linear interpolation. Returns a (2, n_points) array: x = log length, y = expected value.
    """
    x = np.log1p(big.video_duration.to_numpy())
    y = np.log1p(big.watch_ratio.to_numpy())
    q = np.quantile(x, np.linspace(0, 1, n_points + 1))
    b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, n_points - 1)
    xs = np.bincount(b, weights=x, minlength=n_points) / np.maximum(np.bincount(b, minlength=n_points), 1)
    ys = np.bincount(b, weights=y, minlength=n_points) / np.maximum(np.bincount(b, minlength=n_points), 1)
    return np.vstack([xs, ys])


def _curve_at(x, curve):
    """Piecewise-linear curve, extended linearly past both ends (plain interpolation would go flat
    there and make the very shortest and longest videos look better or worse than they are)."""
    xs, ys = curve
    y = np.interp(x, xs, ys)
    lo_slope = (ys[1] - ys[0]) / (xs[1] - xs[0])
    hi_slope = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
    y = np.where(x < xs[0], ys[0] + lo_slope * (x - xs[0]), y)
    return np.where(x > xs[-1], ys[-1] + hi_slope * (x - xs[-1]), y)


def _residual(watch_ratio, log_duration, curve):
    return np.log1p(watch_ratio) - _curve_at(log_duration, curve)


def add_debiased(df, curve):
    """q = percentile (0-1) of how far a play's watch ratio sits above the typical value for its length."""
    r = _residual(df.watch_ratio.to_numpy(), np.log1p(df.video_duration.to_numpy()), curve)
    df["q"] = pd.Series(r).rank(pct=True).to_numpy(np.float32)
    return df


def debias_matrix(truth, log_duration, curve):
    """Same transform for a users x items truth matrix (log_duration is log1p of length in ms)."""
    r = _residual(truth, log_duration[None, :], curve)
    Q = np.full_like(truth, np.nan)
    ok = ~np.isnan(r)
    Q[ok] = pd.Series(r[ok]).rank(pct=True).to_numpy()
    return Q


def time_cut(df):
    return float(df.timestamp.quantile(CUT_Q))


def item_meta(big):
    tags = load_tags()
    n_items = int(max(big.video_id.max(), tags.video_id.max())) + 1
    meta = pd.DataFrame({"video_id": np.arange(n_items)})
    meta["ptag"] = primary_tag(tags, n_items)
    meta = meta.merge(tags[["video_id", "n_tags"]], on="video_id", how="left")
    dur = big.groupby("video_id").video_duration.median()
    meta["log_duration"] = np.log1p(meta.video_id.map(dur)).astype("float32")
    meta = meta.merge(load_authors(), on="video_id", how="left")
    meta["author_id"] = meta.author_id.fillna(-1).astype("int64")
    meta["n_tags"] = meta.n_tags.fillna(0).astype("int8")
    return meta


# ---------------------------------------------------------------- ALS

def fit_als(hist, n_users, n_items):
    from implicit.als import AlternatingLeastSquares
    w = hist.wr.to_numpy(np.float32)
    X = sp.csr_matrix((w, (hist.user_id.to_numpy(), hist.video_id.to_numpy())),
                      shape=(n_users, n_items))
    X.eliminate_zeros()
    m = AlternatingLeastSquares(**ALS_PARAMS)
    m.fit(X, show_progress=False)
    uf, itf = m.user_factors, m.item_factors
    if hasattr(uf, "to_numpy"):
        uf, itf = uf.to_numpy(), itf.to_numpy()
    return np.asarray(uf, np.float32), np.asarray(itf, np.float32)


# ---------------------------------------------------------------- features

def history_stats(hist, meta, label: str = "wr"):
    """Aggregates that only use the history period. `label` is the outcome being summarised
    (raw clipped watch ratio or the debiased percentile); feature names stay the same."""
    h = hist[["user_id", "video_id", label, "watch_ratio"]].rename(columns={label: "wr"}).merge(
        meta[["video_id", "ptag", "author_id"]], on="video_id", how="left")
    h["full"] = (h.watch_ratio >= 1).astype("float32")
    h["replay"] = (h.watch_ratio >= 2).astype("float32")
    item = h.groupby("video_id").agg(item_mean_wr=("wr", "mean"), item_cnt=("wr", "size"),
                                     item_full_rate=("full", "mean"), item_replay_rate=("replay", "mean"))
    item["item_log_cnt"] = np.log1p(item.pop("item_cnt"))
    user = h.groupby("user_id").agg(user_mean_wr=("wr", "mean"), user_cnt=("wr", "size"),
                                    user_full_rate=("full", "mean"))
    ut = h.groupby(["user_id", "ptag"]).agg(ut_sum=("wr", "sum"), ut_cnt=("wr", "size")).reset_index()
    ut = ut.merge(user[["user_mean_wr", "user_cnt"]], left_on="user_id", right_index=True)
    ut["ut_mean_wr"] = (ut.ut_sum + SMOOTH * ut.user_mean_wr) / (ut.ut_cnt + SMOOTH)
    ut["ut_share"] = ut.ut_cnt / ut.user_cnt
    user["user_log_cnt"] = np.log1p(user.pop("user_cnt"))
    glob = h.wr.mean()
    au = h.groupby("author_id").agg(a_sum=("wr", "sum"), a_cnt=("wr", "size"))
    au["author_mean_wr"] = (au.a_sum + SMOOTH * glob) / (au.a_cnt + SMOOTH)
    au["author_log_cnt"] = np.log1p(au.a_cnt)
    return {"item": item, "user": user, "ut": ut[["user_id", "ptag", "ut_mean_wr", "ut_share"]],
            "author": au[["author_mean_wr", "author_log_cnt"]], "glob": glob}


def build_features(pairs, stats, meta, uf, itf):
    """pairs: DataFrame with user_id, video_id. Returns pairs + FEATURES columns."""
    u = pairs.user_id.to_numpy()
    v = pairs.video_id.to_numpy()
    X = pd.DataFrame({"user_id": u, "video_id": v})
    X["als"] = np.einsum("ij,ij->i", uf[u], itf[v]).astype("float32")
    X = X.merge(meta[["video_id", "ptag", "n_tags", "log_duration", "author_id"]], on="video_id", how="left")
    X = X.merge(stats["item"], left_on="video_id", right_index=True, how="left")
    X = X.merge(stats["user"], left_on="user_id", right_index=True, how="left")
    X = X.merge(stats["ut"], on=["user_id", "ptag"], how="left")
    X["ut_mean_wr"] = X.ut_mean_wr.fillna(X.user_mean_wr)
    X["ut_share"] = X.ut_share.fillna(0)
    X = X.merge(stats["author"], left_on="author_id", right_index=True, how="left")
    X["author_mean_wr"] = X.author_mean_wr.fillna(stats["glob"])
    X["author_log_cnt"] = X.author_log_cnt.fillna(0)
    for c in FEATURES:
        if c != "ptag":
            X[c] = X[c].astype("float32")
    X["ptag"] = X.ptag.astype("int16")
    return X


# ---------------------------------------------------------------- steps

def step_als():
    ART.mkdir(parents=True, exist_ok=True)
    big = load_big()
    n_users, n_items = int(big.user_id.max()) + 1, int(big.video_id.max()) + 1
    cut = time_cut(big)
    for name, hist in [("hist", big[big.timestamp < cut]), ("full", big)]:
        uf, itf = fit_als(hist, n_users, n_items)
        np.savez(ART / f"als_{name}.npz", uf=uf, itf=itf)
        print(f"ALS {name}: {len(hist):,} interactions, factors {uf.shape} {itf.shape}", flush=True)


LGB_PARAMS = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=4)
TARGETS = {"raw": "wr", "debiased": "q"}


def step_train(max_rows: int = 1_200_000, seed: int = 0):
    import gc
    import lightgbm as lgb
    big = load_big()
    curve = duration_curve(big)
    np.save(ART / "duration_curve.npy", curve)
    big = add_debiased(big, curve)
    cut = time_cut(big)
    meta = item_meta(big)
    hist, lab = big[big.timestamp < cut], big[big.timestamp >= cut]
    # Keep only label rows whose user and item exist in history, like a real cold-start filter
    lab = lab[lab.user_id.isin(hist.user_id.unique()) & lab.video_id.isin(hist.video_id.unique())]
    lab = lab.sample(min(max_rows, len(lab)), random_state=seed)
    a = np.load(ART / "als_hist.npz")
    stats = {t: history_stats(hist, meta, col) for t, col in TARGETS.items()}
    del big, hist; gc.collect()

    users = lab.user_id.unique()
    rng = np.random.default_rng(seed)
    is_val = lab.user_id.isin(set(rng.choice(users, len(users) // 10, replace=False))).to_numpy()
    log = []
    for t, col in TARGETS.items():
        X = build_features(lab[["user_id", "video_id"]], stats[t], meta, a["uf"], a["itf"])
        y = lab[col].to_numpy()
        dtr = lgb.Dataset(X.loc[~is_val, FEATURES], y[~is_val], categorical_feature=["ptag"])
        dva = lgb.Dataset(X.loc[is_val, FEATURES], y[is_val], categorical_feature=["ptag"], reference=dtr)
        model = lgb.train({**LGB_PARAMS, "seed": seed}, dtr, num_boost_round=800, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
        model.save_model(str(ART / f"lgbm_{t}.txt"), num_iteration=model.best_iteration)
        pred = model.predict(X.loc[is_val, FEATURES], num_iteration=model.best_iteration)
        rmse = float(np.sqrt(np.mean((pred - y[is_val]) ** 2)))
        base = float(np.sqrt(np.mean((y[~is_val].mean() - y[is_val]) ** 2)))
        pd.Series(model.feature_importance("gain"), index=FEATURES).sort_values(ascending=False) \
          .to_csv(ART / f"lgbm_{t}_importance.csv")
        log.append(f"{t:9s} target: best iteration {model.best_iteration}, validation RMSE {rmse:.4f} "
                   f"vs {base:.4f} for predicting the mean ({1 - rmse / base:.1%} lower)")
        del X; gc.collect()
    msg = (f"Trained on {int((~is_val).sum()):,} label-period views, validated on {int(is_val.sum()):,} "
           f"views from held-out users\n" + "\n".join(log))
    (ART / "train_log.txt").write_text(msg)
    print(msg, flush=True)


def step_score(chunk_users: int = 150):
    import gc
    import lightgbm as lgb
    big = load_big()
    curve = np.load(ART / "duration_curve.npy")
    big = add_debiased(big, curve)
    meta = item_meta(big)
    stats = {t: history_stats(big, meta, col) for t, col in TARGETS.items()}
    item_cnt = big.video_id.value_counts()
    del big; gc.collect()

    small = load_matrix("small")[["user_id", "video_id", "watch_ratio", "video_duration"]]
    # The evaluation measure is made length-neutral on the evaluation data itself. In the small
    # matrix every user saw almost everything, so watch ratio falls off with length more steeply
    # than in the big matrix, and a curve fitted on the big matrix would still favour short videos.
    # This curve is only used to score policies, never to train anything.
    curve_eval = duration_curve(small)
    np.save(ART / "duration_curve_eval.npy", curve_eval)
    users = np.sort(small.user_id.unique())
    items = np.sort(small.video_id.unique())
    U, I = len(users), len(items)
    uix = pd.Series(np.arange(U), index=users)
    iix = pd.Series(np.arange(I), index=items)
    truth = np.full((U, I), np.nan, dtype=np.float32)
    truth[uix[small.user_id].to_numpy(), iix[small.video_id].to_numpy()] = small.watch_ratio.to_numpy()
    del small; gc.collect()
    m = meta.set_index("video_id").loc[items]
    truth_q = debias_matrix(truth, m.log_duration.to_numpy(), curve_eval)

    a = np.load(ART / "als_full.npz")
    uf, itf = a["uf"], a["itf"]
    out = {"als": (uf[users] @ itf[items].T).astype(np.float32)}
    for t in TARGETS:
        model = lgb.Booster(model_file=str(ART / f"lgbm_{t}.txt"))
        sc = np.empty((U, I), dtype=np.float32)
        for k in range(0, U, chunk_users):
            uu = users[k:k + chunk_users]
            pairs = pd.DataFrame({"user_id": np.repeat(uu, I), "video_id": np.tile(items, len(uu))})
            X = build_features(pairs, stats[t], meta, uf, itf)
            sc[k:k + len(uu)] = model.predict(X[FEATURES], num_threads=4).reshape(len(uu), I)
            del X, pairs
        out[f"lgbm_{t}"] = sc
        qual = stats[t]["item"].item_mean_wr.reindex(items).fillna(stats[t]["glob"]).to_numpy(np.float32)
        out[f"qual_{t}"] = np.tile(qual, (U, 1))
    pop = np.log1p(item_cnt.reindex(items).fillna(0).to_numpy(np.float32))
    np.savez_compressed(ART / "small_scores.npz", users=users, items=items, truth=truth, truth_q=truth_q,
                        pop=np.tile(pop, (U, 1)), ptag=m.ptag.to_numpy(), author=m.author_id.to_numpy(),
                        log_duration=m.log_duration.to_numpy(),
                        item_views_big=item_cnt.reindex(items).fillna(0).to_numpy(), **out)
    print(f"Scored {U:,} users x {I:,} items", flush=True)


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else ""
    {"als": step_als, "train": step_train, "score": step_score}[step]()
