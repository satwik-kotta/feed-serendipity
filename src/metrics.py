"""Offline metrics on the fully observed small matrix.

Every policy produces a top-K list per user (an int array, users x K, of column
indices into the small-matrix item list). Because the small matrix is ~fully
observed, the true watch ratio of every recommended item is known, so these
are real outcomes, not proxies.

Relevance is measured two ways. `rel` and `ndcg` use the duration-debiased
percentile (how well a video was watched compared with videos of similar
length). `mean_wr`, `full_rate` and `replay_rate` use raw watch ratio, which
mostly rewards short videos.
"""
import numpy as np
import pandas as pd

WR_CLIP = 5.0


def topk_from_scores(scores: np.ndarray, truth: np.ndarray, k: int = 10) -> np.ndarray:
    """Top-k columns per row. Items with no ground truth (blocked by the user) are never recommended."""
    s = np.where(np.isnan(truth), -np.inf, scores)
    part = np.argpartition(-s, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(s, part, 1), axis=1)
    return np.take_along_axis(part, order, 1)


def gini(x) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def per_user(top: np.ndarray, truth_q: np.ndarray, truth_raw: np.ndarray, ptag: np.ndarray,
             item_pop: np.ndarray, secs: np.ndarray = None) -> pd.DataFrame:
    """One row per user with list-level metrics.

    truth_q    duration-debiased percentile (0-1). The main relevance measure; nDCG uses it.
    truth_raw  raw watch ratio, reported alongside.
    secs       video length in seconds; adds watch_secs = expected seconds watched per recommended video.
    """
    U, k = top.shape
    q = np.take_along_axis(truth_q, top, 1)
    wr = np.take_along_axis(truth_raw, top, 1)
    disc = 1 / np.log2(np.arange(2, k + 2))
    dcg = (np.nan_to_num(q) * disc).sum(1)
    ideal = -np.sort(-np.nan_to_num(truth_q), axis=1)[:, :k]
    idcg = (ideal * disc).sum(1)

    tags = ptag[top]
    n_distinct = np.array([len(np.unique(r)) for r in tags])
    same = (tags[:, :, None] == tags[:, None, :]).sum((1, 2)) - k
    ild = 1 - same / (k * (k - 1))

    p = item_pop / item_pop.sum()
    novelty = (-np.log2(np.maximum(p[top], 1e-12))).mean(1)
    out = pd.DataFrame({
        "ndcg": dcg / idcg,
        "rel": np.nanmean(q, 1),
        "top20_rate": (q >= 0.8).mean(1),
        "mean_wr": np.clip(wr, 0, WR_CLIP).mean(1),
        "full_rate": (wr >= 1).mean(1),
        "replay_rate": (wr >= 2).mean(1),
        "distinct_tags": n_distinct,
        "ild": ild,
        "novelty": novelty,
    })
    if secs is not None:
        out.insert(3, "watch_secs", (np.clip(wr, 0, WR_CLIP) * secs[top]).mean(1))
    return out


def author_tiers(item_views: np.ndarray, author: np.ndarray, head: float = 0.01, mid: float = 0.20):
    """Tier each author by total views in the big matrix: head = top 1%, mid = next 19%, tail = rest."""
    a = pd.Series(item_views).groupby(author).sum().sort_values(ascending=False)
    r = np.arange(len(a)) / len(a)
    tier = pd.Series(np.where(r < head, "head", np.where(r < mid, "mid", "tail")), index=a.index)
    return tier


def system_level(top: np.ndarray, author: np.ndarray, ptag: np.ndarray, tiers: pd.Series,
                 n_items: int) -> dict:
    """Metrics over everyone's lists together (the creator / catalog side)."""
    rec_items = top.ravel()
    rec_auth = author[rec_items]
    # Gini over every author who has at least one item in the catalog, so starved authors count as 0
    all_auth = np.unique(author)
    exp = pd.Series(rec_auth).value_counts().reindex(all_auth, fill_value=0)
    t = tiers.reindex(rec_auth).value_counts(normalize=True)
    return {
        "creator_gini": gini(exp.values),
        "authors_reached": int((exp > 0).sum()),
        "head_share": float(t.get("head", 0)),
        "mid_share": float(t.get("mid", 0)),
        "tail_share": float(t.get("tail", 0)),
        "catalog_coverage": len(np.unique(rec_items)) / n_items,
        "tags_reached": len(np.unique(ptag[rec_items])),
    }


def evaluate(top, truth_q, truth_raw, ptag, author, item_pop, tiers, secs=None) -> tuple[pd.DataFrame, dict]:
    pu = per_user(top, truth_q, truth_raw, ptag, item_pop, secs)
    summary = pu.mean().to_dict()
    summary.update(system_level(top, author, ptag, tiers, truth_raw.shape[1]))
    return pu, summary


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> tuple:
    """Mean of (b - a) over users with a 95% bootstrap interval. a, b aligned per user."""
    d = np.asarray(b) - np.asarray(a)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n, len(d)))
    boots = d[idx].mean(1)
    return float(d.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
