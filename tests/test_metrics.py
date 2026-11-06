import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import gini, paired_bootstrap, per_user, system_level, topk_from_scores


def test_topk_skips_blocked_items():
    truth = np.array([[1.0, np.nan, 3.0, 0.5]])
    scores = np.array([[0.1, 9.0, 0.5, 0.2]])
    assert topk_from_scores(scores, truth, 2).tolist() == [[2, 3]]


def test_perfect_ranking_has_ndcg_one():
    truth = np.array([[0.2, 3.0, 1.0, 0.0]])
    top = topk_from_scores(truth.copy(), truth, 2)
    q = truth / 3
    pu = per_user(top, q, truth, ptag=np.array([0, 1, 1, 2]), item_pop=np.ones(4))
    assert np.isclose(pu.ndcg[0], 1.0) and np.isclose(pu.rel[0], 2 / 3)
    assert np.isclose(pu.mean_wr[0], 2.0)
    assert pu.full_rate[0] == 1.0 and pu.replay_rate[0] == 0.5
    assert pu.distinct_tags[0] == 1 and pu.ild[0] == 0


def test_ild_all_different():
    truth = np.ones((1, 3))
    pu = per_user(np.array([[0, 1, 2]]), truth, truth, ptag=np.array([0, 1, 2]), item_pop=np.ones(3))
    assert pu.ild[0] == 1


def test_gini():
    assert np.isclose(gini([1, 1, 1, 1]), 0)
    assert np.isclose(gini([0, 0, 0, 4]), 0.75)


def test_system_level_counts_starved_authors():
    author = np.array([10, 10, 20, 30])
    tiers = pd.Series(["head", "mid", "tail"], index=[10, 20, 30])
    s = system_level(np.array([[0, 1]]), author, np.array([0, 0, 1, 2]), tiers, 4)
    assert s["authors_reached"] == 1 and s["head_share"] == 1.0
    assert np.isclose(s["creator_gini"], gini([2, 0, 0]))


def test_bootstrap_contains_true_mean():
    a, b = np.zeros(500), np.full(500, 0.3) + np.random.default_rng(0).normal(0, 0.1, 500)
    m, lo, hi = paired_bootstrap(a, b)
    assert lo < 0.3 < hi
