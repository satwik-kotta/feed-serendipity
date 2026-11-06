import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.policies import adjacent_tags, baseline, break_streaks, insert_at, mmr, mmr_triggered, new_to_user


def test_insert_at_keeps_order_and_slots():
    assert insert_at([10, 11, 12, 13, 14], [99, 98], slots=(1, 3), k=5) == [10, 99, 11, 98, 12]


def test_break_streaks_counts_history():
    ptag = np.array([0, 0, 0, 0, 1, 0])
    # history already ends with 3 videos of tag 0 -> the second tag-0 item in the list makes 4,
    # so the third (5th in a row) gets replaced
    row = break_streaks([0, 1, 2, 3], pool_row=[0, 1, 2, 3, 5, 4], ptag=ptag, tail_tag=0, tail_len=3, cap=4)
    assert row[:1] == [0] and 4 in row
    run, best = 0, 0
    last = 0
    for t in [0, 0, 0] + list(ptag[row]):
        run = run + 1 if t == last else 1; last = t; best = max(best, run)
    assert best <= 4


def test_break_streaks_respects_locked():
    ptag = np.array([0, 0, 0, 0, 0, 1])
    row = break_streaks([0, 1, 2, 3, 4], [0, 1, 2, 3, 4, 5], ptag, tail_tag=-1, tail_len=0, cap=4, locked=(4,))
    assert row[4] == 4


def test_adjacent_tags_skips_watched():
    adj = np.array([[0, .9, .5], [.9, 0, .1], [.5, .1, 0]])
    recent = np.array([1.0, 0, 0])
    hist = np.array([0.9, 0.08, 0.0])
    assert adjacent_tags(recent, hist, adj, n=2) == [2]


def test_mmr_zero_lambda_equals_baseline():
    rng = np.random.default_rng(0)
    S = rng.random((5, 400))
    ptag = rng.integers(0, 5, 400)
    a, _ = baseline(S)
    b, inj = mmr(S, ptag, lam=0.0)
    assert (a == b).all() and not inj.any()


def test_mmr_high_lambda_diversifies():
    rng = np.random.default_rng(1)
    S = rng.random((5, 400))
    ptag = rng.integers(0, 10, 400)
    b, _ = mmr(S, ptag, lam=5.0)
    assert all(len(set(ptag[r])) == 10 for r in b)


def test_mmr_triggered_without_trigger_is_mmr_plus_breaker():
    rng = np.random.default_rng(2)
    S = rng.random((4, 400))
    ptag = rng.integers(0, 6, 400)
    T = 6
    state = {"narrow": np.zeros(4), "recent_share": np.full((4, T), 1 / T), "hist_share": np.full((4, T), 1 / T),
             "tail_tag": np.full(4, -1), "tail_len": np.zeros(4, int), "adj": np.zeros((T, T))}
    a, _ = mmr(S, ptag, 0.05)
    b, _ = mmr_triggered(S, ptag, state, 0.05)
    assert (a == b).all()


def test_new_to_user():
    ptag = np.array([0, 1, 2])
    hist = np.array([[0.9, 0.08, 0.02]])
    assert new_to_user(np.array([[0, 1, 2]]), ptag, hist).tolist() == [1]
