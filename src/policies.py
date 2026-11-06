"""Re-ranking policies that sit on top of the baseline scores.

Every policy takes the baseline score matrix (users x items, small-matrix items)
and returns a top-K list per user plus a mask of which slots were "injected"
(not chosen by relevance alone).

    baseline            top K by baseline score
    random_explore      2 slots replaced with random videos
    adjacent_all        2 slots from adjacent tags, for every user
    triggered           2 slots from adjacent tags only for users whose recent feed is narrow,
                        plus a streak breaker for everyone (no run of 4+ videos with the same tag)
    mmr                 greedy re-rank that penalises repeating a tag already in the list
    mmr_triggered       MMR + the triggered discovery slots + streak breaker (the recommended policy)

User state (narrowness, trailing streak, tag history) comes from each user's big-matrix
history, so it is what a live system would know before serving the feed:

    python -m src.policies state
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "data" / "models"
K = 10
INJECT_SLOTS = (3, 7)          # 0-based positions 4 and 8 in the list
NARROW_TRIGGER = 0.378         # 60th percentile of rolling narrowness (Phase 2)
STREAK_CAP = 4                 # once 4 in a row share a tag, the next must differ (Phase 2: watch ratio drops after 4+)
RARE_SHARE = 0.03              # a tag counts as "new" for a user below this share of their history
POOL = 300                     # candidate pool per user for re-ranking


# ---------------------------------------------------------------- state (needs the big matrix)

def step_state():
    from src.load import load_matrix, load_tags
    from src.narrowness import _entropy_narrowness, primary_tag, tag_matrix

    z = np.load(ART / "small_scores.npz")
    users = z["users"]
    big = load_matrix("big")[["user_id", "video_id", "timestamp"]]
    big = big.sort_values(["user_id", "timestamp"], kind="stable")
    tags = load_tags()
    n_items = int(max(big.video_id.max(), tags.video_id.max())) + 1
    n_tags = int(tags.tags.explode().max()) + 1
    M = tag_matrix(tags, n_items, n_tags)
    pt = primary_tag(tags, n_items)
    big["ptag"] = pt[big.video_id.to_numpy()]

    # Tag shares of every big-matrix user (for adjacency) and of the small-matrix users (for "rare")
    share_all = (big.groupby(["user_id", "ptag"]).size().unstack(fill_value=0)
                 .reindex(columns=range(n_tags), fill_value=0))
    share_all = share_all.div(share_all.sum(1), axis=0)
    adj = np.corrcoef(share_all.to_numpy().T)          # audience overlap between tags
    adj = np.nan_to_num(adj)
    np.fill_diagonal(adj, 0)

    last = big[big.user_id.isin(users)].groupby("user_id").tail(20)
    W = M[last.video_id.to_numpy()]
    S = pd.DataFrame(W).groupby(last.user_id.to_numpy()).sum().reindex(users).to_numpy()
    narrow = _entropy_narrowness(S, n_tags)
    recent_share = S / S.sum(1, keepdims=True)

    tail = big[big.user_id.isin(users)].groupby("user_id").ptag.apply(
        lambda s: (s.iloc[-1], int((s.iloc[::-1] != s.iloc[-1]).to_numpy().argmax() or len(s))))
    tail = tail.reindex(users)
    np.savez(ART / "user_state.npz", users=users, narrow=narrow.astype(np.float32),
             recent_share=recent_share.astype(np.float32),
             hist_share=share_all.reindex(users).to_numpy(np.float32),
             tail_tag=np.array([t[0] for t in tail]), tail_len=np.array([t[1] for t in tail]),
             adj=adj.astype(np.float32))
    print(f"State for {len(users):,} users. Narrow-trigger share at {NARROW_TRIGGER}: "
          f"{(narrow >= NARROW_TRIGGER).mean():.1%}", flush=True)


# ---------------------------------------------------------------- helpers

def masked(scores, truth):
    """Blocked items (no ground truth) can never be recommended."""
    return np.where(np.isnan(truth), -np.inf, scores)


def candidate_pool(S, pool=POOL):
    pool = min(pool, S.shape[1] - 1)
    part = np.argpartition(-S, pool, axis=1)[:, :pool]
    order = np.argsort(-np.take_along_axis(S, part, 1), axis=1)
    return np.take_along_axis(part, order, 1)


def insert_at(base_row, extra, slots=INJECT_SLOTS, k=K):
    """Put `extra` items at `slots`, fill the rest from base_row in order (skipping duplicates)."""
    out = [-1] * k
    for s, e in zip(slots, extra):
        out[s] = e
    taken = set(extra)
    it = (i for i in base_row if i not in taken)
    for p in range(k):
        if out[p] == -1:
            out[p] = next(it)
    return out


def adjacent_tags(recent_share, hist_share, adj, n=2, rare=RARE_SHARE):
    """Tags close to what the user watches now, that they rarely watch overall."""
    score = recent_share @ adj
    score = np.where(hist_share < rare, score, -np.inf)
    order = np.argsort(-score)
    return [t for t in order[:n] if np.isfinite(score[t])]


def break_streaks(row, pool_row, ptag, tail_tag, tail_len, cap=STREAK_CAP, locked=()):
    """Replace items that would extend a run of `cap` or more same-tag videos (counting the user's history)
    with the best pooled item of a different tag. Slots in `locked` are not changed."""
    row = list(row)
    used = set(row)
    run_tag, run_len = tail_tag, tail_len
    for p in range(len(row)):
        t = ptag[row[p]]
        if t == run_tag and run_len >= cap and p not in locked:
            for c in pool_row:
                if c not in used and ptag[c] != t:
                    used.discard(row[p]); row[p] = c; used.add(c); t = ptag[c]
                    break
        run_len = run_len + 1 if t == run_tag else 1
        run_tag = t
    return row


# ---------------------------------------------------------------- policies

def baseline(S, k=K):
    top = candidate_pool(S, k)
    return top, np.zeros_like(top, dtype=bool)


def random_explore(S, k=K, seed=0):
    rng = np.random.default_rng(seed)
    pool = candidate_pool(S, k)
    U, I = S.shape
    top, inj = np.empty((U, k), int), np.zeros((U, k), bool)
    for u in range(U):
        ok = np.flatnonzero(np.isfinite(S[u]))
        extra = []
        while len(extra) < len(INJECT_SLOTS):
            c = int(rng.choice(ok))
            if c not in pool[u] and c not in extra:
                extra.append(c)
        top[u] = insert_at(pool[u], extra)
        inj[u, list(INJECT_SLOTS)] = True
    return top, inj


def _adjacent_row(S_u, pool_u, ptag, state_u, adj, n=2):
    tags = adjacent_tags(state_u["recent_share"], state_u["hist_share"], adj, n=n)
    extra = []
    for t in tags:
        cand = np.flatnonzero((ptag == t) & np.isfinite(S_u))
        cand = [c for c in cand[np.argsort(-S_u[cand])] if c not in pool_u[:K] and c not in extra]
        if cand:
            extra.append(int(cand[0]))
    return extra


def adjacent_all(S, ptag, state, k=K):
    pool = candidate_pool(S)
    U = S.shape[0]
    top, inj = np.empty((U, k), int), np.zeros((U, k), bool)
    for u in range(U):
        extra = _adjacent_row(S[u], pool[u], ptag, _state(state, u), state["adj"])
        top[u] = insert_at(pool[u], extra)
        inj[u, list(INJECT_SLOTS[:len(extra)])] = True
    return top, inj


def triggered(S, ptag, state, k=K, threshold=NARROW_TRIGGER, streak=True, inject=True):
    pool = candidate_pool(S)
    U = S.shape[0]
    top, inj = np.empty((U, k), int), np.zeros((U, k), bool)
    for u in range(U):
        st = _state(state, u)
        extra = []
        if inject and st["narrow"] >= threshold:
            extra = _adjacent_row(S[u], pool[u], ptag, st, state["adj"])
        row = insert_at(pool[u], extra)
        slots = INJECT_SLOTS[:len(extra)]
        if streak:
            before = list(row)
            row = break_streaks(row, pool[u], ptag, st["tail_tag"], st["tail_len"], locked=slots)
            inj[u] |= np.array(row) != np.array(before)
        top[u] = row
        inj[u, list(slots)] = True
    return top, inj


def mmr(S, ptag, lam, k=K):
    """Greedy: next item maximises (rank score) - lam * (items already in the list with the same tag).
    Rank score is the item's percentile within the user's candidate pool, so lam is in the same units."""
    pool = candidate_pool(S)
    P = pool.shape[1]
    rel = 1 - np.arange(P) / P
    U = S.shape[0]
    top, inj = np.empty((U, k), int), np.zeros((U, k), bool)
    for u in range(U):
        tags = ptag[pool[u]]
        counts, chosen, avail = {}, [], np.ones(P, bool)
        for _ in range(k):
            pen = np.array([counts.get(t, 0) for t in tags])
            val = np.where(avail, rel - lam * pen, -np.inf)
            j = int(val.argmax())
            chosen.append(pool[u, j]); avail[j] = False
            counts[tags[j]] = counts.get(tags[j], 0) + 1
        top[u] = chosen
        inj[u] = ~np.isin(chosen, pool[u, :k])
    return top, inj


SLOT_LAYOUT = {0: (), 1: (4,), 2: INJECT_SLOTS, 3: (2, 5, 8), 4: (2, 4, 6, 8)}


def mmr_triggered(S, ptag, state, lam, k=K, threshold=NARROW_TRIGGER, n_slots=2, base_top=None):
    """The recommended policy: MMR ordering for variety among familiar tags, then for narrow users
    `n_slots` adjacent-tag discovery slots, then the streak breaker for everyone.
    Pass `base_top` (an MMR result for the same lam) to skip recomputing MMR in a sweep."""
    pool = candidate_pool(S)
    if base_top is None:
        base_top, _ = mmr(S, ptag, lam, k)
    layout = SLOT_LAYOUT[n_slots]
    U = S.shape[0]
    top, inj = np.empty((U, k), int), np.zeros((U, k), bool)
    for u in range(U):
        st = _state(state, u)
        seen = set(base_top[u])
        order = list(base_top[u]) + [i for i in pool[u] if i not in seen]
        extra = (_adjacent_row(S[u], np.array(order), ptag, st, state["adj"], n=n_slots)
                 if n_slots and st["narrow"] >= threshold else [])
        slots = layout[:len(extra)]
        row = insert_at(order, extra, slots=slots)
        row = break_streaks(row, order, ptag, st["tail_tag"], st["tail_len"], locked=slots)
        top[u] = row
        inj[u] = ~np.isin(row, pool[u, :k])
    return top, inj


def new_to_user(top, ptag, hist_share, rare=RARE_SHARE):
    """Per user: how many of the K slots are tags under `rare` share of that user's history."""
    return np.array([(hist_share[u][ptag[top[u]]] < rare).sum() for u in range(len(top))])


def _state(state, u):
    return {"narrow": state["narrow"][u], "recent_share": state["recent_share"][u],
            "hist_share": state["hist_share"][u], "tail_tag": state["tail_tag"][u],
            "tail_len": state["tail_len"][u]}


if __name__ == "__main__":
    {"state": step_state}[sys.argv[1]]()
