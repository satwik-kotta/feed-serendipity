"""User-day panel: how narrow was a user's day, and did they come back the next day?"""
import numpy as np
import pandas as pd

N_TAGS = 31


def user_day_panel(f: pd.DataFrame, min_views: int = 20) -> pd.DataFrame:
    """f needs user_id, day, ptag, wr. Returns one row per active user-day with at least
    `min_views` views, plus next-day outcomes.

    nar_day      1 - normalised entropy of the day's main-tag mix
    log_views    log(1 + views today)
    wr_day       mean (clipped) watch ratio today
    log_next     log(1 + views tomorrow)
    next_active  1 if the user watched anything tomorrow
    log_next3    log(1 + views over the next 3 days)
    """
    c = f.groupby(["user_id", "day", "ptag"]).size().rename("n").reset_index()
    c["tot"] = c.groupby(["user_id", "day"]).n.transform("sum")
    p = c.n / c.tot
    c["h"] = -p * np.log(p)
    ud = c.groupby(["user_id", "day"]).agg(h=("h", "sum"), views=("tot", "first")).reset_index()
    ud["nar_day"] = 1 - ud.h / np.log(N_TAGS)
    ud = ud.merge(f.groupby(["user_id", "day"]).wr.mean().rename("wr_day").reset_index(),
                  on=["user_id", "day"])

    days = range(int(f.day.min()), int(f.day.max()) + 1)
    full = pd.MultiIndex.from_product([ud.user_id.unique(), days], names=["user_id", "day"])
    pan = ud.set_index(["user_id", "day"]).reindex(full).reset_index()
    pan["views"] = pan.views.fillna(0)
    g = pan.groupby("user_id").views
    nxt = g.shift(-1)
    pan["log_next"] = np.log1p(nxt)
    pan["next_active"] = (nxt > 0).astype(float).where(nxt.notna())
    pan["log_next3"] = np.log1p(sum(g.shift(-k) for k in (1, 2, 3)))
    pan["log_views"] = np.log1p(pan.views)
    out = pan[pan.views >= min_views].dropna(subset=["nar_day", "log_next"]).reset_index(drop=True)
    return out
