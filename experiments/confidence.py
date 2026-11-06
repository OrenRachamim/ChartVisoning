#!/usr/bin/env python3
"""Test A - is the model right more often when it is confident?

Uses the cached out-of-sample walk-forward predictions (results/wf_<name>_fold*.npz).
Confidence = |p_up - p_down|. For the top q of confident windows we report the
directional hit rate (sign of the call vs sign of the realised return), the
3-class accuracy, the mean long/short return and a per-date t-stat, and compare
the top-5% hit rate with a permutation null (scores shuffled within each date).
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp

from common import RESULTS, save_json, setup_logging

log = setup_logging("confidence")
QUANTILES = (1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01)


def load_oos(name: str) -> dict:
    files = sorted(RESULTS.glob(f"wf_{name}_fold*.npz"))
    if not files:
        raise SystemExit(f"no cached folds for {name}")
    parts = [np.load(f, allow_pickle=True) for f in files]
    keys = ["pred", "score", "labels", "returns", "dates", "momentum"]
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def per_date_t(df: pd.DataFrame, horizon: int) -> tuple[float, int]:
    """t-stat of the mean long/short return, on dates spaced `horizon` apart."""
    daily = df.groupby("date")["ls"].mean().sort_index()
    keep, last = [], None
    for d in daily.index:
        if last is None or np.busday_count(last.date(), d.date()) >= horizon:
            keep.append(d); last = d
    v = daily.loc[keep]
    if len(v) < 3:
        return float("nan"), len(v)
    return float(ttest_1samp(v.values, 0.0).statistic), len(v)


def analyse(out: dict, horizons: tuple[int, ...], n_perm: int = 200, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    res = {}
    dates = pd.to_datetime(out["dates"])
    for i, h in enumerate(horizons):
        score, ret, lab, pred = out["score"][:, i], out["returns"][:, i], out["labels"][:, i], out["pred"][:, i]
        conf = np.abs(score)
        rows = []
        for q in QUANTILES:
            thr = np.quantile(conf, 1 - q) if q < 1 else -1
            m = conf >= thr
            direction = np.sign(score[m])
            hit = (direction * np.sign(ret[m]) > 0)
            ls = direction * ret[m]                     # long when up, short when down
            df = pd.DataFrame({"date": dates[m], "ls": ls})
            t, nd = per_date_t(df, h)
            rows.append({"top_q": q, "n": int(m.sum()), "hit_rate": float(hit.mean()),
                         "acc_3class": float((pred[m] == lab[m]).mean()),
                         "mean_ls_pct": float(ls.mean() * 100), "t_stat": t, "n_dates": nd})
        # permutation null for the top-5% hit rate: shuffle scores within each date
        q = 0.05
        df = pd.DataFrame({"date": dates, "score": score, "ret": ret})
        null = []
        for _ in range(n_perm):
            sh = df.groupby("date")["score"].transform(lambda x: rng.permutation(x.values))
            c = np.abs(sh.values)
            m = c >= np.quantile(c, 1 - q)
            null.append(float((np.sign(sh.values[m]) * np.sign(ret[m]) > 0).mean()))
        obs = rows[QUANTILES.index(0.05)]["hit_rate"]
        null = np.array(null)
        res[h] = {"table": rows, "top5_hit_rate": obs, "null_mean": float(null.mean()),
                  "null_std": float(null.std()), "null_p_value": float((null >= obs).mean()),
                  "z": float((obs - null.mean()) / null.std()) if null.std() > 0 else float("nan")}
        log.info("h=%2d  top5%% hit %.3f  null %.3f±%.3f  p=%.3f   | all: hit %.3f t=%.2f",
                 h, obs, null.mean(), null.std(), res[h]["null_p_value"], rows[0]["hit_rate"], rows[0]["t_stat"])
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--names", nargs="+", default=["close_win120_img64"])
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3, 5, 10])
    args = p.parse_args()
    for name in args.names:
        out = load_oos(name)
        log.info("=== confidence analysis %s (%d OOS windows)", name, len(out["labels"]))
        res = analyse(out, tuple(args.horizons))
        save_json(f"confidence_{name}", {"name": name, "n": int(len(out["labels"])), "horizons": res})


if __name__ == "__main__":
    main()
