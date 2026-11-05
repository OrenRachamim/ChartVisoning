"""Evaluation metrics that answer "is there a statistically usable signal?".

All functions take plain numpy arrays so they can score the CNN and the
baselines alike.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp

from .config import CLASS_NAMES


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    n_cls = len(CLASS_NAMES)
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    recalls = [cm[c, c] / cm[c].sum() if cm[c].sum() else 0.0 for c in range(n_cls)]
    out = {
        "accuracy": float((y_true == y_pred).mean()) if len(y_true) else float("nan"),
        "balanced_accuracy": float(np.mean(recalls)),
        "confusion": cm.tolist(),
    }
    for c, name in enumerate(CLASS_NAMES):
        col = cm[:, c]
        out[f"{name}_precision"] = float(cm[c, c] / col.sum()) if col.sum() else float("nan")
        out[f"{name}_share_of_calls"] = float(col.sum() / cm.sum()) if cm.sum() else float("nan")
    return out


def signal_metrics(score: np.ndarray, ret: np.ndarray, dates: np.ndarray, horizon: int,
                   quantile: float = 0.2) -> dict:
    """Information coefficient and a cross-sectional long/short spread.

    ``score`` is the model's directional score (p_up - p_down), ``ret`` the
    realised forward return, ``dates`` the window end dates. The spread is
    computed per date (top ``quantile`` of scores minus bottom ``quantile``),
    then averaged over dates spaced ``horizon`` business days apart so the
    t-statistic does not count overlapping periods.
    """
    out: dict = {}
    mask = np.isfinite(score) & np.isfinite(ret)
    score, ret, dates = score[mask], ret[mask], dates[mask]
    if len(score) < 10:
        return {"ic": float("nan"), "spread_mean": float("nan"), "spread_t": float("nan"), "n_dates": 0}
    ic, ic_p = spearmanr(score, ret)
    out["ic"] = float(ic)
    out["ic_pvalue"] = float(ic_p)

    df = pd.DataFrame({"score": score, "ret": ret, "date": pd.to_datetime(dates)})
    spreads, ns = [], []
    for d, g in df.groupby("date"):
        if len(g) < 10:
            continue
        k = max(1, int(len(g) * quantile))
        g = g.sort_values("score")
        spreads.append(float(g["ret"].tail(k).mean() - g["ret"].head(k).mean()))
        ns.append(d)
    if not spreads:
        out.update({"spread_mean": float("nan"), "spread_t": float("nan"), "n_dates": 0})
        return out
    sp = pd.Series(spreads, index=pd.DatetimeIndex(ns)).sort_index()
    # keep dates at least `horizon` business days apart -> (nearly) independent
    keep, last = [], None
    for d in sp.index:
        if last is None or np.busday_count(last.date(), d.date()) >= horizon:
            keep.append(d)
            last = d
    sp_ind = sp.loc[keep]
    t, p = ttest_1samp(sp_ind.values, 0.0) if len(sp_ind) > 2 else (float("nan"), float("nan"))
    out.update({
        "spread_mean": float(sp.mean()),           # per-period long/short return
        "spread_mean_pct": float(sp.mean() * 100),
        "spread_t": float(t),
        "spread_pvalue": float(p),
        "n_dates": int(len(sp)),
        "n_independent_dates": int(len(sp_ind)),
        "hit_rate": float((sp_ind > 0).mean()),
    })
    return out


def momentum_predictions(momentum: np.ndarray, thr_lo: float, thr_hi: float) -> np.ndarray:
    """Baseline: the line keeps going the way it went over the last 5 days.
    Thresholds are the class thresholds of the horizon being scored, scaled to
    a 5-day move (crude but transparent)."""
    return np.where(momentum > thr_hi, 2, np.where(momentum < thr_lo, 0, 1))


def summarize_horizon(y_true, y_pred, score, ret, dates, horizon, momentum, thr) -> dict:
    m = classification_metrics(y_true, y_pred)
    m.update(signal_metrics(score, ret, dates, horizon))
    maj = int(np.bincount(y_true, minlength=3).argmax())
    m["majority_acc"] = float((y_true == maj).mean())
    mom_pred = momentum_predictions(momentum, *thr)
    m["momentum_acc"] = float((mom_pred == y_true).mean())
    m["momentum_ic"] = float(spearmanr(momentum, ret)[0])
    m["momentum_signal"] = signal_metrics(momentum, ret, dates, horizon)
    return m
