"""Walk-forward models that look at the index chart.

* logistic regression on the hand-made shape features (and level-only)
* a small CNN that sees the 60-day index chart as an image (optionally with
  the market price as a second line), using the chartvision rasterizer

Each year Y from `first_test_year` on is predicted by a model trained on all
data before Y (expanding window, one refit per year). Target: sign of the
forward return at each horizon. Reported: accuracy vs the base rate (share of
up days), Spearman IC with HAC t-stat, and the same for a label-shuffled null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from chartvision.charts import rasterize_window
from chartvision.model import ChartCNN
from .analysis import HORIZONS, hac_corr_t, hac_t

WINDOW, IMG = 60, 48


def _years(idx: pd.DatetimeIndex, first: int) -> list[int]:
    return [y for y in sorted(set(idx.year)) if y >= first]


def walk_forward_logit(features: pd.DataFrame, fwd: pd.DataFrame, first_test_year: int, horizons=HORIZONS,
                       cols: list[str] | None = None) -> dict:
    cols = cols or list(features.columns)
    df = pd.concat([features[cols], fwd], axis=1).dropna()
    res = {}
    for h in horizons:
        y = (df[f"fwd_{h}"] > 0).astype(int).values
        X = df[cols].values
        pred = np.full(len(df), np.nan); prob = np.full(len(df), np.nan)
        for yr in _years(df.index, first_test_year):
            tr = (df.index.year < yr) & ~np.isnan(X).any(1)
            # purge the last h days before the test year (their labels peek into it)
            tr &= df.index < (pd.Timestamp(f"{yr}-01-01") - pd.tseries.offsets.BDay(h))
            te = df.index.year == yr
            if tr.sum() < 200 or te.sum() == 0:
                continue
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(C=0.1, max_iter=500).fit(sc.transform(X[tr]), y[tr])
            p = clf.predict_proba(sc.transform(X[te]))[:, 1]
            prob[te] = p; pred[te] = (p > 0.5)
        m = ~np.isnan(pred)
        res[h] = _score(pred[m], prob[m], y[m], df[f"fwd_{h}"].values[m], h)
    return res


def _score(pred, prob, y, ret, h) -> dict:
    acc = float((pred == y).mean()); base = float(max(y.mean(), 1 - y.mean()))
    ic, t = hac_corr_t(prob, ret, h)
    # long/short by predicted probability: mean return of (2p-1)*ret
    ls = (2 * prob - 1) * ret
    t_ls, mean_ls = hac_t(ls, h)
    return {"n": int(len(y)), "accuracy": acc, "base_rate": base, "excess_acc": acc - base,
            "ic": ic, "ic_t": t, "ls_mean_pct": 100 * mean_ls, "ls_t": t_ls}


def _windows(series: pd.DataFrame, fwd: pd.DataFrame, window: int = WINDOW):
    X, dates, rets = [], [], []
    vals = series.values.astype(np.float32)
    for end in range(window - 1, len(series)):
        w = vals[end - window + 1 : end + 1]
        r = fwd.iloc[end].values
        if np.isfinite(w).all() and np.isfinite(r).all():
            X.append(rasterize_window(w, IMG).astype(np.float32) / 255.0); dates.append(series.index[end]); rets.append(r)
    return np.stack(X), pd.DatetimeIndex(dates), np.array(rets)


def walk_forward_cnn(series: pd.DataFrame, fwd: pd.DataFrame, first_test_year: int, horizons=HORIZONS,
                     epochs: int = 15, seed: int = 0, shuffle_labels: bool = False) -> dict:
    """`series`: DataFrame with 1-2 columns (index level, optionally price)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    X, dates, rets = _windows(series, fwd[[f"fwd_{h}" for h in horizons]])
    Y = (rets > 0).astype(np.int64)
    if shuffle_labels:
        perm = rng.permutation(len(Y)); Y = Y[perm]; rets_for_score = rets[perm]
    else:
        rets_for_score = rets
    probs = np.full((len(X), len(horizons)), np.nan)
    Xt = torch.from_numpy(X)
    for yr in _years(dates, first_test_year):
        tr = np.where((dates.year < yr) & (dates < pd.Timestamp(f"{yr}-01-01") - pd.tseries.offsets.BDay(max(horizons))))[0]
        te = np.where(dates.year == yr)[0]
        if len(tr) < 300 or len(te) == 0:
            continue
        model = ChartCNN(X.shape[1], len(horizons), 1, width=8, dropout=0.5)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
        yt = torch.from_numpy(Y)
        for ep in range(epochs):
            model.train()
            for b in np.array_split(rng.permutation(tr), max(1, len(tr) // 128)):
                logits, _ = model(Xt[b])
                # 3-way head used as binary: class 2 = up, class 0 = down
                loss = sum(F.cross_entropy(logits[:, i, [0, 2]], yt[b, i]) for i in range(len(horizons)))
                opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits, _ = model(Xt[te])
            probs[te] = torch.softmax(logits[:, :, [0, 2]], -1)[:, :, 1].numpy()
    res = {}
    for i, h in enumerate(horizons):
        m = ~np.isnan(probs[:, i])
        res[h] = _score((probs[m, i] > 0.5).astype(int), probs[m, i], Y[m, i], rets_for_score[m, i], h)
        res[h]["by_year"] = {int(y): float(((probs[m, i] > 0.5).astype(int) == Y[m, i])[dates[m].year == y].mean())
                             for y in sorted(set(dates[m].year))}
    return res
