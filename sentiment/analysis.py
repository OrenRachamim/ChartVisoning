"""Statistics and feature engineering for the sentiment study.

Everything here is built for ONE time series (the index) against ONE target
(the market), where overlapping forward returns make naive t-stats badly
overstated. We therefore use Newey-West (HAC) standard errors with lag =
horizon, report the number of non-overlapping observations, and use a
circular block bootstrap of the return series as the null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HORIZONS = (1, 5, 10, 21, 63)
BUCKETS = [(0, 25, "extreme fear"), (25, 45, "fear"), (45, 55, "neutral"), (55, 75, "greed"), (75, 101, "extreme greed")]


# --------------------------------------------------------------------------
def forward_returns(price: pd.Series, horizons=HORIZONS) -> pd.DataFrame:
    out = pd.DataFrame(index=price.index)
    for h in horizons:
        out[f"fwd_{h}"] = np.log(price.shift(-h) / price)
    return out


def hac_t(x: np.ndarray, lag: int) -> tuple[float, float]:
    """Newey-West t-stat of the mean of x (Bartlett kernel, `lag` lags)."""
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return float("nan"), float("nan")
    e = x - x.mean()
    s = (e ** 2).sum()
    for k in range(1, min(lag, n - 1) + 1):
        w = 1 - k / (lag + 1)
        s += 2 * w * (e[k:] * e[:-k]).sum()
    se = np.sqrt(max(s, 1e-18)) / n
    return float(x.mean() / se), float(x.mean())


def hac_corr_t(x: np.ndarray, y: np.ndarray, lag: int) -> tuple[float, float]:
    """Spearman correlation and a HAC t-stat for it (via the regression of
    ranked y on ranked x)."""
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 30:
        return float("nan"), float("nan")
    rho = spearmanr(x, y)[0]
    rx = (pd.Series(x).rank().values - 0.5 * len(x)); ry = (pd.Series(y).rank().values - 0.5 * len(y))
    rx /= rx.std(); ry /= ry.std()
    # slope of ry on rx = rho; HAC t of the score contributions rx*ry
    t, _ = hac_t(rx * ry, lag)
    return float(rho), t


def bucket_table(level: pd.Series, fwd: pd.DataFrame, horizons=HORIZONS, buckets=BUCKETS) -> pd.DataFrame:
    df = pd.concat([level.rename("level"), fwd], axis=1).dropna()
    rows = []
    for lo, hi, name in buckets:
        m = (df.level >= lo) & (df.level < hi)
        row = {"bucket": name, "range": f"{lo}-{hi if hi < 101 else 100}", "n_days": int(m.sum()), "share_pct": 100 * m.mean()}
        for h in horizons:
            r = df.loc[m, f"fwd_{h}"].values; base = df[f"fwd_{h}"].values
            t, mean = hac_t(r - base.mean(), h)   # excess over unconditional, HAC
            row[f"ret_{h}_pct"] = 100 * r.mean() if len(r) else np.nan
            row[f"base_{h}_pct"] = 100 * base.mean()
            row[f"hit_{h}"] = float((r > 0).mean()) if len(r) else np.nan
            row[f"t_{h}"] = t
            row[f"n_indep_{h}"] = int(m.sum() / h)
        rows.append(row)
    return pd.DataFrame(rows)


def ic_table(features: pd.DataFrame, fwd: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    rows = []
    for col in features.columns:
        row = {"feature": col}
        for h in horizons:
            d = pd.concat([features[col], fwd[f"fwd_{h}"]], axis=1).dropna()
            rho, t = hac_corr_t(d.iloc[:, 0].values, d.iloc[:, 1].values, h)
            row[f"ic_{h}"] = rho; row[f"t_{h}"] = t
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
def shape_features(level: pd.Series) -> pd.DataFrame:
    """Hand-made descriptions of the index chart over the last 5-60 days."""
    f = pd.DataFrame(index=level.index)
    f["level"] = level
    for d in (5, 10, 20):
        f[f"chg_{d}d"] = level - level.shift(d)
    f["min_20d"] = level.rolling(20).min(); f["max_20d"] = level.rolling(20).max()
    f["above_min20"] = level - f["min_20d"]; f["below_max20"] = f["max_20d"] - level
    f["z_60d"] = (level - level.rolling(60).mean()) / level.rolling(60).std()
    x = np.arange(10)
    f["slope_10d"] = level.rolling(10).apply(lambda w: np.polyfit(x, w, 1)[0], raw=True)
    f["rebound_from_fear"] = ((f["min_20d"] < 25) & (f["above_min20"] > 10)).astype(float)
    f["rollover_from_greed"] = ((f["max_20d"] > 75) & (f["below_max20"] > 10)).astype(float)
    f["days_since_fear"] = _days_since(level < 25)
    f["days_since_greed"] = _days_since(level > 75)
    return f


def _days_since(cond: pd.Series) -> pd.Series:
    out, last = [], None
    for i, c in enumerate(cond.values):
        if c:
            last = i
        out.append(np.nan if last is None else i - last)
    return pd.Series(out, index=cond.index).clip(upper=250)


# --------------------------------------------------------------------------
def event_study(level: pd.Series, price: pd.Series, enter_below: float | None = None, enter_above: float | None = None,
                max_h: int = 63, min_gap: int = 21) -> dict:
    """Forward return paths after the index crosses INTO a zone (first day in
    the zone after at least `min_gap` days outside)."""
    cond = (level < enter_below) if enter_below is not None else (level > enter_above)
    idx = level.index
    events, last = [], -10_000
    pos = {d: i for i, d in enumerate(price.index)}
    for i in range(1, len(cond)):
        if cond.iloc[i] and not cond.iloc[i - 1] and i - last >= min_gap:
            d = idx[i]
            if d in pos and pos[d] + max_h < len(price):
                events.append(d); last = i
    paths = np.array([[np.log(price.iloc[pos[d] + h] / price.iloc[pos[d]]) for h in range(1, max_h + 1)] for d in events])
    if not len(paths):
        return {"n": 0}
    rng = np.random.default_rng(0)
    boot = np.array([paths[rng.integers(0, len(paths), len(paths))].mean(0) for _ in range(2000)])
    return {"n": len(events), "dates": [str(d.date()) for d in events], "mean_path_pct": (100 * paths.mean(0)).tolist(),
            "ci_lo_pct": (100 * np.percentile(boot, 2.5, axis=0)).tolist(), "ci_hi_pct": (100 * np.percentile(boot, 97.5, axis=0)).tolist(),
            "hit_21d": float((paths[:, 20] > 0).mean()), "hit_63d": float((paths[:, -1] > 0).mean()),
            "mean_21d_pct": float(100 * paths[:, 20].mean()), "mean_63d_pct": float(100 * paths[:, -1].mean())}


# --------------------------------------------------------------------------
def block_bootstrap_null(level: pd.Series, price: pd.Series, stat_fn, n_boot: int = 500, block: int = 21, seed: int = 0) -> np.ndarray:
    """Recompute `stat_fn(level, price)` on prices rebuilt from circularly
    block-resampled daily returns (the index stays as is). Returns the null
    distribution of the statistic."""
    r = np.log(price).diff().dropna().values
    n = len(r)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        starts = rng.integers(0, n, size=n // block + 1)
        idx = np.concatenate([(s + np.arange(block)) % n for s in starts])[:n]
        p = pd.Series(np.exp(np.concatenate([[0.0], np.cumsum(r[idx])])) * price.iloc[0], index=price.index)
        out.append(stat_fn(level, p))
    return np.array(out)


def strategy(level: pd.Series, price: pd.Series, rule) -> dict:
    """Daily rebalanced long/flat strategy. `rule(level_df) -> exposure in {0,1}`
    evaluated on information available at the close, applied to the next day."""
    px = price.reindex(level.index).dropna()
    lv = level.reindex(px.index)
    expo = rule(lv).shift(1).fillna(0.0)
    r = np.log(px).diff().fillna(0.0)
    strat = expo * r
    def stats(x):
        ann = 252
        cagr = np.exp(x.sum() / len(x) * ann) - 1
        sharpe = x.mean() / x.std() * np.sqrt(ann) if x.std() > 0 else np.nan
        eq = np.exp(x.cumsum()); dd = (eq / eq.cummax() - 1).min()
        return {"cagr_pct": 100 * cagr, "sharpe": sharpe, "max_dd_pct": 100 * dd}
    return {"strategy": stats(strat), "buy_hold": stats(r), "exposure": float(expo.mean()), "n_days": len(r),
            "equity_strategy": np.exp(strat.cumsum()), "equity_bh": np.exp(r.cumsum())}
