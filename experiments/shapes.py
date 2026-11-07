#!/usr/bin/env python3
"""Do recurring chart shapes exist? (no prediction involved)

Part 1 - unsupervised: every 60-day close-price window is normalised to its
own min/max (exactly what the CNN sees) and clustered with k-means. We report
the cluster centroids (the "typical shapes"), how often each occurs, how tight
the cluster is, and - as a side note - what happened in the next 5/10 days.

Part 2 - named patterns: rule-based detectors for pullback-in-uptrend,
cup-and-handle, double bottom, breakout and V-bottom. We count how often each
fires, save example charts, and tabulate the forward returns versus the base
rate (t-stat clustered by date).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp
from sklearn.cluster import KMeans

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import RESULTS, ROOT, UNIVERSE_NAME, base_config, load_frames, setup_logging

log = setup_logging("shapes")
WIN, STRIDE, K, FWD = 60, 3, 12, (5, 10)
SUFFIX = "" if UNIVERSE_NAME == "large" else f"_{UNIVERSE_NAME}"


# --------------------------------------------------------------------------
def build_windows(frames: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Returns (normalised windows [N, WIN], raw windows [N, WIN+max(FWD)], meta)."""
    norm, raw, meta = [], [], []
    for t, df in frames.items():
        px = df["Close"].to_numpy(dtype=np.float64)
        for end in range(WIN - 1, len(px) - max(FWD) - 1, STRIDE):
            w = px[end - WIN + 1 : end + 1]
            lo, hi = w.min(), w.max()
            if hi <= lo or not np.isfinite(w).all():
                continue
            norm.append((w - lo) / (hi - lo))
            raw.append(px[end - WIN + 1 : end + 1 + max(FWD)])
            meta.append((t, df.index[end], *[px[end + h] / px[end] - 1 for h in FWD]))
    meta = pd.DataFrame(meta, columns=["ticker", "date", *[f"fwd_{h}" for h in FWD]])
    return np.array(norm, dtype=np.float32), np.array(raw), meta


def date_clustered_t(values: np.ndarray, dates: pd.Series, h: int) -> tuple[float, int]:
    s = pd.Series(values, index=pd.to_datetime(dates.values)).groupby(level=0).mean().sort_index()
    keep, last = [], None
    for d in s.index:
        if last is None or np.busday_count(last.date(), d.date()) >= h:
            keep.append(d); last = d
    v = s.loc[keep]
    return (float(ttest_1samp(v.values, 0).statistic) if len(v) > 2 else float("nan")), len(v)


# --------------------------------------------------------------------------
def cluster_shapes(X: np.ndarray, meta: pd.DataFrame) -> dict:
    km = KMeans(n_clusters=K, n_init=4, random_state=0).fit(X)
    labels, centers = km.labels_, km.cluster_centers_
    dist = np.linalg.norm(X - centers[labels], axis=1) / np.sqrt(WIN)   # mean abs-ish deviation 0..1
    rows = []
    for c in range(K):
        m = labels == c
        row = {"cluster": c, "share_pct": 100 * m.mean(), "n": int(m.sum()), "tightness": float(dist[m].mean()),
               "n_tickers": int(meta.ticker[m].nunique())}
        for h in FWD:
            r = meta[f"fwd_{h}"][m].values
            t, nd = date_clustered_t(r - meta[f"fwd_{h}"].mean(), meta.date[m], h)
            row[f"fwd_{h}_pct"] = 100 * r.mean(); row[f"fwd_{h}_excess_t"] = t
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("share_pct", ascending=False)
    order = table.cluster.tolist()

    fig, axes = plt.subplots(3, 4, figsize=(16, 9), dpi=110)
    for ax, c in zip(axes.ravel(), order):
        m = labels == c
        idx = np.where(m)[0]
        for i in np.random.default_rng(0).choice(idx, min(40, len(idx)), replace=False):
            ax.plot(X[i], color="#999", lw=0.4, alpha=0.5)
        ax.plot(centers[c], color="#d73a49", lw=2.2)
        r = table.set_index("cluster").loc[c]
        ax.set_title(f"shape {c}: {r.share_pct:.1f}% of windows, spread {r.tightness:.2f}\n"
                     f"next 5d {r.fwd_5_pct:+.2f}% (t={r.fwd_5_excess_t:+.1f}), 10d {r.fwd_10_pct:+.2f}% (t={r.fwd_10_excess_t:+.1f})", fontsize=8)
        ax.set_ylim(-0.05, 1.05); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"typical 60-day shapes ({UNIVERSE_NAME} universe, k-means, {len(X)} windows; grey = 40 random members)", fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT / "experiments" / f"shapes_clusters{SUFFIX}.png"); plt.close(fig)
    np.save(RESULTS / "shape_centers.npy", centers[order])
    return {"table": table.round(3).to_dict("records"), "inertia_per_window": float(km.inertia_ / len(X))}


# --------------------------------------------------------------------------
# named patterns (close prices; w = last 120 days where available, else 60)
def detect_patterns(frames: dict) -> tuple[pd.DataFrame, dict]:
    hits, base = [], []
    for t, df in frames.items():
        px = df["Close"].to_numpy(dtype=np.float64)
        n = len(px)
        for end in range(120, n - max(FWD) - 1):
            w = px[end - 119 : end + 1]            # 120 days ending today
            w60 = w[-60:]
            today = w[-1]
            fw = {h: px[end + h] / today - 1 for h in FWD}
            base.append((t, df.index[end], fw[5], fw[10]))
            found = []

            # 1. pullback in an uptrend: +15% over days -60..-5, then -3%..-10% in the last 5 days,
            #    still above the 50-day average
            ma50 = w[-50:].mean()
            if w[-6] / w[-60] - 1 > 0.15 and -0.10 < today / w[-6] - 1 < -0.03 and today > ma50:
                found.append("pullback")

            # 2. breakout: today's close above the previous 119-day high
            if today > w[:-1].max() * 1.001:
                found.append("breakout_120d_high")

            # 3. V-bottom: within the last 40 days a drop >= 15% from a high, then a rise >= 15% from the low,
            #    with the low in the middle
            w40 = w[-40:]
            i_lo = int(w40.argmin())
            if 8 <= i_lo <= 31:
                if w40[:i_lo].max() / w40[i_lo] - 1 >= 0.15 and today / w40[i_lo] - 1 >= 0.15:
                    found.append("v_bottom")

            # 4. double bottom: two lows within 3% of each other, >= 15 days apart, with a peak >= 8% above
            #    them in between, and price now above that peak
            lows = [i for i in range(5, 115) if w[i] == w[max(0, i - 5): i + 6].min()]
            dbl = False
            for a in range(len(lows)):
                for b in range(a + 1, len(lows)):
                    i, j = lows[a], lows[b]
                    if j - i < 15:
                        continue
                    if abs(w[i] / w[j] - 1) <= 0.03:
                        peak = w[i:j].max()
                        if peak / min(w[i], w[j]) - 1 >= 0.08 and today > peak and (120 - j) <= 30:
                            dbl = True
            if dbl:
                found.append("double_bottom")

            # 5. cup and handle: left rim high in days 0-30, cup low in the middle third 12%-35% below the rim,
            #    right rim within 5% of the left rim at days 85-110, then a handle dip of 3%-10% into today
            left = w[:30]; i_left = int(left.argmax()); rim = left[i_left]
            cup = w[30:85]; low = cup.min()
            right = w[85:110]; i_right = int(right.argmax()); rim2 = right[i_right]
            if (0.12 <= 1 - low / rim <= 0.35 and abs(rim2 / rim - 1) <= 0.05
                    and -0.10 <= today / rim2 - 1 <= -0.03 and low == w[20:100].min()):
                found.append("cup_and_handle")

            for f in found:
                hits.append((f, t, df.index[end], fw[5], fw[10]))

    hits = pd.DataFrame(hits, columns=["pattern", "ticker", "date", "fwd_5", "fwd_10"])
    base = pd.DataFrame(base, columns=["ticker", "date", "fwd_5", "fwd_10"])
    rows = []
    for pat, g in hits.groupby("pattern"):
        row = {"pattern": pat, "n": len(g), "per_1000_days": 1000 * len(g) / len(base), "n_tickers": g.ticker.nunique()}
        for h in FWD:
            row[f"fwd_{h}_pct"] = 100 * g[f"fwd_{h}"].mean()
            row[f"base_{h}_pct"] = 100 * base[f"fwd_{h}"].mean()
            row[f"excess_{h}_t"], row[f"n_dates_{h}"] = date_clustered_t(g[f"fwd_{h}"].values - base[f"fwd_{h}"].mean(), g.date, h)
        rows.append(row)
    summary = pd.DataFrame(rows)
    return hits, {"summary": summary.round(3).to_dict("records"), "n_stock_days": len(base)}


def plot_examples(frames: dict, hits: pd.DataFrame) -> None:
    pats = sorted(hits.pattern.unique())
    fig, axes = plt.subplots(len(pats), 4, figsize=(18, 3.2 * len(pats)), dpi=100)
    rng = np.random.default_rng(1)
    for r, pat in enumerate(pats):
        g = hits[hits.pattern == pat]
        picks = g.iloc[rng.choice(len(g), min(4, len(g)), replace=False)]
        for c, (_, h) in enumerate(picks.iterrows()):
            ax = axes[r, c] if len(pats) > 1 else axes[c]
            df = frames[h.ticker]; i = df.index.get_loc(h.date)
            seg = df["Close"].iloc[max(0, i - 119): i + 11]
            ax.plot(seg.index[: 120], seg.values[: 120], color="#1f6feb", lw=1.2)
            ax.plot(seg.index[119:], seg.values[119:], color="#d73a49", lw=1.2, ls="--")
            ax.axvline(h.date, color="#d73a49", lw=0.6)
            ax.set_title(f"{pat}: {h.ticker} {h.date.date()}  (next 10d {100*h.fwd_10:+.1f}%)", fontsize=8)
            ax.tick_params(labelsize=6); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(ROOT / "experiments" / f"patterns_examples{SUFFIX}.png"); plt.close(fig)


def main() -> None:
    cfg = base_config()
    frames = load_frames(cfg)
    log.info("%d stocks (%s)", len(frames), UNIVERSE_NAME)
    X, _, meta = build_windows(frames)
    log.info("%d windows", len(X))
    clusters = cluster_shapes(X, meta)
    log.info("clusters:\n%s", pd.DataFrame(clusters["table"]).to_string(index=False))
    hits, pats = detect_patterns(frames)
    log.info("named patterns (%d stock-days):\n%s", pats["n_stock_days"], pd.DataFrame(pats["summary"]).to_string(index=False))
    plot_examples(frames, hits)
    hits.to_csv(RESULTS / "pattern_hits.csv", index=False)
    with open(RESULTS / "shapes.json", "w") as f:
        json.dump({"universe": UNIVERSE_NAME, "clusters": clusters, "patterns": pats}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
