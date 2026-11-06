#!/usr/bin/env python3
"""What do the model's most confident windows look like?

Rebuilds the walk-forward validation samples (same config, same order), aligns
them with the cached out-of-sample scores, and characterises the top-q most
confident "up" and "down" calls: simple shape features and the average
normalised curve, compared with all windows."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common import RESULTS, ROOT, base_config, load_json, samples_for, save_json, setup_logging
from chartvision.dataset import split_at, walk_forward_folds
from confidence import load_oos

log = setup_logging("pattern")


def features(series: np.ndarray, ma_idx: int = 0, px_idx: int = 1) -> dict:
    px, ma = series[:, px_idx], series[:, ma_idx]
    r = np.diff(np.log(px))
    return {
        "ret_1d": px[-1] / px[-2] - 1, "ret_5d": px[-1] / px[-6] - 1, "ret_20d": px[-1] / px[-21] - 1,
        "ret_window": px[-1] / px[0] - 1,
        "px_vs_ma": px[-1] / ma[-1] - 1, "ma_slope_5d": ma[-1] / ma[-6] - 1,
        "pos_in_range": (px[-1] - px.min()) / (px.max() - px.min() + 1e-9),
        "drawdown_from_high": px[-1] / px.max() - 1, "runup_from_low": px[-1] / px.min() - 1,
        "vol_20d": r[-20:].std() * np.sqrt(252), "vol_ratio_5_20": r[-5:].std() / (r[-20:].std() + 1e-9),
    }


def main(name: str = "close_rel_base", q: float = 0.05, horizons=(1, 2)) -> None:
    wf = load_json(f"walkforward_{name}")
    ov = wf["overrides"]
    cfg = base_config(**ov)
    samples, _ = samples_for(cfg)
    folds = walk_forward_folds(samples, 3)
    val = []
    for cutoff, end in folds:
        val += split_at(samples, cutoff, end)[1]
    out = load_oos(name)
    assert len(val) == len(out["labels"]), (len(val), len(out["labels"]))
    assert (pd.to_datetime(out["dates"]) == pd.DatetimeIndex([s.end_date for s in val])).all()
    log.info("aligned %d validation windows", len(val))

    feats = pd.DataFrame([features(s.series) for s in val])
    tickers = np.array([s.ticker for s in val])
    norm = np.stack([(s.series[:, 1] / s.series[-1, 1] - 1) for s in val])   # price path, 0 at last day
    report, curves = {}, {}
    for i, h in enumerate(cfg.horizons):
        if h not in horizons:
            continue
        score, ret = out["score"][:, i], out["returns"][:, i]
        thr = np.quantile(np.abs(score), 1 - q)
        up, down = (score >= thr), (score <= -thr)
        groups = {"all": np.ones(len(val), bool), "confident_up": up, "confident_down": down}
        tab = {}
        for g, m in groups.items():
            d = feats[m].mean().round(4).to_dict()
            d["n"] = int(m.sum()); d["hit_rate"] = float((np.sign(score[m]) * np.sign(ret[m]) > 0).mean()) if g != "all" else float("nan")
            d["mean_fwd_ret_pct"] = float(ret[m].mean() * 100)
            d["n_tickers"] = int(len(set(tickers[m]))); d["top_tickers"] = pd.Series(tickers[m]).value_counts().head(5).to_dict()
            tab[g] = d
            curves[f"{h}d_{g}"] = norm[m].mean(0).tolist()
        report[h] = tab
        log.info("h=%d\n%s", h, pd.DataFrame(tab).to_string())
    save_json(f"pattern_{name}", {"name": name, "q": q, "features": report, "curves": curves})

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(horizons), figsize=(6 * len(horizons), 4), dpi=110)
    for ax, h in zip(np.atleast_1d(axes), horizons):
        x = np.arange(-cfg.window + 1, 1)
        for g, c in (("all", "#888"), ("confident_up", "#2a9d3a"), ("confident_down", "#d73a49")):
            ax.plot(x, np.array(curves[f"{h}d_{g}"]) * 100, color=c, label=g)
        ax.set_title(f"average price path before the call, horizon {h}d (top {q:.0%} confident)")
        ax.set_xlabel("days before the forecast"); ax.set_ylabel("% vs last close"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(RESULTS / f"pattern_{name}.png"); print("saved", RESULTS / f"pattern_{name}.png")


if __name__ == "__main__":
    main()
