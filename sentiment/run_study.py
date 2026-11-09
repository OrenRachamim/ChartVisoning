#!/usr/bin/env python3
"""Run the whole sentiment study and write sentiment/results/*.json + figures."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sentiment import analysis as A  # noqa: E402
from sentiment import ml  # noqa: E402
from sentiment.data import SDIR, cnn_fear_greed, crypto_fear_greed, market_data  # noqa: E402

OUT = ROOT / "sentiment" / "results"; OUT.mkdir(exist_ok=True)
FIG = ROOT / "sentiment" / "figures"; FIG.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("study")


def vix_to_index(vix: pd.Series, window: int = 500) -> pd.Series:
    """VIX turned into a 0..100 greed-style score: 100 - rolling percentile."""
    return 100 - pd.Series(vix).rolling(window, min_periods=250).apply(lambda w: (w[:-1] < w[-1]).mean() * 100, raw=True)


def study(name: str, level: pd.Series, price: pd.Series, first_test_year: int, run_ml: bool = True, n_boot: int = 300) -> dict:
    level = level.dropna(); price = price.dropna()
    common = level.index.intersection(price.index)
    level, price = level.loc[common], price.loc[common]
    fwd = A.forward_returns(price)
    res = {"name": name, "n_days": int(len(level)), "start": str(level.index[0].date()), "end": str(level.index[-1].date())}
    log.info("=== %s: %d days %s -> %s", name, len(level), res["start"], res["end"])

    # A. level buckets
    bt = A.bucket_table(level, fwd)
    res["buckets"] = bt.round(4).to_dict("records")
    log.info("buckets:\n%s", bt[["bucket", "n_days", "ret_5_pct", "t_5", "ret_21_pct", "t_21", "ret_63_pct", "t_63"]].round(3).to_string(index=False))

    # B. shape features: IC
    feats = A.shape_features(level)
    ict = A.ic_table(feats, fwd)
    res["ic"] = ict.round(4).to_dict("records")
    log.info("IC (HAC t):\n%s", ict[["feature", "ic_5", "t_5", "ic_21", "t_21", "ic_63", "t_63"]].round(3).to_string(index=False))

    # C. events
    res["events"] = {"into_extreme_fear": A.event_study(level, price, enter_below=20),
                     "into_fear": A.event_study(level, price, enter_below=25),
                     "into_extreme_greed": A.event_study(level, price, enter_above=80),
                     "into_greed": A.event_study(level, price, enter_above=75)}
    for k, e in res["events"].items():
        if e["n"]:
            log.info("event %-20s n=%3d  21d %+.2f%% (hit %.2f)  63d %+.2f%% (hit %.2f)  CI63 [%+.2f, %+.2f]", k, e["n"],
                     e["mean_21d_pct"], e["hit_21d"], e["mean_63d_pct"], e["hit_63d"], e["ci_lo_pct"][-1], e["ci_hi_pct"][-1])

    # D. block-bootstrap null for the key level statistic: 21d return in extreme fear minus in extreme greed
    def spread_stat(lv, px):
        f = A.forward_returns(px, (21,))["fwd_21"]
        d = pd.concat([lv, f], axis=1).dropna(); d.columns = ["l", "r"]
        a, b = d.r[d.l < 25].mean(), d.r[d.l >= 75].mean()
        return (a - b) if np.isfinite(a) and np.isfinite(b) else np.nan
    obs = spread_stat(level, price)
    null = A.block_bootstrap_null(level, price, spread_stat, n_boot=n_boot)
    null = null[np.isfinite(null)]
    res["null_fear_minus_greed_21d"] = {"observed_pct": 100 * obs, "null_mean_pct": 100 * null.mean(), "null_sd_pct": 100 * null.std(),
                                        "p_value": float((null >= obs).mean()) if np.isfinite(obs) else float("nan")}
    log.info("fear-minus-greed 21d: observed %+.2f%%, null %+.2f±%.2f%%, p=%.3f", 100 * obs, 100 * null.mean(), 100 * null.std(), res["null_fear_minus_greed_21d"]["p_value"])

    # E. strategies
    strat = {}
    rules = {"long_in_fear_lt25": lambda lv: (lv < 25).astype(float),
             "long_unless_greed_gt75": lambda lv: (lv <= 75).astype(float),
             "long_after_rebound_from_fear": lambda lv: ((lv.rolling(20).min() < 25) & (lv - lv.rolling(20).min() > 10)).astype(float),
             "long_when_rising_5d": lambda lv: (lv.diff(5) > 0).astype(float)}
    for k, rule in rules.items():
        s = A.strategy(level, price, rule)
        strat[k] = {kk: vv for kk, vv in s.items() if not kk.startswith("equity")}
        # random-timing null: same exposure, random (block) days
        rng = np.random.default_rng(1); sh = []
        expo = rule(level.reindex(price.index).ffill()).fillna(0)
        r = np.log(price).diff().fillna(0)
        for _ in range(300):
            shift = rng.integers(21, len(expo) - 21)
            e2 = pd.Series(np.roll(expo.values, shift), index=expo.index).shift(1).fillna(0)
            x = e2 * r; sh.append(x.mean() / x.std() * np.sqrt(252) if x.std() > 0 else 0)
        strat[k]["random_timing_sharpe_mean"] = float(np.mean(sh)); strat[k]["random_timing_sharpe_p"] = float((np.array(sh) >= s["strategy"]["sharpe"]).mean())
        if k == "long_in_fear_lt25":
            plt.figure(figsize=(10, 4), dpi=110)
            plt.plot(s["equity_bh"], label="buy & hold", color="#888"); plt.plot(s["equity_strategy"], label=k, color="#d73a49")
            plt.yscale("log"); plt.legend(); plt.grid(alpha=.3); plt.title(f"{name}: {k} vs buy & hold (log scale)")
            plt.tight_layout(); plt.savefig(FIG / f"{name}_strategy.png"); plt.close()
    res["strategies"] = strat
    for k, v in strat.items():
        log.info("strategy %-30s CAGR %6.2f%% Sharpe %5.2f DD %6.1f%% (B&H Sharpe %.2f) expo %.2f  random-timing p=%.2f",
                 k, v["strategy"]["cagr_pct"], v["strategy"]["sharpe"], v["strategy"]["max_dd_pct"], v["buy_hold"]["sharpe"], v["exposure"], v["random_timing_sharpe_p"])

    # F. walk-forward models
    if run_ml:
        lo = ml.walk_forward_logit(feats, fwd, first_test_year, cols=["level"])
        sh = ml.walk_forward_logit(feats, fwd, first_test_year)
        cn = ml.walk_forward_cnn(pd.DataFrame({"level": level}), fwd, first_test_year)
        cn2 = ml.walk_forward_cnn(pd.DataFrame({"level": level, "price": price}), fwd, first_test_year)
        cnull = ml.walk_forward_cnn(pd.DataFrame({"level": level}), fwd, first_test_year, shuffle_labels=True)
        res["models"] = {"logit_level_only": lo, "logit_shape_features": sh, "cnn_index_chart": cn,
                         "cnn_index_plus_price": cn2, "cnn_shuffled_labels_null": cnull}
        for k, m in res["models"].items():
            log.info("model %-26s " + "  ".join(f"{h}d acc {m[h]['accuracy']:.3f}(base {m[h]['base_rate']:.3f}) ic {m[h]['ic']:+.3f} t{m[h]['ic_t']:+.1f}" for h in (5, 21, 63)), k)

    # figure: index with market
    fig, ax = plt.subplots(figsize=(12, 4.5), dpi=110)
    ax.plot(price.index, price.values, color="#1f6feb", lw=0.9); ax.set_yscale("log"); ax.set_ylabel("price (log)")
    ax2 = ax.twinx(); ax2.plot(level.index, level.values, color="#d73a49", lw=0.6, alpha=0.8); ax2.set_ylim(0, 100); ax2.set_ylabel("index 0-100")
    ax2.axhspan(0, 25, color="#d73a49", alpha=0.06); ax2.axhspan(75, 100, color="#2a9d3a", alpha=0.06)
    ax.set_title(f"{name}: index (red, right axis) vs market (blue, log)"); fig.tight_layout(); fig.savefig(FIG / f"{name}_overview.png"); plt.close(fig)
    # figure: bucket bars + event paths
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), dpi=110)
    x = np.arange(len(bt)); w = 0.25
    for i, h in enumerate((5, 21, 63)):
        axes[0].bar(x + (i - 1) * w, bt[f"ret_{h}_pct"] - bt[f"base_{h}_pct"], w, label=f"{h}d excess")
    axes[0].set_xticks(x); axes[0].set_xticklabels(bt.bucket, rotation=20, fontsize=8); axes[0].axhline(0, color="k", lw=.5); axes[0].legend(fontsize=8)
    axes[0].set_title("forward return minus unconditional, by bucket (%)")
    for k, c in (("into_extreme_fear", "#d73a49"), ("into_extreme_greed", "#2a9d3a")):
        e = res["events"][k]
        if e["n"]:
            axes[1].plot(range(1, 64), e["mean_path_pct"], color=c, label=f"{k} (n={e['n']})")
            axes[1].fill_between(range(1, 64), e["ci_lo_pct"], e["ci_hi_pct"], color=c, alpha=0.12)
    axes[1].axhline(0, color="k", lw=.5); axes[1].legend(fontsize=8); axes[1].set_title("mean market return after entering a zone (%, 95% bootstrap CI)")
    fig.tight_layout(); fig.savefig(FIG / f"{name}_buckets_events.png"); plt.close(fig)
    with open(OUT / f"{name}.json", "w") as f:
        json.dump(res, f, indent=2, default=str)
    return res


def main() -> None:
    mkt = market_data()
    recon = pd.read_csv(SDIR / "fear_greed_reconstructed.csv", index_col=0, parse_dates=True)
    spx = mkt["^GSPC"]
    which = sys.argv[1:] or ["recon", "vix", "crypto", "cnn", "components"]
    if "recon" in which:
        study("fg_reconstructed_spx", recon["fg_recon"], spx, first_test_year=2013)
    if "vix" in which:
        study("vix_spx", vix_to_index(mkt["^VIX"].dropna()), spx, first_test_year=1998)
    if "vix_nasdaq" in which:
        study("vix_nasdaq", vix_to_index(mkt["^VIX"].dropna()), mkt["^IXIC"], first_test_year=1998)
    if "vix_russell" in which:
        study("vix_russell", vix_to_index(mkt["^VIX"].dropna()), mkt["^RUT"], first_test_year=1998)
    if "vxn_nasdaq" in which and mkt["^VXN"].notna().sum() > 1000:
        study("vxn_nasdaq", vix_to_index(mkt["^VXN"].dropna()), mkt["^IXIC"], first_test_year=2006)
    if "rvx_russell" in which and mkt["^RVX"].notna().sum() > 1000:
        study("rvx_russell", vix_to_index(mkt["^RVX"].dropna()), mkt["^RUT"], first_test_year=2008)
    if "crypto" in which:
        study("crypto_fg_btc", crypto_fear_greed(), mkt["BTC-USD"], first_test_year=2021)
    if "cnn" in which:
        study("cnn_fg_real_1y_spx", cnn_fear_greed(), spx, first_test_year=2027, run_ml=False, n_boot=200)
    if "components" in which:
        comp = {}
        for c in ["momentum", "strength", "breadth", "volatility", "safe_haven", "junk"]:
            bt = A.bucket_table(recon[c], A.forward_returns(spx.reindex(recon.index)))
            ic = A.ic_table(pd.DataFrame({c: recon[c]}), A.forward_returns(spx.reindex(recon.index)))
            comp[c] = {"buckets": bt.round(4).to_dict("records"), "ic": ic.round(4).to_dict("records")[0]}
            log.info("component %-11s ic5 %+.3f (t %+.1f)  ic21 %+.3f (t %+.1f)  ic63 %+.3f (t %+.1f)", c,
                     comp[c]["ic"]["ic_5"], comp[c]["ic"]["t_5"], comp[c]["ic"]["ic_21"], comp[c]["ic"]["t_21"], comp[c]["ic"]["ic_63"], comp[c]["ic"]["t_63"])
        with open(OUT / "components.json", "w") as f:
            json.dump(comp, f, indent=2, default=str)


if __name__ == "__main__":
    main()
