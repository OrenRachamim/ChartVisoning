#!/usr/bin/env python3
"""Stage 1 - one-factor-at-a-time parameter sweep on a stratified subset.

Every run trains one model with a classification head per horizon
(1, 2, 3, 5, 10 days), so the horizon comparison comes for free. Runs are
resumable: finished runs (results/sweep_<name>.json) are skipped.
"""

from __future__ import annotations

import argparse
import time

import pandas as pd

from common import (RESULTS, base_config, cfg_dict, load_json, samples_for, save_json, setup_logging,
                    subset_tickers)
from chartvision.dataset import time_split
from chartvision.train import compact, get_device, run_split

log = setup_logging("sweep")

# name -> overrides on top of base_config(). "label" variants are run for both
# label sources because the best settings may differ.
VARIANTS = {
    "base":        {},
    "ma5":         {"ma_window": 5},
    "ma20":        {"ma_window": 20},
    "ch_ma_only":  {"channels": ("ma",)},
    "ch_close_only": {"channels": ("close",)},
    "win30":       {"window": 30},
    "win120":      {"window": 120},
    "img32":       {"img_size": 32},
    "img64":       {"img_size": 64},
    "width32":     {"model_width": 32},
    "win120_img64": {"window": 120, "img_size": 64},
}


def run_one(name: str, overrides: dict, tickers: list[str], device) -> dict:
    existing = load_json(f"sweep_{name}")
    if existing:
        log.info("skip %s (done)", name)
        return existing
    cfg = base_config(**overrides)
    t0 = time.time()
    samples, frames = samples_for(cfg, tickers)
    _, _, cutoff = time_split(samples, cfg.val_fraction_days)
    log.info("=== %s: %s  (%d windows from %d stocks)", name, overrides, len(samples), len(frames))
    _, report, _, _ = run_split(samples, cfg, cutoff, device=device, select_on="last", verbose=True)
    payload = {"name": name, "overrides": overrides, "config": cfg_dict(cfg), "n_stocks": len(frames),
               "n_windows": len(samples), "minutes": (time.time() - t0) / 60,
               "summary": compact(report), "report": {k: v for k, v in report.items() if k != "history"},
               "history": report["history"]}
    save_json(f"sweep_{name}", payload)
    log.info("%s done in %.1f min: %s", name, payload["minutes"], payload["summary"])
    return payload


def summary_table() -> pd.DataFrame:
    rows = []
    for path in sorted(RESULTS.glob("sweep_*.json")):
        r = load_json(path.stem)
        for h, m in r["summary"].items():
            rows.append({"run": r["name"], "horizon": int(h), **m, "minutes": round(r["minutes"], 1)})
    df = pd.DataFrame(rows)
    if len(df):
        df.to_csv(RESULTS / "sweep_summary.csv", index=False)
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--every", type=int, default=4, help="take every n-th ticker of the universe")
    p.add_argument("--only", nargs="*", default=None, help="run only these variant names")
    p.add_argument("--labels", nargs="+", default=["ma", "close"], choices=["ma", "close"])
    args = p.parse_args()

    tickers = subset_tickers(args.every)
    device = get_device()
    for label in args.labels:
        for name, ov in VARIANTS.items():
            if args.only and name not in args.only:
                continue
            run_one(f"{label}_{name}", {**ov, "label_source": label}, tickers, device)
    df = summary_table()
    pd.set_option("display.width", 250)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
