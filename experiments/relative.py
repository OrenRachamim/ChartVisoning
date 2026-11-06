#!/usr/bin/env python3
"""Test B - market-neutral labels (close_rel): the stock's forward return minus
the same-day mean forward return of all stocks. Runs the base configuration on
the stratified subset, then a 3-fold walk-forward on the full universe, then the
confidence analysis on the out-of-sample predictions."""

from __future__ import annotations

import argparse
import time

from common import RESULTS, base_config, cfg_dict, load_json, samples_for, save_json, setup_logging, subset_tickers
from chartvision.train import get_device, walk_forward
import sweep
import confidence

log = setup_logging("relative")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--skip-subset", action="store_true")
    args = p.parse_args()
    device = get_device()

    if not args.skip_subset:
        t = subset_tickers(4)
        sweep.run_one("closerel_base", {"label_source": "close_rel"}, t, device)
        sweep.run_one("closerel_win120_img64", {"label_source": "close_rel", "window": 120, "img_size": 64}, t, device)

    name = "close_rel_base"
    if not load_json(f"walkforward_{name}"):
        cfg = base_config(label_source="close_rel")
        cfg.epochs = args.epochs
        t0 = time.time()
        samples, frames = samples_for(cfg)
        log.info("=== walk-forward %s (%d windows, %d stocks)", name, len(samples), len(frames))
        res = walk_forward(samples, cfg, n_folds=3, device=device, verbose=True, cache_dir=RESULTS, name=f"wf_{name}")
        save_json(f"walkforward_{name}", {"name": name, "overrides": {"label_source": "close_rel"}, "config": cfg_dict(cfg),
                                          "n_stocks": len(frames), "n_windows": len(samples),
                                          "minutes": (time.time() - t0) / 60, **res})
        log.info("pooled %s: %s", name, res["pooled"])

    out = confidence.load_oos(name)
    res = confidence.analyse(out, (1, 2, 3, 5, 10))
    save_json(f"confidence_{name}", {"name": name, "n": int(len(out["labels"])), "horizons": res})


if __name__ == "__main__":
    main()
