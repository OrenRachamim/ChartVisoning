#!/usr/bin/env python3
"""Stage 3 - does grouping stocks help?

For the chosen configuration and one time split:
  general      : one model on all stocks, evaluated per sector
  sector_only  : a model trained on that sector alone
  general+ft   : the general model fine-tuned on the sector (low LR, few epochs)
Also saves the general model (results/general_<label>.pt) for stage 4.
"""

from __future__ import annotations

import argparse
import json
import time

import pandas as pd
import torch

from common import RESULTS, base_config, cfg_dict, load_json, samples_for, save_json, sector_map, setup_logging
from chartvision.dataset import assign_labels, compute_thresholds, purge_overlap, split_at, time_split
from chartvision.train import compact, evaluate, fit, get_device, predict_samples
from walkforward import pick_best

log = setup_logging("groups")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", nargs="+", default=["close", "ma"])
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--ft-epochs", type=int, default=3)
    p.add_argument("--ft-lr", type=float, default=2e-4)
    p.add_argument("--min-stocks", type=int, default=20)
    p.add_argument("--overrides", default=None)
    args = p.parse_args()
    device = get_device()

    for label in args.labels:
        name, ov = (f"{label}_custom", json.loads(args.overrides)) if args.overrides else pick_best(label)
        if name is None:
            continue
        if load_json(f"groups_{name}"):
            log.info("skip %s (done)", name)
            continue
        cfg = base_config(**{**ov, "label_source": label})
        cfg.epochs = args.epochs
        t0 = time.time()
        samples, frames = samples_for(cfg)
        _, _, cutoff = time_split(samples, cfg.val_fraction_days)
        train_s, val_s = split_at(samples, cutoff)
        train_s = purge_overlap(train_s, cutoff, max(max(cfg.horizons), cfg.path_horizon))
        thr = compute_thresholds(train_s, cfg)
        assign_labels(train_s, thr, cfg.horizons)
        assign_labels(val_s, thr, cfg.horizons)
        log.info("=== groups %s (%d train / %d val, cutoff %s)", name, len(train_s), len(val_s), cutoff.date())

        general, _ = fit(train_s, val_s, cfg, device, select_on="last", verbose=True)
        gen_state = {k: v.detach().cpu().clone() for k, v in general.state_dict().items()}
        torch.save({"state_dict": gen_state, "config": cfg_dict(cfg), "thresholds": {int(h): list(t) for h, t in thr.items()},
                    "cutoff": str(cutoff.date())}, RESULTS / f"general_{label}.pt")

        sectors = sector_map(list(frames))
        rows, detail = [], {}
        for sector, tickers in sorted(sectors.items()):
            if len(tickers) < args.min_stocks:
                continue
            tr = [s for s in train_s if s.ticker in tickers]
            va = [s for s in val_s if s.ticker in tickers]
            if len(va) < 200:
                continue
            res = {}
            res["general"] = compact(evaluate(predict_samples(general, va, cfg, device), cfg, thr))
            sector_model, _ = fit(tr, va, cfg, device, select_on="last", verbose=False)
            res["sector_only"] = compact(evaluate(predict_samples(sector_model, va, cfg, device), cfg, thr))
            ft, _ = fit(tr, va, cfg, device, init_state=gen_state, lr=args.ft_lr, epochs=args.ft_epochs,
                        select_on="last", verbose=False)
            res["general_ft"] = compact(evaluate(predict_samples(ft, va, cfg, device), cfg, thr))
            detail[sector] = {"n_stocks": len(tickers), "n_train": len(tr), "n_val": len(va), **res}
            for variant, table in res.items():
                for h, m in table.items():
                    rows.append({"sector": sector, "variant": variant, "horizon": int(h), "n_val": len(va), **m})
            log.info("%-24s %s", sector, {v: {h: res[v][h]["acc"] for h in (2, 5, 10)} for v in res})

        df = pd.DataFrame(rows)
        df.to_csv(RESULTS / f"groups_{name}.csv", index=False)
        agg = df.groupby(["variant", "horizon"])[["acc", "ic", "spread_t"]].mean().round(3)
        log.info("mean over sectors:\n%s", agg.to_string())
        save_json(f"groups_{name}", {"name": name, "overrides": ov, "cutoff": str(cutoff.date()),
                                     "minutes": (time.time() - t0) / 60, "sectors": detail,
                                     "mean_over_sectors": {f"{v}_{h}": r for (v, h), r in agg.to_dict("index").items()}})


if __name__ == "__main__":
    main()
