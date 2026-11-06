#!/usr/bin/env python3
"""Stage 4 - per-stock models.

Starting from the general model of stage 3, fine-tune a copy on each stock's
own training windows and compare, on that stock's validation windows:
  general   : the shared model
  per_stock : the fine-tuned copy
  ensemble  : average of the two probability vectors (weight w)
Fine-tuned checkpoints are saved to results/perstock_<label>/<TICKER>.pt so
scripts/4_predict.py can blend them (--per-stock-dir).
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from common import RESULTS, base_config, cfg_dict, load_json, samples_for, save_json, setup_logging
from chartvision.config import Config
from chartvision.dataset import assign_labels, purge_overlap, split_at
from chartvision.train import compact, evaluate, fit, get_device, predict_samples
from walkforward import pick_best

log = setup_logging("perstock")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", nargs="+", default=["close", "ma"])
    p.add_argument("--ft-epochs", type=int, default=4)
    p.add_argument("--ft-lr", type=float, default=3e-4)
    p.add_argument("--weights", nargs="+", type=float, default=[0.3, 0.5, 0.7])
    p.add_argument("--max-stocks", type=int, default=None)
    p.add_argument("--overrides", default=None)
    args = p.parse_args()
    device = get_device()

    for label in args.labels:
        ck_path = RESULTS / f"general_{label}.pt"
        if not ck_path.exists():
            log.warning("no general model for %s - run groups.py first", label)
            continue
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        c = ck["config"]
        cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in c.items()
                        if k not in ("raw_dir", "chart_dir", "pred_dir", "model_dir")})
        cfg.stride = 1   # the per-stock set is small: use every day
        thr = {int(h): tuple(t) for h, t in ck["thresholds"].items()}
        cutoff = pd.Timestamp(ck["cutoff"])
        name = f"perstock_{label}"
        if load_json(name):
            log.info("skip %s (done)", name)
            continue
        out_dir = RESULTS / name
        out_dir.mkdir(exist_ok=True)

        from chartvision.model import ChartCNN
        general = ChartCNN(len(cfg.channels), len(cfg.horizons), cfg.path_horizon, width=cfg.model_width).to(device)
        general.load_state_dict(ck["state_dict"])
        gen_state = ck["state_dict"]

        t0 = time.time()
        samples, frames = samples_for(cfg)
        tickers = sorted(frames)[: args.max_stocks] if args.max_stocks else sorted(frames)
        by_ticker: dict[str, list] = {}
        for s in samples:
            by_ticker.setdefault(s.ticker, []).append(s)

        pooled = {v: [] for v in ["general", "per_stock"] + [f"ens_{w}" for w in args.weights]}
        per_stock_rows = []
        for i, t in enumerate(tickers):
            ss = by_ticker.get(t, [])
            tr, va = split_at(ss, cutoff)
            tr = purge_overlap(tr, cutoff, max(max(cfg.horizons), cfg.path_horizon))
            if len(tr) < 200 or len(va) < 50:
                continue
            assign_labels(tr, thr, cfg.horizons)
            assign_labels(va, thr, cfg.horizons)
            ft_cfg = Config(**{**{k: getattr(cfg, k) for k in vars(cfg)}, "batch_size": 64, "num_workers": 0})
            ck_t = out_dir / f"{t}.pt"
            if ck_t.exists():   # resume: reuse an already fine-tuned copy
                model_t = ChartCNN(len(cfg.channels), len(cfg.horizons), cfg.path_horizon, width=cfg.model_width).to(device)
                model_t.load_state_dict(torch.load(ck_t, map_location=device, weights_only=False)["state_dict"])
            else:
                model_t, _ = fit(tr, [], ft_cfg, device, init_state=gen_state, lr=args.ft_lr,
                                 epochs=args.ft_epochs, select_on="last", verbose=False)
                torch.save({"state_dict": {k: v.cpu() for k, v in model_t.state_dict().items()}, "config": cfg_dict(cfg),
                            "thresholds": ck["thresholds"], "cutoff": ck["cutoff"]}, ck_t)

            o_g = predict_samples(general, va, ft_cfg, device)
            o_t = predict_samples(model_t, va, ft_cfg, device)
            variants = {"general": o_g["probs"], "per_stock": o_t["probs"]}
            for w in args.weights:
                variants[f"ens_{w}"] = (1 - w) * o_g["probs"] + w * o_t["probs"]
            row = {"ticker": t, "n_train": len(tr), "n_val": len(va)}
            for v, probs in variants.items():
                o = dict(o_g, probs=probs, pred=probs.argmax(-1), score=probs[:, :, 2] - probs[:, :, 0])
                pooled[v].append(o)
                for hi, h in enumerate(cfg.horizons):
                    row[f"{v}_acc_{h}"] = float((o["pred"][:, hi] == o["labels"][:, hi]).mean())
            per_stock_rows.append(row)
            if (i + 1) % 25 == 0:
                log.info("%d/%d stocks  (%.1f min)", i + 1, len(tickers), (time.time() - t0) / 60)

        df = pd.DataFrame(per_stock_rows)
        df.to_csv(RESULTS / f"{name}.csv", index=False)
        summary = {}
        for v, outs in pooled.items():
            merged = {k: np.concatenate([o[k] for o in outs]) for k in ["pred", "score", "labels", "returns", "dates", "momentum"]}
            summary[v] = compact(evaluate(merged, cfg, thr))
        improved = {h: float((df[f"per_stock_acc_{h}"] > df[f"general_acc_{h}"]).mean()) for h in cfg.horizons}
        log.info("pooled accuracy by variant: %s", {v: {h: m[h]["acc"] for h in cfg.horizons} for v, m in summary.items()})
        log.info("share of stocks where per-stock beats general: %s", improved)
        save_json(name, {"name": name, "label": label, "n_stocks": len(df), "minutes": (time.time() - t0) / 60,
                         "ft_epochs": args.ft_epochs, "ft_lr": args.ft_lr, "pooled": summary,
                         "share_stocks_improved": improved})


if __name__ == "__main__":
    main()
