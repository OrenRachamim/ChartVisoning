#!/usr/bin/env python3
"""Stage 2 - walk-forward validation of the best sweep configurations on the
full universe (3 expanding folds). Confirms that stage-1 numbers are not an
artefact of one validation period."""

from __future__ import annotations

import argparse
import time

import pandas as pd

from common import RESULTS, base_config, cfg_dict, load_json, samples_for, save_json, setup_logging
from chartvision.train import get_device, walk_forward

log = setup_logging("walkforward")


def pick_best(label: str, metric: str = "acc", horizons=(2, 3, 5, 10)) -> tuple[str, dict]:
    """Best sweep variant for a label source by mean ``metric`` over horizons."""
    best, best_v, best_ov = None, -1e9, {}
    for path in RESULTS.glob(f"sweep_{label}_*.json"):
        r = load_json(path.stem)
        vals = [r["summary"][str(h)][metric] for h in horizons if str(h) in r["summary"]]
        v = sum(vals) / len(vals)
        if v > best_v:
            best, best_v, best_ov = r["name"], v, r["overrides"]
    return best, best_ov


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", nargs="+", default=["close", "ma"])
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--metric", default="acc", help="sweep metric used to pick the config")
    p.add_argument("--overrides", default=None, help="JSON dict to force a config instead of the sweep's best")
    args = p.parse_args()

    device = get_device()
    for label in args.labels:
        if args.overrides:
            import json
            name, ov = f"{label}_custom", json.loads(args.overrides)
        else:
            name, ov = pick_best(label, args.metric)
            if name is None:
                log.warning("no sweep results for label %s", label)
                continue
        ov = {**ov, "label_source": label}
        if load_json(f"walkforward_{name}"):
            log.info("skip %s (done)", name)
            continue
        cfg = base_config(**ov)
        cfg.epochs = args.epochs
        t0 = time.time()
        samples, frames = samples_for(cfg)
        log.info("=== walk-forward %s: %s  (%d windows, %d stocks)", name, ov, len(samples), len(frames))
        res = walk_forward(samples, cfg, n_folds=args.folds, device=device, verbose=True,
                           cache_dir=RESULTS, name=f"wf_{name}")
        payload = {"name": name, "overrides": ov, "config": cfg_dict(cfg), "n_stocks": len(frames),
                   "n_windows": len(samples), "minutes": (time.time() - t0) / 60, **res}
        save_json(f"walkforward_{name}", payload)
        log.info("pooled %s: %s", name, res["pooled"])

    rows = []
    for path in sorted(RESULTS.glob("walkforward_*.json")):
        r = load_json(path.stem)
        for h, m in r["pooled"].items():
            rows.append({"run": r["name"], "horizon": int(h), **m})
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(RESULTS / "walkforward_summary.csv", index=False)
        pd.set_option("display.width", 250)
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
