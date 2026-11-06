"""Shared argument parsing for the scripts."""

from __future__ import annotations

import argparse
import logging

from .config import Config
from .universe import UNIVERSE


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--sectors", nargs="*", default=None, choices=list(UNIVERSE),
                   help="restrict the universe to these sectors (default: all)")
    p.add_argument("--max-tickers", type=int, default=Config.max_tickers)
    p.add_argument("--tickers", nargs="*", default=None,
                   help="explicit ticker list (overrides --sectors)")
    p.add_argument("--years", type=int, default=Config.history_years)
    p.add_argument("--ma", type=int, default=Config.ma_window, help="moving average length")
    p.add_argument("--chart-days", type=int, default=Config.chart_days)
    p.add_argument("--window", type=int, default=Config.window, help="days per training image")
    p.add_argument("--stride", type=int, default=Config.stride)
    p.add_argument("--img-size", type=int, default=Config.img_size)
    p.add_argument("--model-width", type=int, default=Config.model_width,
                   help="width of the CNN (16 = fast on CPU, 32 = bigger)")
    p.add_argument("--channels", nargs="+", default=list(Config.channels), choices=["ma", "close"],
                   help="lines drawn in each image (ma = the --ma moving average)")
    p.add_argument("--horizons", nargs="+", type=int, default=list(Config.horizons))
    p.add_argument("--label-source", default=Config.label_source, choices=["ma", "close", "close_rel"],
                   help="what the up/flat/down label measures (close_rel = relative to the market that day)")
    p.add_argument("--flat-threshold", default=Config.flat_threshold,
                   help="fixed |return| for the flat class, or 'auto' for train-set tertiles")
    p.add_argument("--path-horizon", type=int, default=Config.path_horizon)
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch-size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--val-fraction", type=float, default=Config.val_fraction_days)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--workers", type=int, default=Config.num_workers)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def config_from_args(args: argparse.Namespace) -> Config:
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S")
    cfg = Config(
        sectors=args.sectors,
        max_tickers=args.max_tickers,
        history_years=args.years,
        ma_window=args.ma,
        chart_days=args.chart_days,
        window=args.window,
        stride=args.stride,
        img_size=args.img_size,
        model_width=args.model_width,
        channels=tuple(args.channels),
        horizons=tuple(args.horizons),
        label_source=args.label_source,
        flat_threshold="auto" if str(args.flat_threshold) == "auto" else float(args.flat_threshold),
        path_horizon=args.path_horizon,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        val_fraction_days=args.val_fraction,
        seed=args.seed,
        num_workers=args.workers,
    )
    cfg.ensure_dirs()
    return cfg


def resolve_tickers(args: argparse.Namespace, cfg: Config) -> list[str]:
    from .universe import get_universe
    if args.tickers:
        return args.tickers
    return get_universe(cfg.sectors, cfg.max_tickers)
