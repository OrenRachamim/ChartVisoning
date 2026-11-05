"""Shared helpers for the experiment scripts."""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chartvision.config import Config  # noqa: E402
from chartvision.data import add_indicators, load_all  # noqa: E402
from chartvision.dataset import build_samples  # noqa: E402
from chartvision.universe import get_universe, sector_of  # noqa: E402

RESULTS = ROOT / "experiments" / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

SWEEP_HORIZONS = (1, 2, 3, 5, 10)


def setup_logging(name: str) -> logging.Logger:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout),
                                                     logging.FileHandler(RESULTS / f"{name}.log")])
    for noisy in ("matplotlib", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger(name)


def base_config(**overrides) -> Config:
    cfg = Config(horizons=SWEEP_HORIZONS, flat_threshold="auto", path_horizon=10, stride=3,
                 epochs=5, num_workers=3)
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def cfg_dict(cfg: Config) -> dict:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(cfg).items()}


def subset_tickers(every: int = 4) -> list[str]:
    """A stratified subset: every n-th ticker of the universe (all sectors)."""
    return get_universe()[::every]


def load_frames(cfg: Config, tickers: list[str] | None = None) -> dict:
    tickers = tickers if tickers is not None else get_universe()
    frames = load_all(cfg.raw_dir, tickers)
    return {t: add_indicators(df, cfg.ma_window) for t, df in frames.items()}


def samples_for(cfg: Config, tickers: list[str] | None = None):
    frames = load_frames(cfg, tickers)
    return build_samples(frames, cfg), frames


def save_json(name: str, payload: dict) -> Path:
    path = RESULTS / f"{name}.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    return path


def load_json(name: str) -> dict | None:
    path = RESULTS / f"{name}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def sector_map(tickers: list[str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for t in tickers:
        out.setdefault(sector_of(t), []).append(t)
    return out
