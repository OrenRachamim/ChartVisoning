"""Turn price histories into labelled chart windows.

For every stock and every window end date ``t`` we keep:

* the window ``series[t-window+1 : t+1]`` for each drawn channel (ma10, close)
* classification labels per horizon h:  down / flat / up of ``label_source``
  between day t and day t+h
* a regression target: the next ``path_horizon`` MA10 values expressed as a
  percentage change from MA10 at day t (so the model can draw a projected
  path)

Images are rasterized lazily in ``__getitem__`` so memory stays small.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .charts import rasterize_window
from .config import CLASS_NAMES, Config


@dataclass
class Sample:
    ticker: str
    end_date: pd.Timestamp
    series: np.ndarray            # [window, C] raw values
    labels: np.ndarray            # [len(horizons)] int in {0,1,2}
    future_returns: np.ndarray    # [len(horizons)] float (for reporting)
    path: np.ndarray              # [path_horizon] float, pct change of MA from day t


def _classify(ret: float, thr: float) -> int:
    if ret > thr:
        return 2      # up
    if ret < -thr:
        return 0      # down
    return 1          # flat


def build_samples(frames: dict[str, pd.DataFrame], cfg: Config) -> list[Sample]:
    """Create windows from every frame in ``frames`` (ticker -> df with ma col)."""
    ma_col = f"ma{cfg.ma_window}"
    max_h = max(max(cfg.horizons), cfg.path_horizon)
    samples: list[Sample] = []

    chan_cols = {"ma10": ma_col, "ma": ma_col, "close": "Close"}
    cols = [chan_cols[c] for c in cfg.channels]

    for ticker, df in frames.items():
        if ma_col not in df.columns or len(df) < cfg.window + max_h + 1:
            continue
        values = df[cols].to_numpy(dtype=np.float32)
        label_src = df[ma_col if cfg.label_source == "ma10" else "Close"].to_numpy(dtype=np.float32)
        ma = df[ma_col].to_numpy(dtype=np.float32)
        dates = df.index

        last_end = len(df) - max_h - 1
        for end in range(cfg.window - 1, last_end + 1, cfg.stride):
            window = values[end - cfg.window + 1 : end + 1]
            if not np.isfinite(window).all():
                continue
            rets = np.array([label_src[end + h] / label_src[end] - 1.0 for h in cfg.horizons],
                            dtype=np.float32)
            labels = np.array([_classify(r, cfg.flat_threshold) for r in rets], dtype=np.int64)
            path = ma[end + 1 : end + 1 + cfg.path_horizon] / ma[end] - 1.0
            samples.append(Sample(ticker, dates[end], window, labels, rets, path.astype(np.float32)))
    return samples


def time_split(samples: list[Sample], val_fraction_days: float) -> tuple[list[Sample], list[Sample], pd.Timestamp]:
    """Split by calendar: the most recent ``val_fraction_days`` of dates go to
    validation. A gap of ``max horizon`` is not needed because windows are
    labelled with information that ends before the cutoff only on the train
    side; validation windows may *start* before it, which is fine."""
    dates = sorted({s.end_date for s in samples})
    cut_idx = int(len(dates) * (1 - val_fraction_days))
    cutoff = dates[cut_idx]
    train = [s for s in samples if s.end_date < cutoff]
    val = [s for s in samples if s.end_date >= cutoff]
    return train, val, cutoff


def purge_overlap(train: list[Sample], cutoff: pd.Timestamp, max_horizon: int) -> list[Sample]:
    """Drop training windows whose *labels* peek past the cutoff (leakage guard)."""
    gap = pd.tseries.offsets.BDay(max_horizon)
    limit = cutoff - gap
    return [s for s in train if s.end_date <= limit]


class ChartWindowDataset(Dataset):
    def __init__(self, samples: list[Sample], cfg: Config):
        self.samples = samples
        self.size = cfg.img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = rasterize_window(s.series, self.size).astype(np.float32) / 255.0
        return (
            torch.from_numpy(img),
            torch.from_numpy(s.labels),
            torch.from_numpy(s.path),
        )


def class_weights(samples: list[Sample], n_horizons: int) -> torch.Tensor:
    """Inverse-frequency weights per horizon, shape [n_horizons, 3]."""
    w = np.ones((n_horizons, len(CLASS_NAMES)), dtype=np.float32)
    for h in range(n_horizons):
        counts = np.bincount([s.labels[h] for s in samples], minlength=len(CLASS_NAMES)).astype(np.float32)
        counts[counts == 0] = 1
        w[h] = counts.sum() / (len(CLASS_NAMES) * counts)
    return torch.from_numpy(w)


def label_distribution(samples: list[Sample], horizons: tuple[int, ...]) -> dict[int, dict[str, float]]:
    out = {}
    for i, h in enumerate(horizons):
        counts = np.bincount([s.labels[i] for s in samples], minlength=3)
        out[h] = {n: float(c) / max(len(samples), 1) for n, c in zip(CLASS_NAMES, counts)}
    return out
