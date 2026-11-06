"""Turn price histories into labelled chart windows.

For every stock and every window end date ``t`` we keep:

* the window ``series[t-window+1 : t+1]`` for each drawn channel (ma, close)
* the realised forward return of ``label_source`` for every horizon h
  (day t -> day t+h); class labels are derived from these with
  :func:`assign_labels` once the thresholds are known
* a regression target: the next ``path_horizon`` MA values expressed as a
  percentage change from the MA at day t (so the model can draw a path)
* ``momentum``: the chart line's change over the last 5 days of the window,
  used by the momentum baseline

Images are rasterized lazily in ``__getitem__`` so memory stays small.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .charts import rasterize_window
from .config import CLASS_NAMES, Config

Thresholds = dict[int, tuple[float, float]]   # horizon -> (low, high)


@dataclass
class Sample:
    ticker: str
    end_date: pd.Timestamp
    series: np.ndarray            # [window, C] raw values
    future_returns: np.ndarray    # [len(horizons)] float
    path: np.ndarray              # [path_horizon] float, pct change of MA from day t
    momentum: float
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))


def channel_columns(cfg: Config) -> list[str]:
    mapping = {"ma": cfg.ma_col, "ma10": cfg.ma_col, "ma5": cfg.ma_col, "close": "Close"}
    return [mapping[c] for c in cfg.channels]


def build_samples(frames: dict[str, pd.DataFrame], cfg: Config) -> list[Sample]:
    """Create windows from every frame in ``frames`` (ticker -> df with ma col)."""
    ma_col = cfg.ma_col
    max_h = max(max(cfg.horizons), cfg.path_horizon)
    cols = channel_columns(cfg)
    samples: list[Sample] = []

    for ticker, df in frames.items():
        if ma_col not in df.columns or len(df) < cfg.window + max_h + 1:
            continue
        values = df[cols].to_numpy(dtype=np.float32)
        label_src = df[ma_col if cfg.label_source.startswith("ma") else "Close"].to_numpy(dtype=np.float32)
        ma = df[ma_col].to_numpy(dtype=np.float32)
        dates = df.index

        last_end = len(df) - max_h - 1
        for end in range(cfg.window - 1, last_end + 1, cfg.stride):
            window = values[end - cfg.window + 1 : end + 1]
            if not np.isfinite(window).all():
                continue
            rets = np.array([label_src[end + h] / label_src[end] - 1.0 for h in cfg.horizons],
                            dtype=np.float32)
            path = ma[end + 1 : end + 1 + cfg.path_horizon] / ma[end] - 1.0
            mom = float(ma[end] / ma[end - 5] - 1.0) if end >= 5 else 0.0
            samples.append(Sample(ticker, dates[end], window, rets, path.astype(np.float32), mom))
    if cfg.label_source.endswith("_rel"):
        demean_by_date(samples)
    return samples


def demean_by_date(samples: list[Sample]) -> None:
    """Market-neutral labels: subtract, per window end date and horizon, the
    mean forward return of all stocks on that date. What is left is the stock's
    move relative to the cross-section, which removes the common market noise."""
    by_date: dict[pd.Timestamp, list[Sample]] = {}
    for s in samples:
        by_date.setdefault(s.end_date, []).append(s)
    for group in by_date.values():
        if len(group) < 5:
            continue
        mean = np.mean([s.future_returns for s in group], axis=0)
        for s in group:
            s.future_returns = (s.future_returns - mean).astype(np.float32)


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
def compute_thresholds(train: list[Sample], cfg: Config) -> Thresholds:
    """Fixed symmetric threshold, or train-set tertiles when ``flat_threshold``
    is ``"auto"`` (balanced classes for every horizon)."""
    thr: Thresholds = {}
    for i, h in enumerate(cfg.horizons):
        if cfg.flat_threshold == "auto":
            r = np.array([s.future_returns[i] for s in train])
            lo, hi = np.quantile(r, [1 / 3, 2 / 3])
            thr[h] = (float(lo), float(hi))
        else:
            t = float(cfg.flat_threshold)
            thr[h] = (-t, t)
    return thr


def assign_labels(samples: list[Sample], thr: Thresholds, horizons: tuple[int, ...]) -> None:
    los = np.array([thr[h][0] for h in horizons], dtype=np.float32)
    his = np.array([thr[h][1] for h in horizons], dtype=np.float32)
    for s in samples:
        r = s.future_returns
        s.labels = np.where(r > his, 2, np.where(r < los, 0, 1)).astype(np.int64)


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def time_split(samples: list[Sample], val_fraction_days: float) -> tuple[list[Sample], list[Sample], pd.Timestamp]:
    """Split by calendar: the most recent ``val_fraction_days`` of dates go to
    validation."""
    dates = sorted({s.end_date for s in samples})
    cut_idx = int(len(dates) * (1 - val_fraction_days))
    cutoff = dates[cut_idx]
    return split_at(samples, cutoff) + (cutoff,)


def split_at(samples: list[Sample], cutoff: pd.Timestamp, end: pd.Timestamp | None = None):
    train = [s for s in samples if s.end_date < cutoff]
    val = [s for s in samples if s.end_date >= cutoff and (end is None or s.end_date < end)]
    return train, val


def purge_overlap(train: list[Sample], cutoff: pd.Timestamp, max_horizon: int) -> list[Sample]:
    """Drop training windows whose *labels* peek past the cutoff (leakage guard)."""
    limit = cutoff - pd.tseries.offsets.BDay(max_horizon)
    return [s for s in train if s.end_date <= limit]


def walk_forward_folds(samples: list[Sample], n_folds: int, min_train_fraction: float = 0.4):
    """Expanding-window folds: the calendar after ``min_train_fraction`` is cut
    into ``n_folds`` consecutive validation blocks; each fold trains on
    everything before its block."""
    dates = sorted({s.end_date for s in samples})
    start = int(len(dates) * min_train_fraction)
    edges = np.linspace(start, len(dates) - 1, n_folds + 1).astype(int)
    folds = []
    for k in range(n_folds):
        cutoff, end = dates[edges[k]], dates[edges[k + 1]]
        if k == n_folds - 1:
            end = None
        folds.append((cutoff, end))
    return folds


# --------------------------------------------------------------------------
# torch dataset / helpers
# --------------------------------------------------------------------------
class ChartWindowDataset(Dataset):
    def __init__(self, samples: list[Sample], cfg: Config):
        self.samples = samples
        self.size = cfg.img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = rasterize_window(s.series, self.size).astype(np.float32) / 255.0
        return torch.from_numpy(img), torch.from_numpy(s.labels), torch.from_numpy(s.path)


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
        out[h] = {n: round(float(c) / max(len(samples), 1), 3) for n, c in zip(CLASS_NAMES, counts)}
    return out


def flat_features(samples: list[Sample]) -> np.ndarray:
    """Window values normalised per window (same scale as the image) and
    flattened - the input of the non-visual logistic-regression baseline."""
    X = np.stack([s.series for s in samples]).astype(np.float32)   # [N, T, C]
    lo = X.min(axis=(1, 2), keepdims=True)
    hi = X.max(axis=(1, 2), keepdims=True)
    X = (X - lo) / np.where(hi > lo, hi - lo, 1.0)
    return X.reshape(len(samples), -1)
