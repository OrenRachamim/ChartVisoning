"""Forecast the next days for every stock from its latest chart window."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .charts import plot_ma_chart, rasterize_window
from .config import CLASS_NAMES, Config
from .train import load_model

log = logging.getLogger(__name__)


def predict_all(frames: dict[str, pd.DataFrame], cfg: Config, ckpt_path: Path,
                draw_charts: bool = True, per_stock_dir: Path | None = None,
                ensemble_weight: float = 0.5) -> pd.DataFrame:
    """Score every stock with the general model. When ``per_stock_dir`` holds
    fine-tuned checkpoints (``<TICKER>.pt``) their probabilities are blended
    with weight ``ensemble_weight``."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(ckpt_path, device)
    c = ck["config"]
    horizons = tuple(c["horizons"])
    window, img_size = int(c["window"]), int(c["img_size"])
    ma_col = f"ma{c['ma_window']}"
    cols = [ma_col if ch != "close" else "Close" for ch in c["channels"]]

    tickers, imgs = [], []
    for t, df in frames.items():
        if len(df) < window or ma_col not in df.columns:
            continue
        series = df[cols].tail(window).to_numpy(dtype=np.float32)
        if not np.isfinite(series).all():
            continue
        tickers.append(t)
        imgs.append(rasterize_window(series, img_size).astype(np.float32) / 255.0)
    if not tickers:
        raise RuntimeError("no stock has enough history for a forecast window")

    x = torch.from_numpy(np.stack(imgs)).to(device)
    probs, paths = _run(model, x)

    if per_stock_dir is not None:
        for i, t in enumerate(tickers):
            p = per_stock_dir / f"{t}.pt"
            if not p.exists():
                continue
            m_t, _ = load_model(p, device)
            pr, pa = _run(m_t, x[i : i + 1])
            probs[i] = (1 - ensemble_weight) * probs[i] + ensemble_weight * pr[0]
            paths[i] = (1 - ensemble_weight) * paths[i] + ensemble_weight * pa[0]

    rows = []
    for i, t in enumerate(tickers):
        df = frames[t]
        last_ma = float(df[ma_col].iloc[-1])
        row = {"ticker": t, "as_of": df.index[-1].date(), "close": float(df["Close"].iloc[-1]), ma_col: last_ma}
        fp = {}
        for h_i, h in enumerate(horizons):
            p = probs[i, h_i]
            fp[h] = dict(zip(CLASS_NAMES, map(float, p)))
            best = int(p.argmax())
            row[f"pred_{h}d"] = CLASS_NAMES[best]
            row[f"conf_{h}d"] = float(p[best])
            row[f"score_{h}d"] = float(p[2] - p[0])
            for c_i, cn in enumerate(CLASS_NAMES):
                row[f"p_{cn}_{h}d"] = float(p[c_i])
        proj = last_ma * (1.0 + paths[i])
        row["proj_ma_path"] = " ".join(f"{v:.2f}" for v in proj)
        row[f"proj_ma_{len(proj)}d_pct"] = float(paths[i][-1] * 100)
        rows.append(row)
        if draw_charts:
            plot_ma_chart(df, t, cfg.pred_dir / f"{t}_forecast.png", ma_col=ma_col,
                          days=cfg.chart_days, forecast=proj, forecast_probs=fp)

    out = pd.DataFrame(rows)
    h_main = horizons[-1]
    return out.sort_values(f"score_{h_main}d", ascending=False).reset_index(drop=True)


@torch.no_grad()
def _run(model, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    probs, paths = [], []
    for i in range(0, len(x), 512):
        logits, path = model(x[i : i + 512])
        probs.append(torch.softmax(logits, -1).cpu().numpy())
        paths.append(path.cpu().numpy())
    return np.concatenate(probs), np.concatenate(paths)
