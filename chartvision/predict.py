"""Forecast the next 5/10 days for every stock from its latest chart window."""

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
                draw_charts: bool = True) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ck = load_model(ckpt_path, device)
    horizons = tuple(ck["config"]["horizons"])
    window = int(ck["config"]["window"])
    img_size = int(ck["config"]["img_size"])
    ma_col = f"ma{cfg.ma_window}"
    chan_cols = {"ma10": ma_col, "ma": ma_col, "close": "Close"}
    cols = [chan_cols[c] for c in ck["config"]["channels"]]

    rows = []
    tickers, imgs = [], []
    for t, df in frames.items():
        if len(df) < window:
            continue
        series = df[cols].tail(window).to_numpy(dtype=np.float32)
        if not np.isfinite(series).all():
            continue
        tickers.append(t)
        imgs.append(rasterize_window(series, img_size).astype(np.float32) / 255.0)
    if not tickers:
        raise RuntimeError("no stock has enough history for a forecast window")

    x = torch.from_numpy(np.stack(imgs)).to(device)
    with torch.no_grad():
        probs_all, paths_all = [], []
        for i in range(0, len(x), 512):
            logits, path = model(x[i : i + 512])
            probs_all.append(torch.softmax(logits, -1).cpu().numpy())
            paths_all.append(path.cpu().numpy())
    probs = np.concatenate(probs_all)   # [N, H, 3]
    paths = np.concatenate(paths_all)   # [N, P]

    for i, t in enumerate(tickers):
        df = frames[t]
        last_ma = float(df[ma_col].iloc[-1])
        row = {"ticker": t, "as_of": df.index[-1].date(), "close": float(df["Close"].iloc[-1]),
               ma_col: last_ma}
        fp = {}
        for h_i, h in enumerate(horizons):
            p = probs[i, h_i]
            fp[h] = dict(zip(CLASS_NAMES, map(float, p)))
            best = int(p.argmax())
            row[f"pred_{h}d"] = CLASS_NAMES[best]
            row[f"conf_{h}d"] = float(p[best])
            for c_i, c in enumerate(CLASS_NAMES):
                row[f"p_{c}_{h}d"] = float(p[c_i])
        proj = last_ma * (1.0 + paths[i])
        row["proj_ma_path"] = " ".join(f"{v:.2f}" for v in proj)
        row[f"proj_ma_{len(proj)}d_pct"] = float(paths[i][-1] * 100)
        rows.append(row)

        if draw_charts:
            plot_ma_chart(df, t, cfg.pred_dir / f"{t}_forecast.png", ma_col=ma_col,
                          days=cfg.chart_days, forecast=proj, forecast_probs=fp)

    out = pd.DataFrame(rows)
    h_main = horizons[-1]
    out = out.sort_values(f"p_up_{h_main}d", ascending=False).reset_index(drop=True)
    return out
