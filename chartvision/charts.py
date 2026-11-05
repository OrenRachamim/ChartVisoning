"""Chart rendering.

Two renderers live here:

* :func:`plot_ma_chart` - the human-facing chart: the last N (500) days of the
  MA10 line (plus close) for one stock, saved as a PNG with matplotlib. It can
  also draw a forecast path on top of the history.
* :func:`rasterize_window` - the machine-facing renderer: turns a small window
  of the series into a tiny grayscale image (one channel per line). This is
  what the CNN actually "sees". It is written with PIL so tens of thousands of
  windows can be rendered on the fly while training.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --------------------------------------------------------------------------
# human-facing chart
# --------------------------------------------------------------------------
def plot_ma_chart(
    df: pd.DataFrame,
    ticker: str,
    out_path: Path,
    ma_col: str = "ma10",
    days: int = 500,
    forecast: np.ndarray | None = None,
    forecast_probs: dict[int, dict[str, float]] | None = None,
) -> Path:
    """Save a PNG with the last ``days`` days of ``ma_col`` for ``ticker``.

    ``forecast`` (optional) is an array of future MA values, drawn as a dashed
    line starting from the last historical point. ``forecast_probs`` maps a
    horizon (5, 10) to {"down":p, "flat":p, "up":p} and is printed in the
    title.
    """
    tail = df.tail(days)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=110)
    ax.plot(tail.index, tail["Close"], color="#9aa5b1", lw=0.8, label="Close")
    ax.plot(tail.index, tail[ma_col], color="#1f6feb", lw=1.6, label=ma_col.upper())

    if forecast is not None and len(forecast):
        last_date = tail.index[-1]
        future_idx = pd.bdate_range(last_date, periods=len(forecast) + 1)[1:]
        xs = [last_date, *future_idx]
        ys = [tail[ma_col].iloc[-1], *forecast]
        ax.plot(xs, ys, color="#d73a49", lw=1.8, ls="--", label="forecast")
        ax.axvline(last_date, color="#d73a49", lw=0.6, alpha=0.5)

    title = f"{ticker}  -  last {len(tail)} trading days, {ma_col.upper()}"
    if forecast_probs:
        parts = []
        for h, p in sorted(forecast_probs.items()):
            best = max(p, key=p.get)
            parts.append(f"{h}d: {best} ({p[best]:.0%})")
        title += "\nforecast  " + "   |   ".join(parts)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", fontsize=9)
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# machine-facing renderer
# --------------------------------------------------------------------------
def rasterize_window(series: np.ndarray, size: int = 64, line_width: int = 1) -> np.ndarray:
    """Render ``series`` (shape ``[T, C]``) as a ``[C, size, size]`` uint8 image.

    All channels share one vertical scale (the min/max over the whole window)
    so the relative position of e.g. close vs. MA10 is preserved. Each channel
    is a separate polyline drawn with PIL.
    """
    series = np.asarray(series, dtype=np.float32)
    if series.ndim == 1:
        series = series[:, None]
    T, C = series.shape

    lo = np.nanmin(series)
    hi = np.nanmax(series)
    span = hi - lo if hi > lo else 1.0
    norm = (series - lo) / span  # 0..1

    xs = np.linspace(0, size - 1, T)
    img = np.zeros((C, size, size), dtype=np.uint8)
    for c in range(C):
        canvas = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(canvas)
        ys = (size - 1) - norm[:, c] * (size - 1)
        pts = [(float(x), float(y)) for x, y in zip(xs, ys) if np.isfinite(y)]
        if len(pts) >= 2:
            draw.line(pts, fill=255, width=line_width)
        elif pts:
            draw.point(pts, fill=255)
        img[c] = np.asarray(canvas, dtype=np.uint8)
    return img


def save_window_preview(series: np.ndarray, out_path: Path, size: int = 64, scale: int = 4) -> None:
    """Dump one rasterized window as a PNG (handy to inspect what the model sees)."""
    img = rasterize_window(series, size)
    if img.shape[0] == 1:
        rgb = np.repeat(img, 3, axis=0)
    elif img.shape[0] == 2:
        rgb = np.stack([img[1], img[0], np.zeros_like(img[0])])
    else:
        rgb = img[:3]
    pil = Image.fromarray(np.transpose(rgb, (1, 2, 0)))
    pil = pil.resize((size * scale, size * scale), Image.NEAREST)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)
