"""Central configuration. Every script reads its defaults from here and lets
you override them on the command line."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # one CSV per ticker
CHART_DIR = DATA_DIR / "charts"       # 500-day MA chart per ticker
PRED_DIR = DATA_DIR / "predictions"   # charts with the forecast drawn on them
MODEL_DIR = ROOT / "models"


@dataclass
class Config:
    # ---- universe -----------------------------------------------------
    # Sectors to keep (None = all). Picking one or two sectors gives a set of
    # "relatively similar" stocks, which is what the research asks for.
    sectors: list[str] | None = None
    max_tickers: int = 300

    # ---- data ---------------------------------------------------------
    history_years: int = 5          # how much history to download
    ma_window: int = 10             # the moving average drawn on the chart
    chart_days: int = 500           # length of the per-stock chart

    # ---- dataset / images ----------------------------------------------
    window: int = 60                # days shown in each training image
    stride: int = 3                 # step between consecutive windows
    img_size: int = 48              # images are img_size x img_size
    model_width: int = 16           # channels of the first conv block (CPU friendly)
    channels: tuple[str, ...] = ("ma", "close")   # lines drawn, one per channel
    horizons: tuple[int, ...] = (5, 10)           # forecast horizons (days)
    label_source: str = "ma"        # "ma" = movement of the chart line, "close" = tradable price
    # |return| below this -> "flat". A float is a fixed threshold; "auto" uses
    # the train-set tertiles so the 3 classes are balanced for every horizon.
    flat_threshold: float | str = 0.01
    path_horizon: int = 10          # length of the projected path (regression head)

    # ---- training -----------------------------------------------------
    val_fraction_days: float = 0.2  # last 20% of the calendar -> validation
    epochs: int = 12
    batch_size: int = 256
    lr: float = 1e-3
    seed: int = 42
    num_workers: int = 2

    # ---- paths ----------------------------------------------------------
    raw_dir: Path = field(default=RAW_DIR)
    chart_dir: Path = field(default=CHART_DIR)
    pred_dir: Path = field(default=PRED_DIR)
    model_dir: Path = field(default=MODEL_DIR)

    @property
    def ma_col(self) -> str:
        return f"ma{self.ma_window}"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.chart_dir, self.pred_dir, self.model_dir):
            d.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = ("down", "flat", "up")
