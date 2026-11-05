"""Download and cache daily OHLCV history for US stocks via yfinance."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _chunks(seq: list[str], n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def download_history(
    tickers: list[str],
    raw_dir: Path,
    years: int = 5,
    chunk_size: int = 50,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Download ``years`` of daily, dividend/split adjusted bars for every
    ticker and store one CSV per ticker in ``raw_dir``.

    Tickers that are already cached are reused unless ``refresh`` is True.
    Returns a mapping ticker -> DataFrame (indexed by date).
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, pd.DataFrame] = {}

    todo: list[str] = []
    for t in tickers:
        path = raw_dir / f"{t}.csv"
        if path.exists() and not refresh:
            df = load_cached(path)
            if df is not None:
                result[t] = df
                continue
        todo.append(t)

    log.info("%d tickers cached, %d to download", len(result), len(todo))

    for batch in _chunks(todo, chunk_size):
        try:
            data = yf.download(
                batch,
                period=f"{years}y",
                interval="1d",
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception as exc:  # network hiccup etc.
            log.warning("batch download failed (%s); retrying one by one", exc)
            data = None

        for t in batch:
            df = None
            if data is not None and not data.empty:
                try:
                    df = data[t] if len(batch) > 1 else data
                except KeyError:
                    df = None
            if df is None or df.dropna(how="all").empty:
                try:
                    df = yf.download(t, period=f"{years}y", interval="1d",
                                     auto_adjust=True, progress=False)
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                except Exception as exc:
                    log.warning("%s: download failed (%s)", t, exc)
                    continue
            df = _clean(df)
            if df is None:
                log.warning("%s: no usable data, skipped", t)
                continue
            df.to_csv(raw_dir / f"{t}.csv")
            result[t] = df

    return result


def _clean(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or df.empty:
        return None
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1) if "Close" in df.columns.get_level_values(-1) \
            else df.columns.get_level_values(0)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return None
    df = df[COLUMNS].dropna(subset=["Close"])
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    if len(df) < 60:
        return None
    return df


def load_cached(path: Path) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None
    return _clean(df)


def load_all(raw_dir: Path, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Load every cached CSV (optionally restricted to ``tickers``)."""
    out: dict[str, pd.DataFrame] = {}
    for path in sorted(raw_dir.glob("*.csv")):
        t = path.stem
        if tickers is not None and t not in tickers:
            continue
        df = load_cached(path)
        if df is not None:
            out[t] = df
    return out


def add_indicators(df: pd.DataFrame, ma_window: int = 10) -> pd.DataFrame:
    """Add the moving average column used everywhere else (``ma10`` by default)."""
    df = df.copy()
    df[f"ma{ma_window}"] = df["Close"].rolling(ma_window, min_periods=ma_window).mean()
    return df.dropna(subset=[f"ma{ma_window}"])
