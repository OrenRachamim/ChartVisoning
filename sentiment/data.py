"""Data for the sentiment study.

* market series (S&P 500, SPY, VIX, VIX3M, bonds, HYG/LQD, BTC) via yfinance
* 20 years of daily bars for the large-cap universe (breadth components)
* CNN Fear & Greed: the ~250 days the public endpoint returns (validation)
* crypto Fear & Greed from alternative.me (2018 -> today)
* a reconstruction of the CNN index from its published components
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent.parent
SDIR = ROOT / "data" / "sentiment"
SDIR.mkdir(parents=True, exist_ok=True)
log = logging.getLogger(__name__)

MARKET = ["^GSPC", "SPY", "^VIX", "^VIX3M", "TLT", "IEF", "HYG", "LQD", "RSP", "QQQ", "IWM", "BTC-USD"]


def market_data(refresh: bool = False) -> pd.DataFrame:
    path = SDIR / "market.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    raw = yf.download(MARKET, period="max", interval="1d", auto_adjust=True, group_by="ticker",
                      progress=False, threads=False)
    out = pd.DataFrame({t: raw[t]["Close"] for t in MARKET})
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out.to_csv(path)
    return out


def universe_long(tickers: list[str], years: int = 20, refresh: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Close and volume panels (date x ticker) for the breadth components."""
    pc, pv = SDIR / "universe_close.csv", SDIR / "universe_volume.csv"
    if pc.exists() and pv.exists() and not refresh:
        return (pd.read_csv(pc, index_col=0, parse_dates=True), pd.read_csv(pv, index_col=0, parse_dates=True))
    closes, vols = {}, {}
    for i in range(0, len(tickers), 50):
        batch = tickers[i : i + 50]
        raw = yf.download(batch, period=f"{years}y", interval="1d", auto_adjust=True, group_by="ticker",
                          progress=False, threads=True)
        for t in batch:
            try:
                closes[t] = raw[t]["Close"]; vols[t] = raw[t]["Volume"]
            except KeyError:
                pass
    close, vol = pd.DataFrame(closes), pd.DataFrame(vols)
    close.index = pd.to_datetime(close.index).tz_localize(None); vol.index = close.index
    close.to_csv(pc); vol.to_csv(pv)
    return close, vol


def cnn_fear_greed() -> pd.Series:
    df = pd.read_csv(SDIR / "cnn_fear_greed_1y.csv", parse_dates=["date"])
    s = df.set_index("date")["y"].astype(float)
    s.index = s.index.normalize()
    return s[~s.index.duplicated()].sort_index().rename("cnn_fg")


def crypto_fear_greed() -> pd.Series:
    df = pd.read_csv(SDIR / "crypto_fear_greed.csv", parse_dates=["date"])
    s = df.set_index("date")["value"].astype(float)
    s.index = s.index.normalize()
    return s[~s.index.duplicated()].sort_index().rename("crypto_fg")


# --------------------------------------------------------------------------
# reconstruction of the CNN index
# --------------------------------------------------------------------------
def _pct_rank(x: pd.Series, window: int = 500) -> pd.Series:
    """Rolling percentile rank (0..100) of the latest value within the trailing window."""
    return x.rolling(window, min_periods=window // 2).apply(lambda w: (w[:-1] < w[-1]).mean() * 100, raw=True)


def reconstruct_fear_greed(mkt: pd.DataFrame, close: pd.DataFrame, vol: pd.DataFrame,
                           rank_window: int = 500) -> pd.DataFrame:
    """Six of CNN's seven components (put/call is not freely available), each
    turned into a 0..100 score by its percentile within the trailing two years,
    then averaged. Higher = greed."""
    spx = mkt["^GSPC"].dropna()
    comp = pd.DataFrame(index=spx.index)
    # 1. market momentum: S&P 500 vs its 125-day average
    comp["momentum"] = spx / spx.rolling(125).mean() - 1
    # 2. stock price strength: net 52-week highs minus lows, share of stocks
    hi = close.rolling(252).max(); lo = close.rolling(252).min()
    new_hi = (close >= hi).sum(axis=1); new_lo = (close <= lo).sum(axis=1); n = close.notna().sum(axis=1)
    comp["strength"] = ((new_hi - new_lo) / n).rolling(5).mean().reindex(spx.index)
    # 3. breadth: McClellan volume summation index (advancing minus declining volume)
    up = close.diff() > 0
    adv = (vol * up).sum(axis=1); dec = (vol * ~up).sum(axis=1)
    net = ((adv - dec) / (adv + dec)).reindex(spx.index)
    osc = net.ewm(span=19).mean() - net.ewm(span=39).mean()
    comp["breadth"] = osc.cumsum()
    # 4. put/call ratio: not available -> skipped
    # 5. volatility: VIX vs its 50-day average (inverted: high VIX = fear)
    vix = mkt["^VIX"].reindex(spx.index)
    comp["volatility"] = -(vix / vix.rolling(50).mean() - 1)
    # 6. safe haven: stocks minus treasuries over 20 days
    comp["safe_haven"] = (spx.pct_change(20) - mkt["IEF"].reindex(spx.index).pct_change(20))
    # 7. junk bond demand: HYG vs LQD over 20 days (spread narrowing = greed)
    comp["junk"] = (mkt["HYG"].reindex(spx.index).pct_change(20) - mkt["LQD"].reindex(spx.index).pct_change(20))
    scores = comp.apply(_pct_rank, window=rank_window)
    scores["fg_recon"] = scores.mean(axis=1, skipna=False)
    return scores.dropna(subset=["fg_recon"])
