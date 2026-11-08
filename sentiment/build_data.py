#!/usr/bin/env python3
"""Download everything the sentiment study needs and build the reconstructed index."""
import logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from chartvision.universe import LARGE_UNIVERSE
from sentiment.data import SDIR, cnn_fear_greed, market_data, reconstruct_fear_greed, universe_long

logging.basicConfig(level=logging.INFO)
tickers = sorted({t for v in LARGE_UNIVERSE.values() for t in v})
mkt = market_data()
print("market", mkt.shape, mkt.index.min().date(), mkt.index.max().date())
close, vol = universe_long(tickers)
print("universe", close.shape, close.index.min().date())
scores = reconstruct_fear_greed(mkt, close, vol)
scores.to_csv(SDIR / "fear_greed_reconstructed.csv")
print("reconstructed", scores.shape, scores.index.min().date(), scores.index.max().date())
real = cnn_fear_greed()
both = pd.concat([real, scores["fg_recon"]], axis=1).dropna() if (pd := __import__("pandas")) else None
print("overlap with CNN:", len(both), "corr", round(both.corr().iloc[0, 1], 3),
      "corr of 5d changes", round(both.diff(5).corr().iloc[0, 1], 3))
