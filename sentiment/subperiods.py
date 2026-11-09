#!/usr/bin/env python3
"""Is the VIX level effect stable over time? Extreme-fear / extreme-greed
21-day returns per index and per sub-period."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
from sentiment import analysis as A
from sentiment.data import market_data
from sentiment.run_study import vix_to_index, OUT

m = market_data(); lv = vix_to_index(m["^VIX"].dropna()).dropna()
rows = []
for tgt, label in [("^GSPC", "S&P 500"), ("^IXIC", "Nasdaq"), ("^RUT", "Russell 2000")]:
    px = m[tgt].dropna(); idx = lv.index.intersection(px.index)
    for a, b in [("1990", "2001"), ("2002", "2012"), ("2013", "2026")]:
        sl = slice(f"{a}-01-01", f"{b}-12-31")
        bt = A.bucket_table(lv.loc[idx].loc[sl], A.forward_returns(px.loc[idx].loc[sl]))
        ef, eg = bt.iloc[0], bt.iloc[-1]
        rows.append({"index": label, "period": f"{a}-{b}", "extreme_fear_21d_pct": ef.ret_21_pct, "t_fear": ef.t_21,
                     "extreme_greed_21d_pct": eg.ret_21_pct, "t_greed": eg.t_21, "base_21d_pct": ef.base_21_pct, "t_greed_5d": eg.t_5})
df = pd.DataFrame(rows).round(2)
print(df.to_string(index=False))
json.dump(df.to_dict("records"), open(OUT / "subperiods.json", "w"), indent=2)
