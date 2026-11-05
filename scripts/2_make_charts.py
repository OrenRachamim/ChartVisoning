#!/usr/bin/env python3
"""Step 2 - render the last 500 days of the MA10 line for every stock into
data/charts/<TICKER>.png, plus a few example training windows into
data/charts/_windows/ so you can see what the CNN looks at."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm  # noqa: E402

from chartvision.charts import plot_ma_chart, save_window_preview  # noqa: E402
from chartvision.cli import base_parser, config_from_args, resolve_tickers  # noqa: E402
from chartvision.data import add_indicators, load_all  # noqa: E402


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--previews", type=int, default=8, help="number of window previews to dump")
    args = p.parse_args()
    cfg = config_from_args(args)
    tickers = resolve_tickers(args, cfg)
    frames = load_all(cfg.raw_dir, tickers)
    if not frames:
        sys.exit("no cached data - run scripts/1_download.py first")

    ma_col = f"ma{cfg.ma_window}"
    n = 0
    for t, df in tqdm(sorted(frames.items()), desc="charts"):
        df = add_indicators(df, cfg.ma_window)
        plot_ma_chart(df, t, cfg.chart_dir / f"{t}.png", ma_col=ma_col, days=cfg.chart_days)
        n += 1
        if n <= args.previews:
            cols = [ma_col if c != "close" else "Close" for c in cfg.channels]
            series = df[cols].tail(cfg.window).to_numpy()
            save_window_preview(series, cfg.chart_dir / "_windows" / f"{t}_last{cfg.window}d.png",
                                size=cfg.img_size)
    print(f"wrote {n} charts to {cfg.chart_dir}")


if __name__ == "__main__":
    main()
