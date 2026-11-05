#!/usr/bin/env python3
"""Step 4 - forecast the next 5 / 10 days for every stock.
Writes data/predictions/forecast.csv and one chart per stock with the
projected MA10 path drawn on it."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from chartvision.cli import base_parser, config_from_args, resolve_tickers  # noqa: E402
from chartvision.data import add_indicators, load_all  # noqa: E402
from chartvision.predict import predict_all  # noqa: E402


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--model", default=None, help="checkpoint path (default models/chart_cnn.pt)")
    p.add_argument("--no-charts", action="store_true", help="skip drawing forecast charts")
    p.add_argument("--top", type=int, default=20, help="rows to print")
    p.add_argument("--per-stock-dir", default=None,
                   help="folder with fine-tuned <TICKER>.pt models to blend in (see experiments/perstock.py)")
    p.add_argument("--ensemble-weight", type=float, default=0.5, help="weight of the per-stock model")
    args = p.parse_args()
    cfg = config_from_args(args)
    ckpt = Path(args.model) if args.model else cfg.model_dir / "chart_cnn.pt"
    if not ckpt.exists():
        sys.exit(f"{ckpt} not found - run scripts/3_train.py first")

    tickers = resolve_tickers(args, cfg)
    frames = {t: add_indicators(df, cfg.ma_window) for t, df in load_all(cfg.raw_dir, tickers).items()}
    table = predict_all(frames, cfg, ckpt, draw_charts=not args.no_charts,
                        per_stock_dir=Path(args.per_stock_dir) if args.per_stock_dir else None,
                        ensemble_weight=args.ensemble_weight)
    out = cfg.pred_dir / "forecast.csv"
    table.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    show = [c for c in table.columns if c.startswith(("ticker", "as_of", "close", "pred_", "conf_", "proj_ma_1"))]
    print(table[show].head(args.top).to_string(index=False))
    print(f"\nfull table: {out}")
    if not args.no_charts:
        print(f"charts:     {cfg.pred_dir}/<TICKER>_forecast.png")


if __name__ == "__main__":
    main()
