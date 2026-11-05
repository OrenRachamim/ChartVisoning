#!/usr/bin/env python3
"""Step 1 - download daily history for the stock universe into data/raw/."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chartvision.cli import base_parser, config_from_args, resolve_tickers  # noqa: E402
from chartvision.data import download_history  # noqa: E402


def main() -> None:
    p = base_parser(__doc__)
    p.add_argument("--refresh", action="store_true", help="re-download cached tickers")
    args = p.parse_args()
    cfg = config_from_args(args)
    tickers = resolve_tickers(args, cfg)
    frames = download_history(tickers, cfg.raw_dir, years=cfg.history_years, refresh=args.refresh)
    print(f"downloaded/cached {len(frames)} of {len(tickers)} tickers into {cfg.raw_dir}")


if __name__ == "__main__":
    main()
