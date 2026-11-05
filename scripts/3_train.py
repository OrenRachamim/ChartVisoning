#!/usr/bin/env python3
"""Step 3 - build labelled chart windows from every stock and train the CNN.
Writes models/chart_cnn.pt and models/training_report.json."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from chartvision.cli import base_parser, config_from_args, resolve_tickers  # noqa: E402
from chartvision.data import add_indicators, load_all  # noqa: E402
from chartvision.dataset import build_samples  # noqa: E402
from chartvision.train import train  # noqa: E402


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = config_from_args(args)
    tickers = resolve_tickers(args, cfg)
    frames = {t: add_indicators(df, cfg.ma_window) for t, df in load_all(cfg.raw_dir, tickers).items()}
    if not frames:
        sys.exit("no cached data - run scripts/1_download.py first")
    samples = build_samples(frames, cfg)
    print(f"{len(samples)} windows from {len(frames)} stocks")
    ckpt = train(samples, cfg)
    print(f"model saved to {ckpt}")


if __name__ == "__main__":
    main()
