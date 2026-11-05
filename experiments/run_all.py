#!/usr/bin/env python3
"""Run every experiment stage in order (each stage is resumable):

    python experiments/run_all.py            # sweep -> walk-forward -> groups -> per-stock -> report
    python experiments/run_all.py --from 3   # start at stage 3
"""

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ["sweep.py", "walkforward.py", "groups.py", "perstock.py", "report.py"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from", dest="start", type=int, default=1)
    args = p.parse_args()
    for i, stage in enumerate(STAGES, 1):
        if i < args.start:
            continue
        print(f"\n===== stage {i}: {stage} =====", flush=True)
        rc = subprocess.call([sys.executable, "-u", str(HERE / stage)], cwd=str(HERE))
        if rc != 0:
            sys.exit(f"{stage} failed ({rc})")


if __name__ == "__main__":
    main()
