#!/usr/bin/env python3
"""Run the whole research pipeline end to end:

    python run_all.py                      # all sectors, up to 300 stocks
    python run_all.py --sectors Technology # one group of similar stocks
    python run_all.py --epochs 20 --window 90

Steps: download -> charts -> train -> predict.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent / "scripts"
STEPS = ["1_download.py", "2_make_charts.py", "3_train.py", "4_predict.py"]


def main() -> None:
    extra = sys.argv[1:]
    for step in STEPS:
        print(f"\n===== {step} =====", flush=True)
        rc = subprocess.call([sys.executable, str(SCRIPTS / step), *extra])
        if rc != 0:
            sys.exit(f"{step} failed with exit code {rc}")


if __name__ == "__main__":
    main()
