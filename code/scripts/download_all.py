"""Download every dataset the harness can use (the paper needs only download_tslib.py).

    python scripts/download_all.py                 # TSLib collection + OPSD
    python scripts/download_all.py --skip opsd     # only the TSLib collection (ETT*, ECL, Traffic, Weather, Exchange)

Each source keeps its own script (checksums pinned inside): download_tslib.py | download_opsd.py
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
JOBS = {
    "tslib": ["download_tslib.py"],
    "opsd": ["download_opsd.py"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", nargs="*", default=[], choices=list(JOBS))
    args = ap.parse_args()
    failed = []
    for name, cmd in JOBS.items():
        if name in args.skip:
            continue
        print(f"\n===== {name} =====", flush=True)
        r = subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]])
        if r.returncode != 0:
            failed.append(name)
    if failed:
        print(f"\nFAILED: {failed} (re-run later, see DATASETS.md)")
        sys.exit(1)
    print("\nall datasets present")


if __name__ == "__main__":
    main()
