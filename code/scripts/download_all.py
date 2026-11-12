"""Download every dataset used by the two papers (laptop: ~4.5 GB incl. extraction).

    python scripts/download_all.py                 # everything except the 40 GB CESNET per-IP archive
    python scripts/download_all.py --skip streams  # skip the Google Drive sources (needs gdown)

Each source keeps its own script (checksums pinned inside):
  download_tslib.py (ETT*, ECL, Traffic, Weather, Exchange) | download_opsd.py | download_uci_ecl.py |
  download_cesnet.py --extract | download_streams.py (Energy/PEMS/Air/CA-Stream, TrafficStream raw zip)
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
    "uci": ["download_uci_ecl.py"],
    "cesnet": ["download_cesnet.py", "--extract"],
    "streams": ["download_streams.py", "--only", "energy", "pems", "air", "ca"],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip", nargs="*", default=[], choices=list(JOBS))
    ap.add_argument("--with-trafficstream", action="store_true", help="also fetch the 2.1 GB TrafficStream zip")
    args = ap.parse_args()
    failed = []
    for name, cmd in JOBS.items():
        if name in args.skip:
            continue
        if name == "streams" and args.with_trafficstream:
            cmd = cmd + ["trafficstream", "--extract"]
        print(f"\n===== {name} =====", flush=True)
        r = subprocess.run([sys.executable, str(HERE / cmd[0]), *cmd[1:]])
        if r.returncode != 0:
            failed.append(name)
    if failed:
        print(f"\nFAILED: {failed} (re-run later; Google Drive may rate-limit gdown, see DATASETS.md)")
        sys.exit(1)
    print("\nall datasets present")


if __name__ == "__main__":
    main()
