"""Download Open Power System Data "Time series" package, version 2020-10-06 (hourly, single index).

Source: https://data.open-power-system-data.org/time_series/2020-10-06/
        (DOI 10.25832/time_series/2020-10-06). No explicit data licence is declared: datapackage.json has no
        "licenses" field and the landing page only asks for attribution ("Open Power System Data. 2020. Data
        Package Time series. Version 2020-10-06. https://doi.org/10.25832/time_series/2020-10-06. (Primary data
        from various sources, for a complete list see URL)."). Primary data: ENTSO-E Transparency Platform terms.
        The CSV sha256 below equals the "hash" published for it in datapackage.json.
Files:
  * time_series_60min_singleindex.csv  -- ',' separated, '.' decimal, UTC + CET/CEST timestamp columns
  * datapackage.json                   -- Frictionless metadata (licence, column descriptions)

Usage:
    python scripts/download_opsd.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dl import default_data_root, download  # noqa: E402

VERSION = "2020-10-06"
BASE = f"https://data.open-power-system-data.org/time_series/{VERSION}"
# name -> expected (recorded on first download, 2026-11-12)
FILES = {
    "time_series_60min_singleindex.csv": {"size": 130339665,
                                          "sha256": "6a7f2bc571314cbf9c321cc03437691cd4be95c3a6f075e60ff99e8035c704c8"},
    "datapackage.json": {"size": 303438,
                         "sha256": "2f4b7ae48e1352deb1991728e013fe42e742af5e81a5140d87185e470cd0ac36"},
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=None, help="data root (default: $TSL_DATA or code/data)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    root = (args.root or default_data_root()) / "opsd"
    for name, expect in FILES.items():
        download(f"{BASE}/{name}", root / name, expect, force=args.force)
    print(f"OPSD time_series {VERSION} present under {root}")


if __name__ == "__main__":
    main()
