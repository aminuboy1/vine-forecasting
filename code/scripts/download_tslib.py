"""Download the standard long-term forecasting collection (Autoformer / Time-Series-Library).

Source: the official thuml Hugging Face mirror, pinned to a fixed commit so that the bytes are
reproducible:  https://huggingface.co/datasets/thuml/Time-Series-Library  (licence: CC-BY-4.0 as
declared on the dataset card; the underlying datasets keep their original licences, see DATASETS.md).
ETT files can alternatively be fetched from the original ETDataset GitHub repo (--ett-source github);
the files are byte-identical (same git blob ids).

Usage:
    python scripts/download_tslib.py                    # all 8 datasets (~265 MB)
    python scripts/download_tslib.py --only ETTh1 ETTh2 # subset
    TSL_DATA=/scratch/data python scripts/download_tslib.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _dl import default_data_root, download  # noqa: E402

HF_REPO = "thuml/Time-Series-Library"
HF_COMMIT = "2b66e59ee19dac8f6f19fb5d4997f289fdfea357"  # verified 2026-11-12
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/{HF_COMMIT}"
ETT_GITHUB = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small"

# name -> (repo path, size, git_blob_sha1 (non-LFS) or None, sha256 (LFS oid or recorded on first download))
FILES = {
    "ETTh1": ("ETT-small/ETTh1.csv", 2589657, "a52c4925778c07c1ef1a2cf6fd01594919717d9e",
                "f18de3ad269cef59bb07b5438d79bb3042d3be49bdeecf01c1cd6d29695ee066"),
    "ETTh2": ("ETT-small/ETTh2.csv", 2417960, "6e3760778598fdee5208213d3fce122637307eec",
                "a3dc2c597b9218c7ce1cd55eb77b283fd459a1d09d753063f944967dd6b9218b"),
    "ETTm1": ("ETT-small/ETTm1.csv", 10360719, "62c9f979e2dc5b05d9cc5b38891e60e6baf77417",
                "6ce1759b1a18e3328421d5d75fadcb316c449fcd7cec32820c8dafda71986c9e"),
    "ETTm2": ("ETT-small/ETTm2.csv", 9677236, "f9d46027c987d442d2d8f81308d1d8c4d1d35db8",
                "db973ca252c6410a30d0469b13d696cf919648d0f3fd588c60f03fdbdbadd1fd"),
    "exchange_rate": ("exchange_rate/exchange_rate.csv", 637800, "593763acecde07a4fcb6582741e615cca91a07b7",
                "48b4d9d3d508f5104162e85b9a6042e3557fde11aa9f2944eba8c0d0efc89842"),
    "weather": ("weather/weather.csv", 7235425, "490cacf73d5129165cbdabe74f99c39763ae8ebf",
                "34ee981d07313e51da2a50bb600072c8ae4a69cb4b0651f4cb93a069d7a2ba63"),
    "electricity": ("electricity/electricity.csv", 95581762, None,
                    "7e45845d54c5219bad0ae6bc1b5316cf8ff9cead5d33fa998a5a51c2e4a497ad"),
    "traffic": ("traffic/traffic.csv", 136478119, None,
                "cb06463d56fa17d87f47027cd9389ceae82a69eddee51cdb61480e120dab0b16"),
}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", default=None, help=f"subset of {list(FILES)}")
    ap.add_argument("--root", type=Path, default=None, help="data root (default: $TSL_DATA or code/data)")
    ap.add_argument("--ett-source", choices=["hf", "github"], default="hf")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    root = (args.root or default_data_root()) / "tslib"
    names = args.only or list(FILES)
    for n in names:
        if n not in FILES:
            raise SystemExit(f"unknown dataset {n}; choose from {list(FILES)}")
        path, size, blob, sha = FILES[n]
        if n.startswith("ETT") and args.ett_source == "github":
            url = f"{ETT_GITHUB}/{Path(path).name}"
        else:
            url = f"{HF_BASE}/{path}"
        download(url, root / path, {"size": size, "git_blob_sha1": blob, "sha256": sha}, force=args.force)
    print(f"All requested files present under {root}")


if __name__ == "__main__":
    main()
