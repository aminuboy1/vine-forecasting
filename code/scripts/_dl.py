"""Shared download helpers (stdlib only so they also run outside the venv).

Every file is verified after download with at least one of:
  * sha256          (Hugging Face LFS oid, or a hash we recorded on first download)
  * git_blob_sha1   (the git object id HF reports for non-LFS files: sha1(b"blob <size>\\0" + content))
  * md5             (Zenodo-provided checksum)
  * size            (byte count)
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

UA = {"User-Agent": "tsl-dataset-downloader/1.0 (research harness)"}


def _hashers():
    return {"sha256": hashlib.sha256(), "md5": hashlib.md5(), "sha1": hashlib.sha1()}


def file_digests(path: Path) -> dict:
    size = path.stat().st_size
    hs = _hashers()
    blob = hashlib.sha1(f"blob {size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            for h in hs.values():
                h.update(chunk)
            blob.update(chunk)
    out = {k: h.hexdigest() for k, h in hs.items()}
    out["git_blob_sha1"] = blob.hexdigest()
    out["size"] = size
    return out


def verify(path: Path, expect: dict) -> list[str]:
    """Return a list of mismatch messages (empty = OK)."""
    got = file_digests(path)
    errs = []
    for k, v in expect.items():
        if v is None:
            continue
        if str(got[k]) != str(v):
            errs.append(f"{path.name}: {k} expected {v} got {got[k]}")
    return errs


def download(url: str, dest: Path, expect: dict | None = None, retries: int = 3, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    expect = expect or {}
    if dest.exists() and not force:
        errs = verify(dest, expect) if expect else []
        if not errs:
            print(f"[skip] {dest} already present and verified")
            return dest
        print(f"[redo] {dest} failed verification: {errs}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            print(f"[get ] {url}\n       -> {dest}")
            req = urllib.request.Request(url, headers=UA)
            t0 = time.time()
            with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
                total = int(r.headers.get("Content-Length") or 0)
                done = 0
                last = 0.0
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    done += len(buf)
                    if time.time() - last > 5:
                        last = time.time()
                        pct = f"{100 * done / total:5.1f}%" if total else ""
                        print(f"       {done / 1e6:9.1f} MB {pct}", flush=True)
            os.replace(tmp, dest)
            print(f"       done in {time.time() - t0:.1f}s ({dest.stat().st_size / 1e6:.1f} MB)")
            break
        except Exception as e:  # noqa: BLE001
            print(f"       attempt {attempt} failed: {e}", file=sys.stderr)
            if attempt == retries:
                raise
            time.sleep(3 * attempt)
    errs = verify(dest, expect) if expect else []
    if errs:
        bad = dest.with_suffix(dest.suffix + ".bad")
        shutil.move(dest, bad)
        raise RuntimeError("Checksum verification failed: " + "; ".join(errs))
    if expect:
        print(f"       verified: {', '.join(k for k, v in expect.items() if v is not None)}")
    return dest


def default_data_root() -> Path:
    return Path(os.environ.get("TSL_DATA", Path(__file__).resolve().parents[1] / "data"))
