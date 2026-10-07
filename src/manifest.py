"""Record and verify the raw data snapshot.

Usage
-----
    python src/manifest.py           # write data/manifest.json from data/raw/
    python src/manifest.py --check   # confirm data/raw/ still matches the manifest

The manifest records, for every ticker: row count, first and last date, file
size, and a SHA-256 hash (a fingerprint of the file's exact bytes). It also
records when the files were downloaded and which library versions were used.

data/raw/ stays out of git; data/manifest.json is committed. Anyone with the
same snapshot can run --check to prove their files are identical to yours, and
any change to a file, however small, changes its hash.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data  # noqa: E402  (data.py in the same folder)

MANIFEST = data.ROOT / "data" / "manifest.json"
PACKAGES = ["yfinance", "pandas", "pyarrow", "numpy"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def describe(ticker: str) -> dict:
    """Facts about one ticker's file."""
    path = data.raw_path(ticker)
    df = data.load_raw(ticker)
    modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return {
        "ticker": ticker,
        "file": path.relative_to(data.ROOT).as_posix(),
        "rows": int(len(df)),
        "first_date": df.index[0].date().isoformat() if len(df) else None,
        "last_date": df.index[-1].date().isoformat() if len(df) else None,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "downloaded_at": modified.isoformat(timespec="seconds"),
    }


def build() -> int:
    tickers = data.universe_tickers()
    files, missing = [], []
    for t in tickers:
        if data.raw_path(t).exists():
            files.append(describe(t))
        else:
            missing.append(t)

    if not files:
        print("No files in data/raw/. Run `python src/data.py` first.")
        return 1

    # File modification times are when data.py saved each file.
    dates = sorted({f["downloaded_at"][:10] for f in files})
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "download_dates": dates,
        "source": "Yahoo Finance via yfinance, auto_adjust=True",
        "requested_range": {"start": data.START, "end_exclusive": data.END},
        "python": platform.python_version(),
        "packages": {p: package_version(p) for p in PACKAGES},
        "ticker_count": len(files),
        "missing": missing,
        "files": files,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")

    for f in files:
        print(f"  {f['ticker']:<6} {f['rows']:>5} rows  {f['first_date']} to {f['last_date']}"
              f"  {f['sha256'][:12]}")
    print(f"\n{len(files)} files recorded, downloaded on {', '.join(dates)}")
    print(f"yfinance {manifest['packages']['yfinance']}, Python {manifest['python']}")
    if missing:
        print(f"MISSING {len(missing)}: {', '.join(missing)}. "
              "Run `python src/data.py`, then rebuild the manifest.")
    print(f"Wrote {MANIFEST.relative_to(data.ROOT)}")
    return 1 if missing else 0


def check(quiet: bool = False) -> bool:
    """True if every file in the manifest exists and its hash still matches."""
    if not MANIFEST.exists():
        print("No data/manifest.json. Run `python src/manifest.py` first.")
        return False
    manifest = json.loads(MANIFEST.read_text())
    problems = []
    for f in manifest["files"]:
        path = data.ROOT / f["file"]
        if not path.exists():
            problems.append(f"{f['ticker']}: file missing")
        elif sha256(path) != f["sha256"]:
            problems.append(f"{f['ticker']}: file changed since the manifest was written")
    for t in manifest.get("missing", []):
        problems.append(f"{t}: never downloaded")

    installed = package_version("yfinance")
    recorded = manifest["packages"].get("yfinance")
    if installed != recorded and not quiet:
        print(f"Note: yfinance {installed} is installed; the snapshot used {recorded}. "
              "This only matters if you re-download.")

    if problems:
        print("Snapshot does NOT match the manifest:")
        for p in problems:
            print(f"  {p}")
        return False
    if not quiet:
        print(f"OK: all {len(manifest['files'])} files match the manifest.")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="verify data/raw/ against data/manifest.json")
    args = parser.parse_args()
    if args.check:
        sys.exit(0 if check() else 1)
    sys.exit(build())


if __name__ == "__main__":
    main()
