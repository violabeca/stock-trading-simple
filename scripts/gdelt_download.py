"""Download GDELT 2.0 files (Events, Mentions, GKG) to a folder, resumable and MD5-verified.

GDELT publishes one file of each kind every 15 minutes at data.gdeltproject.org/gdeltv2/;
its ``masterfilelist.txt`` lists every file with its size and MD5. This script reads that
list, keeps the files in a time window and fetches the missing ones with a few parallel
connections (a static file host, not the throttled DOC API). A file is written as
``.part`` and renamed only when its size and MD5 match, so a killed run resumes by
re-running the same command; finished files are skipped by size.

  uv run python scripts/gdelt_download.py --dest /Volumes/TradingData/gdelt --since 2025-01-01
  uv run python scripts/gdelt_download.py --dest /tmp/g --since 2025-01-01T00 \
      --until 2025-01-01T03
  uv run python scripts/gdelt_download.py --dest /Volumes/TradingData/gdelt \
      --since 2025-01-01 --verify

Layout: <dest>/{events,mentions,gkg}/<YYYY>/<MM>/<file>.zip, plus <dest>/failed.txt for
any file that could not be fetched (a few files are missing on GDELT's side).
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import re
import shutil
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

HOST = "data.gdeltproject.org"
MASTER = f"https://{HOST}/gdeltv2/masterfilelist.txt"
KIND = {"export": "events", "mentions": "mentions", "gkg": "gkg"}
NAME = re.compile(r"/gdeltv2/((\d{4})(\d{2})\d{8})\.(export|mentions|gkg)\.")
RETRIES = 4
LOCAL = threading.local()


def parse_master(path: Path, since: str, until: str) -> list[tuple[int, str, str, Path]]:
    """Read masterfilelist.txt and keep the files whose timestamp is in ``[since, until)``.

    Args:
        path: The downloaded master list (``size md5 url`` per line).
        since: First timestamp as ``YYYYMMDDHHMMSS`` digits (a prefix is padded with zeros).
        until: First timestamp NOT wanted, same format.

    Returns:
        ``(size, md5, filename, relative_path)`` per file, oldest first.
    """
    out = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        m = NAME.search(parts[2])
        if not m or not (since <= m.group(1) < until):
            continue
        name = parts[2].rsplit("/", 1)[1]
        out.append((int(parts[0]), parts[1], name,
                    Path(KIND[m.group(4)]) / m.group(2) / m.group(3) / name))
    return sorted(out, key=lambda t: t[2])


def fetch(size: int, md5: str, name: str, dest: Path) -> str:
    """Download one file to ``dest`` (via ``.part``), verifying size and MD5.

    Args:
        size: Expected byte size.
        md5: Expected MD5 hex digest.
        name: File name on the server.
        dest: Final local path.

    Returns:
        ``"skip"`` (already there), ``"ok"`` (downloaded) or ``"fail: <reason>"``.
    """
    if dest.exists() and dest.stat().st_size == size:
        return "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    reason = ""
    for attempt in range(1, RETRIES + 1):
        try:
            conn = getattr(LOCAL, "conn", None) or http.client.HTTPSConnection(HOST, timeout=60)
            LOCAL.conn = conn
            conn.request("GET", f"/gdeltv2/{name}")
            resp = conn.getresponse()
            if resp.status != 200:
                resp.read()
                reason = f"HTTP {resp.status}"
                if resp.status == 404:
                    break
                continue
            h, n = hashlib.md5(), 0
            with part.open("wb") as f:
                while chunk := resp.read(1 << 20):
                    f.write(chunk)
                    h.update(chunk)
                    n += len(chunk)
            if n == size and h.hexdigest() == md5:
                part.rename(dest)
                return "ok"
            reason = f"size/md5 mismatch ({n} of {size})"
        except (OSError, http.client.HTTPException) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            LOCAL.conn = None
        time.sleep(2 * attempt)
    part.unlink(missing_ok=True)
    return f"fail: {reason}"


def verify(files: list[tuple[int, str, str, Path]], root: Path) -> None:
    """Recompute the MD5 of every local file in the window and report the bad ones.

    Args:
        files: The wanted files (see :func:`parse_master`).
        root: Download folder.
    """
    bad = missing = 0
    for size, md5, _, rel in files:
        p = root / rel
        if not p.exists():
            missing += 1
        elif p.stat().st_size != size or hashlib.md5(p.read_bytes()).hexdigest() != md5:
            bad += 1
            print("BAD", p)
    print(f"verify: {len(files)} wanted, {missing} missing, {bad} corrupt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dest", type=Path, required=True, help="download folder (created)")
    ap.add_argument("--since", required=True, help="first day, YYYY-MM-DD[THH]")
    ap.add_argument("--until", default="2999-01-01", help="first day NOT wanted (default: all)")
    ap.add_argument("--kinds", default="events,mentions,gkg", help="comma list")
    ap.add_argument("--workers", type=int, default=8, help="parallel connections (default 8)")
    ap.add_argument("--verify", action="store_true", help="only re-check local files' MD5")
    args = ap.parse_args()

    def digits(s: str) -> str:
        return re.sub(r"\D", "", s).ljust(14, "0")

    args.dest.mkdir(parents=True, exist_ok=True)
    master = args.dest / "masterfilelist.txt"
    if not master.exists() or time.time() - master.stat().st_mtime > 3600:
        with urllib.request.urlopen(MASTER, timeout=120) as r, master.open("wb") as f:
            shutil.copyfileobj(r, f)
    kinds = set(args.kinds.split(","))
    files = [t for t in parse_master(master, digits(args.since), digits(args.until))
             if t[3].parts[0] in kinds]
    total = sum(t[0] for t in files)
    print(f"{len(files)} files, {total / 1e9:.1f} GB in the window; "
          f"free on drive {shutil.disk_usage(args.dest).free / 1e9:.0f} GB", flush=True)
    if args.verify:
        return verify(files, args.dest)
    if total > shutil.disk_usage(args.dest).free * 0.95:
        raise SystemExit("not enough free space for this window")

    counts = {"ok": 0, "skip": 0, "fail": 0}
    t0, done_bytes, lock = time.monotonic(), 0, threading.Lock()
    failed = args.dest / "failed.txt"

    def work(t: tuple[int, str, str, Path]) -> None:
        nonlocal done_bytes
        res = fetch(t[0], t[1], t[2], args.dest / t[3])
        with lock:
            key = res.split(":")[0]
            counts[key] += 1
            done_bytes += t[0] if key == "ok" else 0
            if key == "fail":
                failed.open("a").write(f"{t[2]}\t{res}\n")
            n = sum(counts.values())
            if n % 2000 == 0 or n == len(files):
                el = time.monotonic() - t0
                print(f"{n}/{len(files)} {counts} | {done_bytes / 1e6 / max(el, 1):.1f} MB/s | "
                      f"{datetime.now():%H:%M:%S}", flush=True)

    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(work, files))
    print(f"done: {counts}", flush=True)


if __name__ == "__main__":
    main()
