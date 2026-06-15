"""RAM-safe BREADTH build: extend pertkr from 186 -> the full ~682-name liquid OPRA
cross-section (every underlying with >=500 trading days), using on-disk greeks only
(no download). Reuses build_pertkr's extract/merge but drives them in TICKER-CHUNKS so
the number of simultaneously-open ParquetWriters (and thus peak RAM) is bounded,
deletes each chunk's shards after merge (bounded transient disk), and SELF-THROTTLES
on a free-RAM guard so it never OOMs. Resumable: already-built tickers are skipped, so
re-running continues where it stopped.

Launch when RAM is durably free. Example (when free>=20GB):
  taskset -c 0-15 python3 tools/build_pertkr_breadth.py --workers 12 --chunk 130 --min_avail_gb 12
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import PERTKR, SURFACE_PANEL

import sys, os, glob, time, shutil, argparse
import multiprocessing as mp
import pandas as pd
import build_pertkr as B
from optengine.data import greeks_store as G

PANEL = SURFACE_PANEL


def mem_avail_gb():
    for ln in open("/proc/meminfo"):
        if ln.startswith("MemAvailable:"):
            return int(ln.split()[1]) / 1024 / 1024
    return 0.0


def disk_free_gb(path=PERTKR):
    s = os.statvfs(path)
    return s.f_bavail * s.f_frsize / 1e9


def breadth_universe(min_days=500):
    d = pd.read_parquet(PANEL, columns=["underlying", "date"])
    g = d.groupby("underlying")["date"].nunique()
    cand = sorted(t for t in g[g >= min_days].index if isinstance(t, str))
    built = {os.path.basename(f)[:-8] for f in glob.glob(os.path.join(B.OUT, "*.parquet"))}
    return [t for t in cand if t not in built]      # NEW names only (resumable)


def merge_one(t):
    shards = sorted(glob.glob(os.path.join(B.SH, f"{t}__b*.parquet")))
    if not shards:
        return (t, 0)
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    df.sort_values(["asof_i", "expiry_i", "cp", "Strike"], inplace=True, kind="stable")
    df.to_parquet(os.path.join(B.OUT, f"{t}.parquet"), compression="zstd", index=False)
    for s in shards:                                 # bounded transient disk: drop shards now
        os.remove(s)
    return (t, len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=130, help="tickers per chunk (bounds open writers/peak RAM)")
    ap.add_argument("--min_avail_gb", type=float, default=12.0, help="wait/abort if free RAM below this")
    ap.add_argument("--min_disk_gb", type=float, default=20.0)
    ap.add_argument("--min_days", type=int, default=500)
    a = ap.parse_args()
    os.makedirs(B.SH, exist_ok=True)
    uni = breadth_universe(a.min_days)
    dates = sorted(d for y in range(2017, 2027) for d in G.available_dates(y))
    chunks = [uni[i:i + a.chunk] for i in range(0, len(uni), a.chunk)]
    print(f"BREADTH build: {len(uni)} NEW tickers in {len(chunks)} chunks of <= {a.chunk}, "
          f"{len(dates)} days, workers={a.workers}", flush=True)
    t0 = time.time()
    for ci, names in enumerate(chunks):
        # ---- RAM/disk guard: wait for the box to be free, then proceed (or abort) ----
        waited = 0
        while mem_avail_gb() < a.min_avail_gb:
            if waited == 0:
                print(f"  [guard] free RAM {mem_avail_gb():.1f}G < {a.min_avail_gb}G — waiting...",
                      flush=True)
            time.sleep(60); waited += 60
            if waited > 3600:
                print("  [guard] RAM did not free in 60min — stopping cleanly (resume later).", flush=True)
                return
        if disk_free_gb() < a.min_disk_gb:
            print(f"  [guard] disk free {disk_free_gb():.0f}G < {a.min_disk_gb}G — stopping.", flush=True)
            return
        # ---- extract this chunk (each zip opened once; strided days across workers) ----
        W = a.workers
        blocks = [(i, dates[i::W], names) for i in range(W)]
        te = time.time()
        with mp.Pool(W, maxtasksperchild=1) as pool:
            for _ in pool.imap_unordered(B._worker, blocks):
                pass
        # ---- merge this chunk in parallel, then shards are removed by merge_one ----
        with mp.Pool(min(W, 8)) as pool:
            done = pool.map(merge_one, names)
        nrows = sum(n for _, n in done)
        print(f"  chunk {ci+1}/{len(chunks)}: {len([1 for _,n in done if n])} tickers merged "
              f"({nrows/1e6:.1f}M rows) [{time.time()-te:.0f}s, free RAM {mem_avail_gb():.1f}G, "
              f"disk {disk_free_gb():.0f}G, total {time.time()-t0:.0f}s]", flush=True)
    built = len(glob.glob(os.path.join(B.OUT, "*.parquet")))
    print(f"DONE: pertkr now has {built} tickers [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
