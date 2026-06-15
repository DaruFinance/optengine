"""High-throughput FO pull: a FLAT (root,date) work-pool over all 8 roots x 2017-2026.
Each task recursively pulls one (root,date)'s expiry zips (awscli-concurrent), distills the
15:59-ET snapshot, writes fop_raw/<root>/<date>.parquet (race-free, atomic, resumable). Pinned
to cores 0-15; awscli concurrency tuned so workers don't thrash the link or starve other agents
(on 16-31). Recent years first so the engine can run on recent data while older streams in.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules

import os, sys, time
import multiprocessing as mp
from fop_build import build_one, list_dates

ROOTS = ["ETH", "BTC", "PAO", "ON", "RTM", "OZF", "OZT", "UB1", "OZC", "OZS", "OZW",
         "OZN", "OZB", "OG", "LO", "NQ", "ES"]   # 17 roots, 6 asset classes (all units IV-validated)
YEARS = ["2026", "2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018", "2017"]
WORKERS = 8

if __name__ == "__main__":
    os.sched_setaffinity(0, range(0, 16))
    tasks = []
    for y in YEARS:
        try:
            dates = list_dates(ROOTS[0], y)          # date prefixes are root-independent
        except Exception:
            dates = []
        for d in dates:
            for r in ROOTS:
                tasks.append((r, d, y))
    print(f"FOP pull: {len(tasks)} (root,date) tasks, {WORKERS} workers, recent-first", flush=True)
    t0 = time.time(); done = 0; rows = 0
    with mp.Pool(WORKERS) as pool:
        for n in pool.imap(build_one, tasks, chunksize=2):   # imap keeps recent-first order
            done += 1; rows += (n or 0)
            if done % 200 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(tasks)} tasks  {rows/1e6:.1f}M rows  "
                      f"[{el:.0f}s, {done/el:.1f} task/s, ETA {(len(tasks)-done)/(done/el)/3600:.1f}h]", flush=True)
    print(f"ALL ROOTS DONE: {done} tasks, {rows/1e6:.1f}M rows [{time.time()-t0:.0f}s]", flush=True)
