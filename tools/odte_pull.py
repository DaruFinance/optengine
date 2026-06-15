"""Pull a focused intraday ES short-dated dataset (the 0DTE-adjacent arm). Reuses the validated intraday
distiller (odte_intraday.pull_one) over a date range, parallel, writes one parquet per root with the full
intraday ladder + a date column. Empirical expiry (last-quote-date per contract) is computed at finalize.
Pinned to cores 0-15, modest workers + nice so it shares bandwidth with the running daily FO pull.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import ODTE_RAW

import os, sys, time, argparse
import multiprocessing as mp
import pandas as pd
from odte_intraday import pull_one
from fop_build import list_dates

OUT = ODTE_RAW


def _one(args):
    root, date, year = args
    try:
        rows = pull_one(root, date, year)
    except Exception:
        rows = []
    return (date, rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="E1A")
    ap.add_argument("--years", default="2025")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--weekdays", default="")          # e.g. "0" = Mondays only (faster: only expiry days)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.sched_setaffinity(0, range(0, 16))
    import datetime as _dt
    wds = set(int(x) for x in a.weekdays.split(",")) if a.weekdays else None
    tasks = []
    for y in a.years.split(","):
        try:
            dates = list_dates(a.root, y)
        except Exception:
            dates = []
        for d in dates:
            if wds is not None:
                wd = _dt.date(int(d[:4]), int(d[4:6]), int(d[6:8])).weekday()
                if wd not in wds:
                    continue
            tasks.append((a.root, d, y))
    print(f"odte pull {a.root}: {len(tasks)} (date) tasks, {a.workers} workers", flush=True)
    t0 = time.time(); allrows = []; done = 0
    with mp.Pool(a.workers) as pool:
        for date, rows in pool.imap_unordered(_one, tasks):
            done += 1
            for r in rows:
                allrows.append((int(date),) + r)
            if done % 25 == 0:
                print(f"  {done}/{len(tasks)}  {len(allrows)} rows  [{time.time()-t0:.0f}s]", flush=True)
    if not allrows:
        print("NO DATA"); return
    df = pd.DataFrame(allrows, columns=["date", "snap", "cp", "Strike", "Month", "ExpYear", "bid", "ask"])
    fp = os.path.join(OUT, f"{a.root}.parquet")
    df.to_parquet(fp, compression="zstd", index=False)
    print(f"DONE {time.time()-t0:.0f}s: {len(df)} intraday rows, {df.date.nunique()} dates -> {fp}", flush=True)


if __name__ == "__main__":
    main()
