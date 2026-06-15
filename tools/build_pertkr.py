"""Transpose the daily-zip Greeks store -> per-ticker parquet (one file/ticker, all
dates, lean engine-ready columns). Kills the per-ticker zip re-open bottleneck
(112 ms/day -> ~constant). Parallel by strided day-blocks on cores 0-15, RAM-safe
(one zip's rows in memory per worker at a time). One-time build; never deletes inputs.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import PERTKR, SURFACE_PANEL

import sys, os, glob, time, zipfile, argparse
import multiprocessing as mp
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
from optengine.data import greeks_store as G

OUT = PERTKR
SH = os.path.join(OUT, "_shards")
PANEL = SURFACE_PANEL

IDX_ETF = ["SPXW", "SPX", "NDX", "RUT", "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK",
           "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "XLC", "TLT", "HYG", "LQD",
           "GLD", "SLV", "USO", "UNG", "EEM", "FXI", "VXX", "UVXY", "SMH", "XBI", "KRE",
           "GDX", "EWZ", "XOP", "ARKK", "TQQQ", "SQQQ"]

OUTCOLS = ["asof_i", "expiry_i", "cp", "Strike", "DaysToMaturity", "MidImpliedVol",
           "MidDelta", "MidGamma", "MidTheta", "MidVega", "MidTheoPrice", "LastBidPrice",
           "LastMidPrice", "LastAskPrice", "UnderLastMidPrice", "converged"]


def universe(n_single=180):
    sp = pd.read_parquet(PANEL, columns=["underlying", "total_oi"])
    rank = sp.groupby("underlying")["total_oi"].median().sort_values(ascending=False)
    top = [t for t in rank.index if isinstance(t, str)][:n_single]
    uni = []
    for t in IDX_ETF + top:
        if t not in uni:
            uni.append(t)
    return uni


def _process(df, d):
    n = len(df)
    def g(c):
        return (pd.to_numeric(df[c], errors="coerce").astype("float32").values
                if c in df else np.full(n, np.nan, "float32"))
    exp = (pd.to_numeric(df["Expiration"], errors="coerce").fillna(0).astype("int32").values
           if "Expiration" in df else np.zeros(n, "int32"))
    out = pd.DataFrame({
        "asof_i": np.full(n, int(pd.Timestamp(d).strftime("%Y%m%d")), "int32"),
        "expiry_i": exp,
        "cp": np.where(df["CallPut"].astype(str).str.upper().str[0] == "C", 1, -1).astype("int8"),
        "Strike": g("Strike"), "DaysToMaturity": g("DaysToMaturity"),
        "MidImpliedVol": g("MidImpliedVol"), "MidDelta": g("MidDelta"),
        "MidGamma": g("MidGamma"), "MidTheta": g("MidTheta"), "MidVega": g("MidVega"),
        "MidTheoPrice": g("MidTheoPrice"), "LastBidPrice": g("LastBidPrice"),
        "LastMidPrice": g("LastMidPrice"), "LastAskPrice": g("LastAskPrice"),
        "UnderLastMidPrice": g("UnderLastMidPrice"),
        "converged": (df["ImpliedVolConvergence"].astype(str).str.startswith("Converged").values
                      if "ImpliedVolConvergence" in df else np.zeros(n, bool)),
    })
    return out[OUTCOLS]


def _worker(args):
    block_id, dates, tickers = args
    writers = {}
    for d in dates:
        zp = G._zip_path(d)
        if not os.path.exists(zp):
            continue
        try:
            z = zipfile.ZipFile(zp)
        except Exception:
            continue
        try:
            names = set(z.namelist())
            for t in tickers:
                m = f"{t[0].upper()}/{t}.csv"
                if m not in names:
                    continue
                try:
                    with z.open(m) as fh:
                        df = pd.read_csv(fh, usecols=lambda c: c in G._USECOLS)
                except Exception:
                    continue
                if not len(df):
                    continue
                tbl = pa.Table.from_pandas(_process(df, d), preserve_index=False)
                if t not in writers:
                    writers[t] = pq.ParquetWriter(
                        os.path.join(SH, f"{t}__b{block_id:02d}.parquet"),
                        tbl.schema, compression="zstd")
                writers[t].write_table(tbl)
        finally:
            z.close()
    for w in writers.values():
        w.close()
    return block_id, len(dates)


def merge(tickers):
    done = 0
    for t in tickers:
        shards = sorted(glob.glob(os.path.join(SH, f"{t}__b*.parquet")))
        if not shards:
            continue
        df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
        df.sort_values(["asof_i", "expiry_i", "cp", "Strike"], inplace=True, kind="stable")
        df.to_parquet(os.path.join(OUT, f"{t}.parquet"), compression="zstd", index=False)
        done += 1
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n_single", type=int, default=180)
    a = ap.parse_args()
    os.makedirs(SH, exist_ok=True)
    uni = universe(a.n_single)
    years = range(2017, 2027)
    if a.test:
        uni, years = ["SPXW", "SPY", "AAPL"], [2024]
    dates = sorted(d for y in years for d in G.available_dates(y))
    W = a.workers
    blocks = [(i, dates[i::W], uni) for i in range(W)]   # strided: each zip opened once total
    print(f"universe={len(uni)} tickers, dates={len(dates)} ({dates[0]}..{dates[-1]}), workers={W}", flush=True)
    t0 = time.time()
    with mp.Pool(W, maxtasksperchild=1) as pool:
        for bid, nd in pool.imap_unordered(_worker, blocks):
            print(f"  block {bid:02d} extracted ({nd} strided days) [{time.time()-t0:.0f}s]", flush=True)
    print(f"extract done {time.time()-t0:.0f}s; merging per ticker...", flush=True)
    n = merge(uni)
    sz = sum(os.path.getsize(p) for p in glob.glob(os.path.join(OUT, "*.parquet"))) / 1e9
    print(f"DONE: merged {n} tickers -> {OUT} ({sz:.1f} GB) [{time.time()-t0:.0f}s total]", flush=True)


if __name__ == "__main__":
    main()
