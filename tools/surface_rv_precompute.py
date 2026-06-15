"""Surface-RV precompute: for each (liquid instrument, day) fit the causal arb-checked SVI
surface and store each contract's RICHNESS = market MidImpliedVol − SVI-fitted IV (iv_KT).
This makes the surface load-bearing as a STRATEGY: surface-RV archetypes sell the richest /
buy the cheapest options vs the fitted surface. No-lookahead: the surface for day t is fit
from day t's quotes only (same causal discipline as the rest of the engine).

Output: richness/<ticker>.parquet with [asof_i, cp, Strike, expiry_i, richness], to be merged
into the array-frames by the surface-RV corpus. Parallel across instruments; modest cores so
it shares the box with the FO pull and other agents.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import PERTKR, RICHNESS

import os, sys, glob, time, argparse
import multiprocessing as mp
import numpy as np, pandas as pd
from optengine.data import greeks_store as G

PK = PERTKR
OUT = RICHNESS


def liquid_subset(n=0):
    fs = sorted(glob.glob(os.path.join(PK, "*.parquet")), key=os.path.getsize, reverse=True)
    return [os.path.basename(f)[:-8] for f in (fs[:n] if n else fs)]


def one_instrument(ticker):
    """Richness = market IV − the smooth per-(day, expiry) smile (weighted quadratic in
    log-moneyness, ATM-weighted). Same rich/cheap ranking as an SVI residual, ~1000× faster
    (numpy, no scipy/re-read). Causal: each day fit from that day's own quotes only."""
    fp = os.path.join(OUT, f"{ticker}.parquet")
    if os.path.exists(fp):
        return (ticker, 0)
    try:
        df = pd.read_parquet(os.path.join(PK, f"{ticker}.parquet"),
                             columns=["asof_i", "cp", "Strike", "expiry_i", "MidImpliedVol",
                                      "UnderLastMidPrice", "converged"])
    except Exception:
        return (ticker, -1)
    df = df[df["converged"].to_numpy(bool) & (df["MidImpliedVol"].to_numpy(float) > 0)
            & (df["Strike"].to_numpy(float) > 0) & (df["UnderLastMidPrice"].to_numpy(float) > 0)]
    if not len(df):
        return (ticker, 0)
    out = []
    for (di, ei), g in df.groupby(["asof_i", "expiry_i"], sort=False):
        if len(g) < 6:
            continue
        F = float(np.median(g["UnderLastMidPrice"].to_numpy(float)))
        k = np.log(g["Strike"].to_numpy(float) / F)
        iv = g["MidImpliedVol"].to_numpy(float)
        w = np.exp(-(k / 0.5) ** 2)                    # ATM-weight (downweight noisy wings)
        try:
            c = np.polyfit(k, iv, 2, w=w)              # smooth smile: iv ≈ a + b·k + c·k²
        except Exception:
            continue
        rich = iv - np.polyval(c, k)                   # >0 = richer than the smile (sell candidate)
        sub = g[["asof_i", "cp", "Strike", "expiry_i"]].copy(); sub["richness"] = rich
        out.append(sub)
    if not out:
        return (ticker, 0)
    os.makedirs(OUT, exist_ok=True)
    pd.concat(out, ignore_index=True).to_parquet(fp + ".tmp", compression="zstd", index=False)
    os.replace(fp + ".tmp", fp)
    return (ticker, len(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    G._PK_DIR = PK
    insts = liquid_subset(a.n)
    print(f"surface-RV precompute: {len(insts)} liquid instruments, {a.workers} workers", flush=True)
    t0 = time.time(); done = 0
    with mp.Pool(a.workers, maxtasksperchild=2) as pool:
        for tk, nd in pool.imap_unordered(one_instrument, insts):
            done += 1
            if done % 5 == 0 or done == len(insts):
                print(f"  {done}/{len(insts)} ({tk}:{nd} days) [{time.time()-t0:.0f}s]", flush=True)
    print(f"DONE {time.time()-t0:.0f}s -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
