"""Gamma-scalp hedge-frequency robustness — is the null an artifact of DAILY delta-hedging?

The delta-hedged option return IS the discrete gamma-scalp P&L, and its level depends on how often you
re-hedge: hedging more often cuts path variance but pays more slippage; hedging less often is cheaper but
noisier. If "the spread eats it" only held at one hedge cadence the result would be fragile. So we re-run the
canonical delta-hedged straddle (both signs) at hedge_every ∈ {1,2,3,5} trading days on a liquid slice and
re-deflate at each frequency. `tools/hedge_freq.py` (uses the engine's new behaviour-preserving hedge_every).
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FINDINGS, PERTKR

import os, sys, glob, time
import multiprocessing as mp
import numpy as np, pandas as pd
from scipy import stats as ss
from optengine.position import LegSpec
from optengine.wfo import wfo
from optengine.data import greeks_store as G
from corpus import _stats, dsr_pvalue, GRID, EXITS, MIN_TRADES

PK = PERTKR
OUT = FINDINGS
FREQS = [1, 2, 3, 5]


def _d(k):
    return k["dte"]


def short_straddle(k):
    return [LegSpec(+1, -1, "atm", dte=_d(k)), LegSpec(-1, -1, "atm", dte=_d(k))]


def long_straddle(k):
    return [LegSpec(+1, +1, "atm", dte=_d(k)), LegSpec(-1, +1, "atm", dte=_d(k))]


ARCHES = {"short_straddle": short_straddle, "long_straddle": long_straddle}


def liquid_subset(n=60):
    fs = sorted(glob.glob(os.path.join(PK, "*.parquet")), key=os.path.getsize, reverse=True)
    return [os.path.basename(f)[:-8] for f in fs[:n]]


def _task(args):
    ticker, dates = args
    G.clear_cache(); G._PK_DIR = PK
    frames = G.build_array_frames(ticker)
    if not frames:
        return []
    rows = []
    for k in FREQS:
        for an, fn in ARCHES.items():
            try:
                oos, wins, _ = wfo(ticker, dates, fn, GRID, exit_grid=EXITS, obj="rrr",
                                   delta_hedge=True, frames=frames, hedge_every=k)
            except Exception as e:
                rows.append(dict(instrument=ticker, archetype=an, hedge_every=k, n=0, error=repr(e)[:70])); continue
            st = _stats(ticker, an, oos, wins); st["hedge_every"] = k
            rows.append(st)
    G.clear_cache()
    return rows


def main():
    insts = liquid_subset(60)
    dates = sorted(d for y in range(2017, 2027) for d in G.available_dates(y))
    print(f"hedge-frequency sweep: {len(insts)} names x {len(ARCHES)} archetypes x {len(FREQS)} freqs", flush=True)
    t0 = time.time(); allrows = []
    with mp.Pool(6, maxtasksperchild=1) as pool:
        for j, rows in enumerate(pool.imap_unordered(_task, [(i, dates) for i in insts])):
            allrows.extend(rows)
            if (j + 1) % 15 == 0:
                print(f"  {j+1}/{len(insts)} [{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(allrows); df.to_csv(os.path.join(OUT, "hedge_freq_raw.csv"), index=False)
    print(f"\nper hedge frequency (FDR+DSR deflation within each):", flush=True)
    out = []
    for k in FREQS:
        s = df[(df["hedge_every"] == k) & (df["n"] >= MIN_TRADES) & df.get("sr_trade", pd.Series(dtype=float)).notna()].copy()
        if not len(s):
            continue
        s["p"] = 2 * ss.t.sf(s["t"].abs(), s["n"] - 1)
        m = len(s); order = np.argsort(s["p"].to_numpy()); ps = s["p"].to_numpy()[order]
        bh = ps <= (np.arange(1, m + 1) / m) * 0.10
        kmax = (np.where(bh)[0].max() + 1) if bh.any() else 0
        fp = np.zeros(m, bool)
        if kmax:
            fp[order[:kmax]] = True
        trial = s["sr_trade"].to_numpy()
        dsr = np.array([dsr_pvalue(r.sr_trade, r.skew, r.kurt, r.n, trial)[0] for r in s.itertuples()])
        surv = int((fp & (dsr > 0.95)).sum())
        npos = int((s["oos_net"] > 0).sum())
        out.append(dict(hedge_every=k, n=len(s), net_positive=npos, survivors=surv,
                        median_net=round(float(s["oos_net"].median())), median_t=round(float(s["t"].median()), 2)))
        print(f"  hedge_every={k}d: {len(s)} strats | net+ {npos}/{len(s)} | FDR&DSR survivors={surv} | "
              f"median net ${s['oos_net'].median():,.0f} | median t {s['t'].median():.2f}", flush=True)
    pd.DataFrame(out).to_csv(os.path.join(OUT, "hedge_freq_curve.csv"), index=False)
    print(f"\nDONE {time.time()-t0:.0f}s -> hedge_freq_curve.csv", flush=True)


if __name__ == "__main__":
    main()
