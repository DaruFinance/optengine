"""Box spread / put-call-parity financing family — the one named strategy corner with zero prior coverage.

A long BOX = bull call spread (long call K1, short call K2) + bear put spread (long put K2, short put K1),
K1<K2. Its payoff at expiry is the constant (K2-K1) regardless of the underlying — a synthetic zero-coupon
bond, and the cleanest put-call-parity / financing trade. Its only "edge" is a parity mispricing: if the box
trades below PV(K2-K1) you lock a financing arb. We test whether ANY such edge survives the 4-leg bid/ask. The
box is structurally delta-neutral (long forward at K1 minus forward at K2), so no delta-hedge. Same WFO + real
bid/ask + BH-FDR/DSR deflation; GROSS vs NET reported. `tools/boxspread.py`.
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
LO, HI = 0.95, 1.05


def _d(k):
    return k["dte"]


def long_box(k):
    """Long box: +call(K1) -call(K2) -put(K1) +put(K2). Locks (K2-K1) at expiry."""
    return [LegSpec(+1, +1, "moneyness", LO, dte=_d(k)), LegSpec(+1, -1, "moneyness", HI, dte=_d(k)),
            LegSpec(-1, -1, "moneyness", LO, dte=_d(k)), LegSpec(-1, +1, "moneyness", HI, dte=_d(k))]


def short_box(k):
    return [LegSpec(s.cp, -s.qty, s.sel, s.sel_val, s.dte) for s in long_box(k)]


ARCHES = {"long_box": long_box, "short_box": short_box}
_THIS = sys.modules[__name__]


def liquid_subset(n=140):
    fs = sorted(glob.glob(os.path.join(PK, "*.parquet")), key=os.path.getsize, reverse=True)
    return [os.path.basename(f)[:-8] for f in fs[:n]]


def _task(args):
    ticker, dates = args
    G.clear_cache(); G._PK_DIR = PK
    frames = G.build_array_frames(ticker)
    if not frames:
        return []
    rows = []
    for an, fn in ARCHES.items():
        try:
            oos, wins, _ = wfo(ticker, dates, fn, GRID, exit_grid=EXITS, obj="rrr",
                               delta_hedge=False, frames=frames, capture=1.0)        # NET
            og, _, _ = wfo(ticker, dates, fn, GRID, exit_grid=EXITS, obj="rrr",
                           delta_hedge=False, frames=frames, capture=0.0)            # GROSS
        except Exception as e:
            rows.append(dict(instrument=ticker, archetype=an, n=0, error=repr(e)[:80])); continue
        st = _stats(ticker, an, oos, wins)
        st["gross_net"] = round(float(og["net"].sum())) if (og is not None and len(og)) else 0.0
        rows.append(st)
    G.clear_cache()
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=140); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    insts = liquid_subset(a.n)
    dates = sorted(d for y in range(2017, 2027) for d in G.available_dates(y))
    print(f"box-spread corpus: {len(insts)} liquid names x {len(ARCHES)} archetypes (GROSS+NET)", flush=True)
    t0 = time.time(); allrows = []
    with mp.Pool(a.workers, maxtasksperchild=1) as pool:
        for j, rows in enumerate(pool.imap_unordered(_task, [(i, dates) for i in insts])):
            allrows.extend(rows)
            if (j + 1) % 20 == 0:
                print(f"  {j+1}/{len(insts)} [{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(allrows); df.to_csv(os.path.join(OUT, "boxspread_raw.csv"), index=False)
    s = df[(df["n"] >= MIN_TRADES) & df.get("sr_trade", pd.Series(dtype=float)).notna()].copy()
    if len(s):
        s["p"] = 2 * ss.t.sf(s["t"].abs(), s["n"] - 1)
        m = len(s); order = np.argsort(s["p"].to_numpy()); ps = s["p"].to_numpy()[order]
        bh = ps <= (np.arange(1, m + 1) / m) * 0.10
        kmax = (np.where(bh)[0].max() + 1) if bh.any() else 0
        fp = np.zeros(m, bool)
        if kmax:
            fp[order[:kmax]] = True
        s["fdr_pass"] = fp
        trial = s["sr_trade"].to_numpy()
        s["dsr"] = [dsr_pvalue(r.sr_trade, r.skew, r.kurt, r.n, trial)[0] for r in s.itertuples()]
        s["dsr_pass"] = s["dsr"] > 0.95
        s = s.sort_values("t", ascending=False)
    s.to_csv(os.path.join(OUT, "boxspread_strategies.csv"), index=False)
    print(f"\nDONE {time.time()-t0:.0f}s. {len(df)} cells, {len(s)} with >={MIN_TRADES} trades.", flush=True)
    if len(s):
        print(f"  GROSS-positive: {int((s['gross_net']>0).sum())}/{len(s)} | NET-positive: {int((s['oos_net']>0).sum())}/{len(s)} | "
              f"FDR&DSR survivors: {int((s['fdr_pass']&s['dsr_pass']).sum())}", flush=True)
        print(s.head(8)[["instrument", "archetype", "n", "gross_net", "oos_net", "t", "dsr"]].to_string(index=False))


if __name__ == "__main__":
    main()
