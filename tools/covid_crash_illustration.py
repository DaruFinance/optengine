"""March-2020 COVID crash stress ILLUSTRATION (descriptive, NOT an OOS/WFO claim).

The WFO corpus cannot make an out-of-sample statement about the March-2020 crash itself: the earliest
FO data is Jan-2020, so the crash falls inside the first 126-day in-sample warm-up of every root. To still
SHOW that the extended data captures the tail, we run three FIXED-parameter structures continuously through
2020-H1 (params chosen a priori, no IS tuning, delta-hedged to match the corpus engine), and report the
realized net P&L by month and the worst trade:
    short_straddle  -- short vol / short gamma+vega (should be destroyed by the vol spike)
    long_put_tail   -- long 10-delta OTM put, Universa-style crash convexity (should spike)
    long_straddle   -- long vol / long gamma+vega (should profit)
A weekly entry cadence (entry_gap=5) means "sell/buy this structure every week through the crash."
This is a stress illustration of the DATA, explicitly separated from the deflation/survivor analysis.

Reads fop_pertkr. Writes findings/covid_crash_illustration.csv. No lookahead: fixed structures, realized P&L.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FINDINGS, FOP_PERTKR

import os, sys
import numpy as np, pandas as pd
from optengine.position import LegSpec, run_structure
from optengine.data import greeks_store as G
from optengine.costs import OptionCostModel
from fop_finalize import ROOT_MULT, ROOT_FEE

FOP = FOP_PERTKR
OUT = FINDINGS
K = {"dte": (25, 35)}                        # fixed monthly tenor (a-priori, not tuned)
ROOTS = ["ES", "RTM", "OZB", "OZN", "BTC", "LO"]   # equity, rates, crypto, energy — all crashed/spiked Mar-2020

STRUCTS = {
    "short_straddle": [LegSpec(+1, -1, "atm", dte=K["dte"]), LegSpec(-1, -1, "atm", dte=K["dte"])],
    "long_put_tail":  [LegSpec(-1, +1, "delta", 0.10, dte=K["dte"])],
    "long_straddle":  [LegSpec(+1, +1, "atm", dte=K["dte"]), LegSpec(-1, +1, "atm", dte=K["dte"])],
}


def main():
    G._PK_DIR = FOP
    cost = OptionCostModel()
    rows = []
    for root in ROOTS:
        if not os.path.exists(os.path.join(FOP, f"{root}.parquet")):
            print(f"{root}: no chain", flush=True); continue
        G.clear_cache(); fr = G.build_array_frames(root)
        dates = sorted(d for d in fr.keys() if 20200101 <= int(d) <= 20200701)
        if len(dates) < 20:
            print(f"{root}: only {len(dates)} dates in 2020-H1, skip", flush=True); continue
        mult = float(ROOT_MULT.get(root, 100.0)); fee = float(ROOT_FEE.get(root, 1.5))
        for name, legs in STRUCTS.items():
            tr = run_structure(root, dates, legs, entry_gap=5, exit_dte=2, cost=cost,
                               delta_hedge=True, frames=fr, mult=mult, commission=fee)
            if tr is None or not len(tr):
                print(f"  {root} {name}: no trades", flush=True); continue
            tr = tr.copy(); tr["entry"] = tr["entry"].astype(str)
            tr["mo"] = tr["entry"].str.slice(0, 6)
            net = tr["net"].to_numpy()
            # Q1 (entries Jan-Mar) = into-the-crash; worst single trade
            q1 = tr[tr["mo"].isin(["202001", "202002", "202003"])]
            rows.append(dict(root=root, struct=name, n=len(tr),
                             net_total=round(net.sum(), 0), net_q1=round(q1["net"].sum(), 0),
                             worst_trade=round(net.min(), 0), best_trade=round(net.max(), 0),
                             mean_per_trade=round(net.mean(), 0), win=round((net > 0).mean(), 2)))
            by_mo = tr.groupby("mo")["net"].sum().round(0).to_dict()
            print(f"{root:4s} {name:14s}: n={len(tr):3d} net_total={net.sum():10.0f} "
                  f"net_Q1={q1['net'].sum():10.0f} worst={net.min():9.0f} | by-month {by_mo}", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "covid_crash_illustration.csv"), index=False)
    if len(R):
        print("\n=== crash-window (Q1-2020 entries) net P&L by vega sign ===", flush=True)
        sv = R[R.struct == "short_straddle"]["net_q1"].sum()
        lp = R[R.struct == "long_put_tail"]["net_q1"].sum()
        ls = R[R.struct == "long_straddle"]["net_q1"].sum()
        print(f"  short_straddle (short vol): {sv:+.0f}   <- should be deeply negative", flush=True)
        print(f"  long_put_tail  (long tail): {lp:+.0f}   <- should be positive", flush=True)
        print(f"  long_straddle  (long vol) : {ls:+.0f}   <- should be positive", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
