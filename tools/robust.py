"""Robustness on corpus survivors: re-run each at cost stress (spread_capture 0.5/1.0/1.5)
and check OOS sub-period stability (first vs second half of trades). Flags fragile edges.
Reads findings/corpus_strategies.csv + pertkr; groups survivors by instrument to reuse the
converged-chain frames. Runs on cores 0-15 (launch under taskset)."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FINDINGS

import sys, os, argparse
import numpy as np, pandas as pd
from optengine.wfo import wfo
from optengine import archetypes as A
from optengine.data import greeks_store as G
from optengine.position import _converged, to_arrays

OUT = FINDINGS
GRID = {"dte": [(25, 35), (40, 55)]}
EXITS = (2, 5, 10)


def _frames(inst, dates):
    return G.build_array_frames(inst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=60, help="if no pass flags, test top-N by t")
    a = ap.parse_args()
    src = os.path.join(OUT, "corpus_deflated.csv")
    if not os.path.exists(src):
        src = os.path.join(OUT, "corpus_strategies.csv")
    s = pd.read_csv(src)
    surv = s[s["t"] > 2.5].copy()            # strong positive net-of-cost survivors
    if not len(surv):
        surv = s.sort_values("t", ascending=False).head(a.top).copy()
    dates = sorted(d for y in range(2017, 2027) for d in G.available_dates(y))
    print(f"robustness on {len(surv)} strategies across {surv['instrument'].nunique()} instruments", flush=True)
    rows = []
    for inst, grp in surv.groupby("instrument"):
        fr = _frames(inst, dates)
        for r in grp.itertuples():
            rec = {"instrument": inst, "archetype": r.archetype, "base_net": getattr(r, "oos_net", np.nan)}
            for cap in (0.5, 1.0, 1.5):
                oos, _w, _n = wfo(inst, dates, getattr(A, r.archetype), GRID, exit_grid=EXITS,
                                  obj="rrr", delta_hedge=True, capture=cap, frames=fr)
                ok = oos is not None and len(oos)
                rec[f"net_cap{cap}"] = round(float(oos["net"].sum())) if ok else 0
                if cap == 1.0 and ok:
                    h = oos.sort_values("exit"); n = len(h) // 2
                    rec["half1"] = round(float(h["net"].iloc[:n].sum()))
                    rec["half2"] = round(float(h["net"].iloc[n:].sum()))
                    rec["n"] = len(h)
            rec["robust"] = bool(rec.get("net_cap1.5", 0) > 0 and rec.get("half1", 0) > 0 and rec.get("half2", 0) > 0)
            rows.append(rec)
        G.clear_cache()
    rb = pd.DataFrame(rows)
    rb.to_csv(os.path.join(OUT, "robustness.csv"), index=False)
    nr = int(rb["robust"].sum()) if len(rb) else 0
    print(f"\nDONE. tested={len(rb)}  ROBUST (survives 1.5x cost AND both OOS halves >0)={nr}")
    if len(rb):
        print(rb.sort_values("net_cap1.5", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
