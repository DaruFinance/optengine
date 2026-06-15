"""Conditioned vol-selling — does TIMING the VRP harvest improve short-vol archetypes
vs unconditional selling? For each instrument, build array-frames + causal surface-panel
signals (shifted 1 day, no-lookahead), then run each short-vol archetype unconditional
and conditioned on: vrp_pos (implied>realized), contango (term_slope>0), and both. This
is the novelty contribution beyond the unconditional survival map. Cores 0-15.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FINDINGS, PERTKR

import sys, os, glob, argparse
import multiprocessing as mp
import numpy as np, pandas as pd
from optengine.wfo import wfo
from optengine import archetypes as A, signals as S
from optengine.data import greeks_store as G
from optengine.position import _converged, to_arrays

OUT = FINDINGS
GRID = {"dte": [(25, 35), (40, 55)]}
EXITS = (2, 5, 10)
ARCHES = ["short_straddle", "short_strangle", "iron_condor", "put_write", "put_credit_spread"]


def _conditions(feat):
    if feat is None or not len(feat):
        return {}
    idx = feat.index.to_numpy()
    v = feat["vrp30"].to_numpy(); ts = feat["term_slope"].to_numpy()
    return {"vrp_pos": set(idx[np.where(v > 0)[0]]),
            "contango": set(idx[np.where(ts > 0)[0]]),
            "vrp_pos_contango": set(idx[np.where((v > 0) & (ts > 0))[0]])}


def _net(oos):
    return (round(float(oos["net"].sum())), len(oos)) if (oos is not None and len(oos)) else (0, 0)


def _task(args):
    inst, dates = args
    G.clear_cache()
    fr = G.build_array_frames(inst)
    try:
        conds = _conditions(S.features(inst))
    except Exception:
        conds = {}
    G.clear_cache()
    rows = []
    for arch in ARCHES:
        build = getattr(A, arch)
        try:
            net, n = _net(wfo(inst, dates, build, GRID, exit_grid=EXITS, obj="rrr",
                             delta_hedge=True, frames=fr)[0])
        except Exception:
            net, n = 0, 0
        rows.append(dict(instrument=inst, archetype=arch, condition="(none)", n=n, oos_net=net))
        for cname, days in conds.items():
            if not days:
                continue
            try:
                net2, n2 = _net(wfo(inst, dates, build, GRID, exit_grid=EXITS, obj="rrr",
                                   delta_hedge=True, frames=fr, entry_days=days)[0])
            except Exception:
                net2, n2 = 0, 0
            rows.append(dict(instrument=inst, archetype=arch, condition=cname, n=n2, oos_net=net2))
    G.clear_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    insts = sorted(os.path.basename(p)[:-8] for p in glob.glob(f"{PERTKR}/*.parquet"))
    if a.limit:
        insts = insts[:a.limit]
    dates = sorted(d for y in range(2017, 2027) for d in G.available_dates(y))
    allr = []
    with mp.Pool(a.workers, maxtasksperchild=1) as pool:
        for rows in pool.imap_unordered(_task, [(i, dates) for i in insts]):
            allr.extend(rows)
    df = pd.DataFrame(allr)
    df.to_csv(os.path.join(OUT, "conditioned.csv"), index=False)
    print("mean OOS net per archetype x condition (does conditioning help?):")
    print(df.groupby(["archetype", "condition"])["oos_net"].agg(["mean", "median", "size"]).round(0).to_string())


if __name__ == "__main__":
    main()
