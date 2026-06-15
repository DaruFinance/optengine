"""Futures-options corpus: the 35 archetypes x FO roots (ES/NQ/LO/OG/OZN/OZB/BTC/ETH), same
WFO + real-bid/ask + delta-hedge + no-lookahead engine as OPRA, with the per-root contract
multiplier and CME fee. Tests whether the OPRA two-sided-tax / no-edge finding GENERALIZES to
futures options + CME crypto. Writes fop_corpus_strategies.csv + fop_ledgers/ (same schema as
the OPRA corpus, so deflate/vega_breakdown/portfolio run on it unchanged).
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FINDINGS, FOP_PERTKR

import sys, os, glob, time
import multiprocessing as mp
import numpy as np, pandas as pd
from scipy import stats as ss
from optengine.wfo import wfo
from optengine import archetypes as A
from optengine.data import greeks_store as G
from corpus import _stats, dsr_pvalue, GRID, EXITS, MIN_TRADES
from fop_finalize import ROOT_MULT, ROOT_FEE

FOP = FOP_PERTKR
OUT = FINDINGS
LEDG = os.path.join(OUT, "fop_ledgers")


def roots():
    return sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(FOP, "*.parquet")))


def _task(root):
    G._PK_DIR = FOP; G.clear_cache()          # point the array-frame loader at the FO chains
    frames = G.build_array_frames(root)
    dates = sorted(frames.keys())
    mult = float(ROOT_MULT.get(root, 100.0)); fee = float(ROOT_FEE.get(root, 1.5))
    rows, ledgers = [], []
    for arch in A.ALL:
        try:
            oos, wins, _nc = wfo(root, dates, getattr(A, arch), GRID, exit_grid=EXITS, is_len=126,
                                 oos_len=63, obj="rrr", delta_hedge=True, frames=frames,
                                 mult=mult, commission=fee)
        except Exception as e:
            rows.append(dict(instrument=root, archetype=arch, n=0, error=repr(e)[:80])); continue
        rows.append(_stats(root, arch, oos, wins))
        if oos is not None and len(oos):
            o = oos.copy(); o["instrument"] = root; o["archetype"] = arch; ledgers.append(o)
    if ledgers:
        try:
            pd.concat(ledgers, ignore_index=True).to_parquet(
                os.path.join(LEDG, f"{root}.parquet"), compression="zstd", index=False)
        except Exception:
            pass
    G.clear_cache()
    return rows


def main():
    os.makedirs(LEDG, exist_ok=True)
    rs = roots()
    print(f"FOP corpus: {len(rs)} roots {rs} x {len(A.ALL)} archetypes = {len(rs)*len(A.ALL)} cells", flush=True)
    t0 = time.time(); allrows = []
    with mp.Pool(min(6, max(1, len(rs))), maxtasksperchild=1) as pool:
        for rows in pool.imap_unordered(_task, rs):
            allrows.extend(rows)
            print(f"  {len(allrows)//len(A.ALL)}/{len(rs)} roots [{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(allrows); df.to_csv(os.path.join(OUT, "fop_corpus_raw.csv"), index=False)
    s = df[(df["n"] >= MIN_TRADES) & df.get("sr_trade", pd.Series(dtype=float)).notna()].copy()
    if len(s):
        s["p"] = 2 * ss.t.sf(s["t"].abs(), s["n"] - 1)
        m = len(s)
        order = np.argsort(s["p"].to_numpy()); ps = s["p"].to_numpy()[order]
        bh = ps <= (np.arange(1, m + 1) / m) * 0.10
        kmax = np.where(bh)[0].max() + 1 if bh.any() else 0
        fp = np.zeros(m, bool)
        if kmax:
            fp[order[:kmax]] = True
        s["fdr_pass"] = fp
        trial = s["sr_trade"].to_numpy()
        dv = [dsr_pvalue(r.sr_trade, r.skew, r.kurt, r.n, trial) for r in s.itertuples()]
        s["dsr"] = [d[0] for d in dv]; s["dsr_pass"] = s["dsr"] > 0.95
        s = s.sort_values("t", ascending=False)
    s.to_csv(os.path.join(OUT, "fop_corpus_strategies.csv"), index=False)
    print(f"\nDONE {time.time()-t0:.0f}s. {len(df)} cells, {len(s)} with >={MIN_TRADES} OOS trades.")
    if len(s):
        print(f"  net-positive: {int((s['oos_net']>0).sum())}/{len(s)} | FDR&DSR survivors: {int((s['fdr_pass']&s['dsr_pass']).sum())}")
        print(s.head(12)[["instrument", "archetype", "n", "oos_net", "win", "t", "sharpe_ann", "dsr"]].to_string(index=False))


if __name__ == "__main__":
    main()
