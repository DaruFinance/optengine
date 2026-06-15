"""Variance-swap (log-strip) corpus — the MODEL-FREE variance risk premium harvest a referee will
ask for beyond the ATM-straddle proxy. A short variance swap is replicated (Demeterfi-Derman-Kamal-Zou
/ the VIX construction) by a static strip of OTM options weighted ∝ ΔK/K²: OTM puts below spot, OTM
calls above, ATM both. Selling that strip = short variance; buying it = long variance. We run it through
the SAME WFO + real bid/ask + daily delta-hedge + deflation as the main corpus, and report GROSS (mid)
vs NET (full taker cross) so the spread tax on the *correct* VRP instrument — not just a straddle — is
explicit. The strip pays the spread on ~10 legs per cycle, so if anything this is the harshest cost test.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FINDINGS, PERTKR

import os, sys, glob, time
import multiprocessing as mp
import numpy as np, pandas as pd
from scipy import stats as ss
from optengine.position import LegSpec, run_structure
from optengine.wfo import wfo
from optengine.data import greeks_store as G
from corpus import _stats, dsr_pvalue, GRID, EXITS, MIN_TRADES

PK = PERTKR
OUT = FINDINGS
LEDG = os.path.join(OUT, "vs_ledgers")
LADDER = [0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20]   # moneyness grid; ΔK≈const → w∝1/K²


def _d(k):
    return k["dte"]


def short_logstrip(k):
    """Short the 1/K²-weighted OTM strip (model-free short variance swap)."""
    legs = []
    for m in LADDER:
        w = round(1.0 / (m * m), 3)                       # ΔK/K² var-swap weight (ATM-heavy)
        if abs(m - 1.0) < 1e-9:
            legs.append(LegSpec(+1, -w, "moneyness", 1.0, dte=_d(k)))   # short ATM call
            legs.append(LegSpec(-1, -w, "moneyness", 1.0, dte=_d(k)))   # short ATM put
        else:
            cp = -1 if m < 1.0 else +1                    # OTM put below / OTM call above
            legs.append(LegSpec(cp, -w, "moneyness", m, dte=_d(k)))
    return legs


def long_logstrip(k):
    """Long the strip (model-free long variance swap)."""
    return [LegSpec(s.cp, -s.qty, s.sel, s.sel_val, s.dte) for s in short_logstrip(k)]


VS_ALL = ["short_logstrip", "long_logstrip"]
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
    rows, ledgers = [], []
    for arch in VS_ALL:
        fn = getattr(_THIS, arch)
        try:
            oos, wins, _ = wfo(ticker, dates, fn, GRID, exit_grid=EXITS, obj="rrr",
                               delta_hedge=True, frames=frames, capture=1.0)         # NET
            og, _, _ = wfo(ticker, dates, fn, GRID, exit_grid=EXITS, obj="rrr",
                           delta_hedge=True, frames=frames, capture=0.0)             # GROSS (mid)
        except Exception as e:
            rows.append(dict(instrument=ticker, archetype=arch, n=0, error=repr(e)[:90])); continue
        st = _stats(ticker, arch, oos, wins)
        st["gross_net"] = round(float(og["net"].sum())) if (og is not None and len(og)) else 0.0
        rows.append(st)
        if oos is not None and len(oos):
            o = oos.copy(); o["instrument"] = ticker; o["archetype"] = arch; ledgers.append(o)
    if ledgers:
        try:
            pd.concat(ledgers, ignore_index=True).to_parquet(os.path.join(LEDG, f"{ticker}.parquet"),
                                                             compression="zstd", index=False)
        except Exception:
            pass
    G.clear_cache()
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=140); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(LEDG, exist_ok=True)
    insts = liquid_subset(a.n)
    dates = sorted(d for y in range(2017, 2027) for d in G.available_dates(y))
    print(f"varswap corpus: {len(insts)} liquid instruments x {len(VS_ALL)} archetypes "
          f"(GROSS+NET) [{a.workers} workers]", flush=True)
    t0 = time.time(); allrows = []
    with mp.Pool(a.workers, maxtasksperchild=1) as pool:
        for k, rows in enumerate(pool.imap_unordered(_task, [(i, dates) for i in insts])):
            allrows.extend(rows)
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(insts)} [{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(allrows)
    df.to_csv(os.path.join(OUT, "varswap_raw.csv"), index=False)
    s = df[(df["n"] >= MIN_TRADES) & df.get("sr_trade", pd.Series(dtype=float)).notna()].copy()
    if len(s):
        s["p"] = 2 * ss.t.sf(s["t"].abs(), s["n"] - 1)
        m = len(s); order = np.argsort(s["p"].to_numpy()); ps = s["p"].to_numpy()[order]
        bh = ps <= (np.arange(1, m + 1) / m) * 0.10
        kmax = np.where(bh)[0].max() + 1 if bh.any() else 0
        fp = np.zeros(m, bool)
        if kmax:
            fp[order[:kmax]] = True
        s["fdr_pass"] = fp
        trial = s["sr_trade"].to_numpy()
        s["dsr"] = [dsr_pvalue(r.sr_trade, r.skew, r.kurt, r.n, trial)[0] for r in s.itertuples()]
        s["dsr_pass"] = s["dsr"] > 0.95
        s = s.sort_values("t", ascending=False)
    s.to_csv(os.path.join(OUT, "varswap_strategies.csv"), index=False)
    print(f"\nDONE {time.time()-t0:.0f}s. {len(df)} cells, {len(s)} with >={MIN_TRADES} trades.", flush=True)
    if len(s):
        gpos = int((s["gross_net"] > 0).sum()); npos = int((s["oos_net"] > 0).sum())
        surv = int((s["fdr_pass"] & s["dsr_pass"]).sum())
        print(f"  GROSS-positive: {gpos}/{len(s)} | NET-positive: {npos}/{len(s)} | FDR&DSR survivors: {surv}", flush=True)
        print(s.head(10)[["instrument", "archetype", "n", "gross_net", "oos_net", "win", "t", "dsr"]].to_string(index=False))


if __name__ == "__main__":
    main()
