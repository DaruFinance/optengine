"""Surface-RV corpus: tests the vol-surface relative-value strategy family the engine's SVI
surface enables but never traded. Archetypes select by RICHNESS (market IV − SVI-fit IV,
precomputed causally) instead of delta: sell the richest / buy the cheapest options vs the
fitted surface, plus call/put-side rich-vs-cheap RV spreads. Same WFO + real bid/ask + daily
delta-hedge + deflation as the main corpus. Runs on the liquid subset that has a richness file.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FINDINGS, PERTKR, RICHNESS

import os, sys, glob, time
import multiprocessing as mp
import numpy as np, pandas as pd
from scipy import stats as ss
from optengine.position import LegSpec, run_structure
from optengine.wfo import wfo
from optengine.data import greeks_store as G
from corpus import _stats, dsr_pvalue, GRID, EXITS, MIN_TRADES

PK = PERTKR
RICH = RICHNESS
OUT = FINDINGS
LEDG = os.path.join(OUT, "rv_ledgers")


def _d(k):
    return k["dte"]


# --- surface-RV archetypes (sel = 'rich' richest-vs-surface / 'cheap' cheapest) ---
def surface_short_rich_strangle(k):                       # sell the richest call + richest put
    return [LegSpec(+1, -1, "rich", dte=_d(k)), LegSpec(-1, -1, "rich", dte=_d(k))]


def surface_long_cheap_strangle(k):                       # buy the cheapest call + cheapest put
    return [LegSpec(+1, +1, "cheap", dte=_d(k)), LegSpec(-1, +1, "cheap", dte=_d(k))]


def surface_rv_call(k):                                   # call-side RV: short rich, long cheap
    return [LegSpec(+1, -1, "rich", dte=_d(k)), LegSpec(+1, +1, "cheap", dte=_d(k))]


def surface_rv_put(k):                                    # put-side RV: short rich, long cheap
    return [LegSpec(-1, -1, "rich", dte=_d(k)), LegSpec(-1, +1, "cheap", dte=_d(k))]


def surface_sell_rich_call(k):                            # single-leg: sell the richest call
    return [LegSpec(+1, -1, "rich", dte=_d(k))]


def surface_sell_rich_put(k):
    return [LegSpec(-1, -1, "rich", dte=_d(k))]


RV_ALL = ["surface_short_rich_strangle", "surface_long_cheap_strangle", "surface_rv_call",
          "surface_rv_put", "surface_sell_rich_call", "surface_sell_rich_put"]
_THIS = sys.modules[__name__]


def frames_with_richness(ticker):
    rp = os.path.join(RICH, f"{ticker}.parquet")
    if not os.path.exists(rp):
        return None
    G._PK_DIR = PK
    frames = G.build_array_frames(ticker)
    rdf = pd.read_parquet(rp)
    rmap = {(int(a), int(c), round(float(s), 4), int(e)): float(v) for a, c, s, e, v in
            zip(rdf.asof_i, rdf.cp, rdf.Strike, rdf.expiry_i, rdf.richness)}
    for day, fr in frames.items():
        di = int(day)
        fr["rich"] = np.array([rmap.get((di, int(cp), round(float(K), 4), int(ex)), np.nan)
                               for cp, K, ex in zip(fr["cp"], fr["K"], fr["exp"])])
    return frames


def _task(args):
    ticker, dates = args
    G.clear_cache()
    frames = frames_with_richness(ticker)
    if frames is None:
        return []
    rows, ledgers = [], []
    for arch in RV_ALL:
        try:
            oos, wins, _ = wfo(ticker, dates, getattr(_THIS, arch), GRID, exit_grid=EXITS,
                               obj="rrr", delta_hedge=True, frames=frames)
        except Exception as e:
            rows.append(dict(instrument=ticker, archetype=arch, n=0, error=repr(e)[:80])); continue
        rows.append(_stats(ticker, arch, oos, wins))
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
    os.makedirs(LEDG, exist_ok=True)
    insts = sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(RICH, "*.parquet")))
    dates = sorted(d for y in range(2017, 2027) for d in G.available_dates(y))
    print(f"surface-RV corpus: {len(insts)} instruments x {len(RV_ALL)} archetypes = "
          f"{len(insts)*len(RV_ALL)} cells", flush=True)
    t0 = time.time(); allrows = []
    with mp.Pool(8, maxtasksperchild=1) as pool:
        for k, rows in enumerate(pool.imap_unordered(_task, [(i, dates) for i in insts])):
            allrows.extend(rows)
            if (k + 1) % 10 == 0:
                print(f"  {k+1}/{len(insts)} [{time.time()-t0:.0f}s]", flush=True)
    df = pd.DataFrame(allrows)
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
        dv = [dsr_pvalue(r.sr_trade, r.skew, r.kurt, r.n, trial) for r in s.itertuples()]
        s["dsr"] = [d[0] for d in dv]; s["dsr_pass"] = s["dsr"] > 0.95
        s = s.sort_values("t", ascending=False)
    s.to_csv(os.path.join(OUT, "surface_rv_strategies.csv"), index=False)
    print(f"\nDONE {time.time()-t0:.0f}s. {len(df)} cells, {len(s)} with >={MIN_TRADES} trades.")
    if len(s):
        print(f"  net-positive: {int((s['oos_net']>0).sum())}/{len(s)} | FDR&DSR survivors: "
              f"{int((s['fdr_pass']&s['dsr_pass']).sum())}")
        print(s.head(12)[["instrument", "archetype", "n", "oos_net", "win", "t", "sharpe_ann", "dsr"]].to_string(index=False))


if __name__ == "__main__":
    main()
