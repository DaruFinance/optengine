"""Dispersion / correlation strategy test — the canonical structural-vol trade the main study omits.
Short INDEX vol, long a vega-matched basket of CONSTITUENT vols: harvests the implied-correlation
premium (profits when realized correlation < implied). We test whether that premium survives the real
bid/ask, using the SAME engine as the corpus (run_structure: real quoted spread, daily delta-hedge,
real commission) — no reimplemented P&L.

Construction (needs no holdings weights): short one index ATM straddle (vega V), long the constituents
ATM straddles VEGA-WEIGHTED to total V, equal vega per name. For ATM straddles at a common DTE the
Black-Scholes vega ∝ underlying price S, so equal-vega weighting is w_i = (S_index/N)/S_i — exact, and
S (=under_entry) comes straight out of run_structure. Daily delta-hedge neutralises the directional leg
so the spread is a clean long-dispersion / short-implied-correlation bet.

dispersion P&L per cycle = (short index straddle, net) + Σ_i w_i·(LONG constituent straddle_i, net).
Each leg is run in the ACTUAL direction it is traded (index short, constituents long) so run_structure
subtracts the real spread on the correct side — we never negate a short leg to fake a long (that would
flip the cost sign and credit the spread). Each (index, basket) = one strategy.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FINDINGS, PERTKR

import os, sys, time
import numpy as np, pandas as pd
from optengine.position import LegSpec, run_structure
from optengine.data import greeks_store as G
from optengine.costs import OptionCostModel

OUT = FINDINGS
PK = PERTKR
DTE = (25, 35)

# index -> liquid constituents present in pertkr (mega-caps dominate index vol).
BASKETS = {
    "SPY": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "JPM", "V",
            "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "COST", "BAC", "KO"],
    "QQQ": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "TSLA", "AVGO", "COST",
            "PEP", "ADBE", "NFLX", "AMD", "INTC", "CSCO", "QCOM", "TXN", "AMAT", "MU"],
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB"],
    "SMH": ["NVDA", "AVGO", "AMD", "INTC", "QCOM", "TXN", "AMAT", "MU", "LRCX", "KLAC"],
}


def _straddle(side, lo=DTE[0], hi=DTE[1]):
    """side = -1 short straddle, +1 long straddle (both legs same sign)."""
    return [LegSpec(+1, side, "atm", dte=(lo, hi)), LegSpec(-1, side, "atm", dte=(lo, hi))]


def _straddle_cycles(ticker, dates, entry, cost, side):
    """Delta-hedged straddle cycles, real cost, in the ACTUAL traded direction (side=±1).
    -> DataFrame[entry, net, under_entry], or None. net = gross − real spread for THIS side."""
    if not os.path.exists(os.path.join(PK, f"{ticker}.parquet")):
        return None
    G.clear_cache()
    fr = G.build_array_frames(ticker)
    tr = run_structure(ticker, dates, _straddle(side), entry_gap=7, exit_dte=2, cost=cost,
                       delta_hedge=True, frames=fr, entry_days=entry)
    if tr is None or not len(tr):
        return None
    return tr[["entry", "net", "under_entry"]].copy()


def run_basket(index, members, dates, entry, cost):
    idx = _straddle_cycles(index, dates, entry, cost, side=-1)        # SHORT the index
    if idx is None:
        return None
    idx = idx.set_index("entry")
    con = {}
    for mb in members:
        c = _straddle_cycles(mb, dates, entry, cost, side=+1)         # LONG each constituent
        if c is not None:
            con[mb] = c.set_index("entry")
    if len(con) < 4:
        return None
    N = len(con)
    rows = []
    for e, ir in idx.iterrows():
        Sidx = float(ir["under_entry"]); long_pnl = 0.0; nlong = 0
        for cdf in con.values():
            if e in cdf.index:
                cr = cdf.loc[e]
                cr = cr.iloc[0] if isinstance(cr, pd.DataFrame) else cr
                Sc = float(cr["under_entry"])
                if Sc > 0:
                    w = (Sidx / N) / Sc                       # equal-vega split (vega ∝ S for ATM)
                    long_pnl += w * float(cr["net"]); nlong += 1   # net already = long gross − spread
        if nlong >= max(4, N // 2):
            rows.append(dict(entry=e, net=float(ir["net"]) + long_pnl, nlong=nlong))  # short idx + long basket
    if len(rows) < 10:
        return None
    return pd.DataFrame(rows)


def main():
    dates = sorted(d for y in range(2017, 2027) for d in G.available_dates(y))
    entry = set(dates[::21])           # ~monthly entry candidates
    net_cost = OptionCostModel()                       # full taker cross + commission
    gross_cost = OptionCostModel(spread_capture=0.0, commission=0.0)   # mid-only, no fees
    out = []; t0 = time.time()
    for index, members in BASKETS.items():
        rn = run_basket(index, members, dates, entry, net_cost)
        rg = run_basket(index, members, dates, entry, gross_cost)
        if rn is None or not len(rn):
            print(f"{index}: no result", flush=True); continue
        n = len(rn); mu = rn.net.mean(); sd = rn.net.std(ddof=1)
        t = mu / (sd / np.sqrt(n)) if sd > 0 else 0.0
        gmu = rg.net.mean() if rg is not None else float("nan")
        gt = (rg.net.mean() / (rg.net.std(ddof=1) / np.sqrt(len(rg)))) if rg is not None and rg.net.std(ddof=1) > 0 else float("nan")
        out.append(dict(instrument=index, archetype="dispersion_short_idx_long_basket",
                        n=n, gross_net=round(rg.net.sum()) if rg is not None else None,
                        gross_t=round(float(gt), 2), oos_net=round(rn.net.sum()),
                        win=round(float((rn.net > 0).mean()), 3), mean=round(mu), t=round(float(t), 2),
                        sharpe_ann=round(float((mu / sd) * np.sqrt(12)), 2) if sd > 0 else 0,
                        worst=round(float(rn.net.min())), skew=round(float(pd.Series(rn.net).skew()), 2),
                        avg_legs=round(float(rn.nlong.mean()), 1)))
        print(f"{index} dispersion: n={n}  GROSS=${rg.net.sum():,.0f}(t={gt:.2f})  "
              f"NET=${rn.net.sum():,.0f}(t={t:.2f})  win={100*(rn.net>0).mean():.0f}%  "
              f"spread_tax=${rg.net.sum()-rn.net.sum():,.0f}  [{time.time()-t0:.0f}s]", flush=True)
    pd.DataFrame(out).to_csv(os.path.join(OUT, "dispersion_strategies.csv"), index=False)
    surv = sum(1 for o in out if o["t"] >= 3 and o["oos_net"] > 0)
    print(f"\nDONE. {len(out)} dispersion baskets -> dispersion_strategies.csv | net-survivors(t>=3,net>0): {surv}", flush=True)


if __name__ == "__main__":
    main()
