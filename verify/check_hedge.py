"""Delta-hedging isolates the vol premium from direction. Same SPXW short ATM
straddle, full 2024, unhedged vs daily delta-hedged. Expect the unhedged P&L to be
direction-dominated and noisy; the hedged P&L to reflect the realized-vs-implied
vol gap (the actual VRP being harvested), net of real spread costs."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules

import sys
from optengine.position import run_structure, LegSpec
from optengine.data import greeks_store as G

dates = G.available_dates(2024)
struct = [LegSpec(cp=+1, qty=-1, sel="atm", dte=(25, 35)),
          LegSpec(cp=-1, qty=-1, sel="atm", dte=(25, 35))]
print(f"SPXW short ATM straddle, full 2024 ({len(dates)} days), ~30DTE, exit@2DTE, real costs\n")


def summ(tag, tr):
    if not len(tr):
        print(f"  {tag}: no trades"); return
    w = (tr["net"] > 0).mean()
    print(f"  {tag:24s} n={len(tr):2d}  net=${tr['net'].sum():>9,.0f}  "
          f"win={w:>3.0%}  med=${tr['net'].median():>7,.0f}  "
          f"best=${tr['net'].max():>8,.0f}  worst=${tr['net'].min():>9,.0f}")


tr_u = run_structure("SPXW", dates, struct, entry_gap=7, exit_dte=2, capture=1.0, delta_hedge=False)
tr_h = run_structure("SPXW", dates, struct, entry_gap=7, exit_dte=2, capture=1.0, delta_hedge=True)
print("NET P&L per 1-lot, after real bid/ask + commission:")
summ("UNHEDGED (directional)", tr_u)
summ("DELTA-HEDGED (vol)", tr_h)

if len(tr_h):
    og = tr_h["opt_net"].sum(); hp = tr_h["hedge_pnl"].sum()
    print(f"\n  Hedged decomposition: option leg net=${og:,.0f}  +  hedge P&L=${hp:,.0f}  "
          f"=  ${og+hp:,.0f}")
    print("  Interpretation: the hedge absorbs the directional move; what's left is the "
          "short-gamma/short-vega vol P&L — the VRP, net of cost. This is the harvest the\n"
          "  literature says is real-but-thin; the engine now measures it on real SPXW quotes.")
