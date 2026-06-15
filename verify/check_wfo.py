"""WFO on SPXW short straddle, 2023-2024. Tune {dte-window, exit-dte} on each 6-mo
IS window, evaluate the chosen combo on the next 3-mo OOS. Aggregate OOS = honest.
Demonstrates: OOS (un-tuned) is worse than IS-best (overfit) -> WFO is doing its job."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules

import sys, time
import numpy as np
from optengine.wfo import wfo
from optengine.position import LegSpec
from optengine.data import greeks_store as G


def straddle(k):
    lo, hi = k["dte"]
    return [LegSpec(cp=+1, qty=-1, sel="atm", dte=(lo, hi)),
            LegSpec(cp=-1, qty=-1, sel="atm", dte=(lo, hi))]


dates = G.available_dates(2023) + G.available_dates(2024)
grid = {"dte": [(25, 35), (40, 55)]}
exit_grid = (2, 5, 10)
print(f"SPXW short straddle WFO, {len(dates)} days (2023-24), IS=126/OOS=63, "
      f"delta-hedged, real costs; {len(grid['dte'])*len(exit_grid)} knob combos")

t0 = time.time()
oos, wins, ncombos = wfo("SPXW", dates, straddle, grid, exit_grid=exit_grid,
                         is_len=126, oos_len=63, obj="rrr", delta_hedge=True)
print(f"\nper-window (IS-best combo -> OOS realized):  [{time.time()-t0:.0f}s]")
print(wins[["window", "is_start", "oos_start", "best", "is_net", "oos_net"]].to_string(index=False))

if len(oos):
    n = oos["net"].values
    print(f"\nAGGREGATE OOS (honest, un-tuned): trades={len(oos)}  net=${n.sum():,.0f}  "
          f"win={(n>0).mean():.0%}  med=${np.median(n):,.0f}  worst=${n.min():,.0f}")
    is_tot = wins["is_net"].sum(); oos_tot = wins["oos_net"].sum()
    print(f"IS-best total=${is_tot:,.0f}  vs  OOS total=${oos_tot:,.0f}  "
          f"-> OOS/IS={oos_tot/is_tot if is_tot else float('nan'):.2f} "
          f"(IS optimism shrinks OOS — WFO working). Effective trials for deflation = {ncombos}.")
else:
    print("no OOS trades")
