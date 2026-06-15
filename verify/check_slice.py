"""First real backtest on the engine core: short ATM straddle on SPXW (European),
weekly entry ~30 DTE, exit at 2 DTE, real bid/ask costs. Shows gross vs net."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules

import sys
import pandas as pd
from optengine.position import run_structure, LegSpec
from optengine.data import greeks_store as G

dates = [d for d in G.available_dates(2024) if d <= "20240401"]
print(f"SPXW short ATM straddle, {len(dates)} trading days (Q1-2024), ~30DTE, exit@2DTE")

struct = [LegSpec(cp=+1, qty=-1, sel="atm", dte=(25, 35)),
          LegSpec(cp=-1, qty=-1, sel="atm", dte=(25, 35))]

for cap, lbl in [(0.0, "GROSS (mid, zero spread)"), (1.0, "NET (full taker spread + comm)")]:
    tr = run_structure("SPXW", dates, struct, entry_gap=7, exit_dte=2, capture=cap)
    if not len(tr):
        print(f"  {lbl}: no trades"); continue
    print(f"\n{lbl}: {len(tr)} trades")
    print(tr[["entry", "exit", "under_entry", "under_exit", "gross", "net"]].to_string(index=False))
    col = "net"
    print(f"  total {col}=${tr[col].sum():,.0f}  win-rate={(tr[col]>0).mean():.0%}  "
          f"med=${tr[col].median():,.0f}  worst=${tr[col].min():,.0f}")

# explicit gross-vs-net drag from one run
tr = run_structure("SPXW", dates, struct, entry_gap=7, exit_dte=2, capture=1.0)
drag = tr["gross"].sum() - tr["net"].sum()
print(f"\nSpread+commission drag (gross-net) over {len(tr)} trades: ${drag:,.0f}  "
      f"(${drag/max(len(tr),1):,.0f}/trade)  -- the 'dies in the spread' tax, measured on real quotes")
