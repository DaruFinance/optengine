"""P2 fan-out demo: 4 archetypes x 3 instruments = 12 WFOs in parallel on cores 0-15,
RAM-monitored. Proves the engine scales structure x instrument, the corpus axis."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules

import sys, time
import pandas as pd
from optengine.fan import fan, ram_available_gb

pd.set_option("display.width", 170)
print(f"RAM available before: {ram_available_gb():.1f} GB")

insts = ["SPXW", "SPY", "QQQ"]
archs = ["short_straddle", "short_strangle", "iron_condor", "put_write"]
grid = {"dte": [(25, 35), (40, 55)]}
exit_grid = (2, 5, 10)
years = [2023, 2024]
tasks = [(i, a, grid, exit_grid, years) for i in insts for a in archs]

print(f"fanning {len(tasks)} (instrument x archetype) WFOs across <=12 cores (0-15), "
      f"2023-24, delta-hedged, real costs...")
t0 = time.time()
df = fan(tasks, workers=12, ram_floor_gb=10.0)
dt = time.time() - t0

cols = [c for c in ["instrument", "archetype", "n", "oos_net", "win", "med", "worst", "ncombos", "error"] if c in df.columns]
print(f"\n{df.sort_values(['archetype','instrument'])[cols].to_string(index=False)}")
print(f"\nRAM available after: {ram_available_gb():.1f} GB   |   wall {dt:.0f}s for {len(tasks)} WFOs "
      f"({len(tasks)*df.get('ncombos', pd.Series([6])).max()*5:.0f}+ backtests)")
print("Honest OOS net-of-cost per structure x instrument — the survival map, in miniature.")
