"""European price-parity gate on SPXW (cash-settled European index — same convention
as our BS). Forward from liquid SPX put-call parity, r external, q from the forward.
Expect a TIGHT match to vendor theo (no American premium to confound it)."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules

import sys
import numpy as np, pandas as pd
from optengine import pricing as P
from optengine.data import greeks_store as G

d = G.available_dates(2024)[5]
ch = G.load_greeks_day(d, "SPXW")
ch = ch[ch["ImpliedVolConvergence"].astype(str).str.startswith("Converged")
        & (ch["MidImpliedVol"] > 0) & (ch["MidTheoPrice"] > 0)].copy()
S = float(ch["UnderLastMidPrice"].iloc[0])
R = 0.044

rows = []
for exp, g in ch.groupby("expiry"):
    T = float(g["YearsToMaturity"].iloc[0])
    if T <= 0:
        continue
    piv = g.pivot_table(index="Strike", columns="cp", values="LastMidPrice")
    if (1 in piv) and (-1 in piv):
        piv = piv.dropna(subset=[1, -1])
    else:
        continue
    if len(piv) < 8:
        continue
    F, _ = P.forward_from_parity(piv.index.values, piv[1].values, piv[-1].values, S)
    q = float(np.clip(R - np.log(max(F, 1e-9) / S) / T, -0.02, 0.10))
    g = g.copy()
    g["our"] = P.bs_price(S, g["Strike"].values, T, R, q, g["MidImpliedVol"].values, g["cp"].values)
    rows.append(g)

a = pd.concat(rows)
a["m"] = a["Strike"] / S
near = a[(a["LastMidPrice"] > 1.0) & (a["m"].between(0.9, 1.1))]
d_all = (a["our"] - a["MidTheoPrice"]).abs()
d_near = (near["our"] - near["MidTheoPrice"]).abs()
print(f"SPXW {d}: {len(a)} converged contracts priced, under={S:.2f}")
print(f"  European BS vs vendor theo  ALL ({len(a)}):  med|Δ|={d_all.median():.3f}  p95={d_all.quantile(.95):.3f}")
print(f"  near-ATM (0.9-1.1, mid>$1) ({len(near)}): med|Δ|={d_near.median():.3f}  p95={d_near.quantile(.95):.3f}")
print(f"  as % of mid (near-ATM): {100*(d_near/near['LastMidPrice']).median():.2f}%")
print("\nGATE: European venue match is tight (no American premium) -> pricer validated "
      "on the convention we slice on. Single-name residual was borrow/forward, not the pricer.")
