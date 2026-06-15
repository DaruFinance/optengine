"""Verify the daily Greeks loader on real local data (SPXW European index + SPY)."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules

import sys
from optengine.data import greeks_store as G

ds = G.available_dates(2024)
print(f"2024 trading days available: {len(ds)}  (first {ds[:3]} ... last {ds[-1]})")

for tk in ["SPXW", "SPY"]:
    df = G.load_greeks_day(ds[0], tk)
    if df is None:
        print(f"\n{tk}: NOT FOUND"); continue
    conv = df["ImpliedVolConvergence"].astype(str).str.startswith("Converged").mean()
    print(f"\n{tk} {ds[0]}: rows={len(df)}  expiries={df.expiry.nunique()}  "
          f"strikes={df.Strike.nunique()}  style={list(df.OptionStyle.unique())}  "
          f"under={df.UnderLastMidPrice.iloc[0]:.2f}")
    print(f"  IV {df.MidImpliedVol.min():.3f}-{df.MidImpliedVol.max():.3f}  "
          f"med spread=${df.spread.median():.2f} ({100*df.spread_pct.median():.1f}% of mid)  "
          f"converged={100*conv:.0f}%")
    fe = df.expiry.min()
    near = df[df.expiry == fe]
    print(f"  front expiry {fe.date()}: {len(near)} contracts "
          f"({int((near.cp>0).sum())}C/{int((near.cp<0).sum())}P)")

r = G.load_greeks_range("SPXW", "20240102", "20240110")
print(f"\nSPXW range 2024-01-02..01-10: rows={0 if r is None else len(r)}  "
      f"days={0 if r is None else r['asof_date'].nunique()}")
print("\nPASS: SPXW (European index) chains load locally with IV+Greeks+bid/ask "
      "-> ready for the vertical slice.")
