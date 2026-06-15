"""Correctness gate: optengine.pricing vs REAL Algoseek daily Greeks.

Loads the vendor sample (American-BSM IV + Greeks), derives r,q from the chain via
put-call parity (reusing the same forward_from_parity the surface layer uses), then
prices every contract with our European BS at the vendor IV and compares.

Expectation (this validates BOTH the pricer AND the American-pricer requirement):
  - CALLs on a (near-)zero-dividend name: American == European -> tight match.
  - ITM PUTs: vendor American >= our European by the early-exercise premium -> a
    visible, ONE-SIDED gap that grows with moneyness. That gap is physics, not a bug.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import ALGOSEEK_SAMPLE

import sys, gzip, io
import numpy as np, pandas as pd

from optengine import pricing as P

SAMPLE = ALGOSEEK_SAMPLE


def load():
    with gzip.open(SAMPLE, "rt") as fh:
        df = pd.read_csv(fh)
    df["cp"] = np.where(df["CallPut"].str.upper().str[0] == "C", 1, -1)
    df = df[df["ImpliedVolConvergence"].astype(str).str.startswith("Converged")].copy()
    df = df[(df["MidImpliedVol"] > 0) & (df["YearsToMaturity"] > 0)
            & (df["MidTheoPrice"] > 0)].copy()
    return df


def main():
    df = load()
    S = float(df["UnderLastMidPrice"].median())
    print(f"loaded {len(df)} converged rows  ticker={df['Ticker'].iloc[0]}  "
          f"under={S:.2f}  expiries={df['Expiration'].nunique()}")

    # Lesson from v1: single-name put-call parity recovers the FORWARD (r-q) well but
    # NOT the absolute discount rate (intercept is noise-dominated at one name, few
    # strikes). So take r from an external curve and derive q from the parity forward.
    R_EXT = 0.044  # Jan-2023 ~3M T-bill / EFFR; ZBRA pays no dividend (expect q~0)
    rows = []
    for exp, g in df.groupby("Expiration"):
        T = float(g["YearsToMaturity"].iloc[0])
        piv = g.pivot_table(index="Strike", columns="cp", values="LastMidPrice")
        piv = piv.dropna(subset=[1, -1]) if (1 in piv and -1 in piv) else piv.iloc[0:0]
        r = R_EXT
        if len(piv) >= 5:
            F, _D = P.forward_from_parity(piv.index.values, piv[1].values, piv[-1].values, S)
            q = r - np.log(max(F, 1e-9) / S) / T
            q = float(np.clip(q, -0.02, 0.10))
        else:
            q = 0.0  # too few parity strikes (front weeklies) -> assume no carry
        F = S * np.exp((r - q) * T)
        gg = g.copy()
        gg["our_px"] = P.bs_price(S, gg["Strike"].values, T, r, q,
                                  gg["MidImpliedVol"].values, gg["cp"].values)
        grk = P.bs_greeks(S, gg["Strike"].values, T, r, q,
                          gg["MidImpliedVol"].values, gg["cp"].values)
        gg["our_delta"] = grk["delta"]
        gg["our_gamma"] = grk["gamma"]
        gg["our_vega_1pct"] = grk["vega"] / 100.0      # vendor convention: per 1% vol
        gg["our_theta_day"] = grk["theta"] / 365.0     # vendor convention: per day
        gg["our_rho_1pct"] = grk["rho"] / 100.0
        gg["r"], gg["q"], gg["F"], gg["moneyness"] = r, q, F, gg["Strike"] / S
        rows.append(gg)
        print(f"  exp {exp}: T={T:.4f} F={F:.2f} r={r*100:.2f}% q={q*100:.2f}% "
              f"n={len(gg)} (parity strikes={len(piv)})")

    a = pd.concat(rows, ignore_index=True)

    def report(name, vend, ours, scale=1.0):
        d = (a[ours] - a[vend]).abs()
        print(f"  {name:14s}  med|Δ|={d.median():.5f}  p95|Δ|={d.quantile(.95):.5f}  "
              f"max={d.max():.5f}")

    print("\n== European-BS vs vendor American-BSM (vendor IV as input) ==")
    print("ALL contracts:")
    for nm, v, o in [("price", "MidTheoPrice", "our_px"),
                     ("delta", "MidDelta", "our_delta"),
                     ("gamma", "MidGamma", "our_gamma"),
                     ("vega/1%", "MidVega", "our_vega_1pct"),
                     ("theta/day", "MidTheta", "our_theta_day"),
                     ("rho/1%", "MidRho", "our_rho_1pct")]:
        report(nm, v, o)

    calls = a[a.cp > 0]
    print(f"\nCALLS only ({len(calls)})  -- expect TIGHT (Am call == Eur for q~0):")
    for nm, v, o in [("price", "MidTheoPrice", "our_px"), ("delta", "MidDelta", "our_delta"),
                     ("vega/1%", "MidVega", "our_vega_1pct")]:
        d = (calls[o] - calls[v]).abs()
        print(f"  {nm:10s} med|Δ|={d.median():.5f}  p95={d.quantile(.95):.5f}")

    puts = a[a.cp < 0]
    itm_puts = puts[puts.moneyness > 1.03]
    print(f"\nITM PUTS ({len(itm_puts)})  -- expect ONE-SIDED Am>Eur early-exercise gap:")
    gap = (itm_puts["MidTheoPrice"] - itm_puts["our_px"])
    print(f"  signed (vendor-ours): med={gap.median():.4f}  min={gap.min():.4f}  "
          f"max={gap.max():.4f}   (positive => American premium, as expected)")
    print("\nPASS criterion: calls price med|Δ| < 0.05 and delta med|Δ| < 0.01;")
    print("ITM-put gap one-sided positive => motivates American pricer (next module).")


if __name__ == "__main__":
    main()
