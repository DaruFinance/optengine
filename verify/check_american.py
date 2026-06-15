"""Verify optengine.american: (1) reduces to European where no early exercise,
(2) converges in steps, (3) closes the ITM-put American premium vs real Algoseek."""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import ALGOSEEK_SAMPLE

import sys, gzip
import numpy as np, pandas as pd
from optengine import pricing as P, american as A

SAMPLE = ALGOSEEK_SAMPLE


def test_no_early_exercise():
    # American CALL with q=0 must equal European BS (never exercised early).
    rng = np.random.default_rng(1); n = 400
    S = rng.uniform(50, 300, n); K = S * rng.uniform(0.7, 1.4, n)
    T = rng.uniform(0.05, 1.5, n); r = rng.uniform(0, 0.06, n)
    sig = rng.uniform(0.1, 0.8, n); cp = np.ones(n)
    am = A.american_price(S, K, T, r, 0.0, sig, cp, steps=800)
    eu = P.bs_price(S, K, T, r, 0.0, sig, cp)
    print(f"  Am call (q=0) vs Eur BS: max|Δ|={np.max(np.abs(am-eu)):.2e}  "
          f"(expect ~0: short-circuited)")


def test_convergence():
    # ITM American put, r>0: premium over European, converging in steps.
    S, K, T, r, q, sig, cp = 90.0, 100.0, 1.0, 0.05, 0.0, 0.30, -1.0
    eu = float(P.bs_price(S, K, T, r, q, sig, cp))
    print(f"  ITM Am put convergence (Eur BS={eu:.4f}):")
    prev = None
    for steps in (128, 256, 512, 1024, 2048):
        am = float(A.binomial_american(S, K, T, r, q, sig, cp, steps=steps)[0])
        ch = "" if prev is None else f"  Δvs prev={am-prev:+.4f}"
        print(f"    steps={steps:5d}  Am={am:.4f}  premium={am-eu:+.4f}{ch}")
        prev = am


def test_algoseek_itm_puts():
    with gzip.open(SAMPLE, "rt") as fh:
        df = pd.read_csv(fh)
    df = df[df["ImpliedVolConvergence"].astype(str).str.startswith("Converged")]
    df = df[(df.MidImpliedVol > 0) & (df.YearsToMaturity > 0) & (df.MidTheoPrice > 0)].copy()
    df["cp"] = np.where(df.CallPut.str.upper().str[0] == "C", 1, -1)
    S = float(df.UnderLastMidPrice.median())
    R_EXT = 0.044
    rows = []
    for exp, g in df.groupby("Expiration"):
        T = float(g.YearsToMaturity.iloc[0])
        piv = g.pivot_table(index="Strike", columns="cp", values="LastMidPrice")
        if (1 in piv) and (-1 in piv):
            piv = piv.dropna(subset=[1, -1])
        else:
            piv = piv.iloc[0:0]
        if len(piv) >= 5:
            F, _ = P.forward_from_parity(piv.index.values, piv[1].values, piv[-1].values, S)
            q = float(np.clip(R_EXT - np.log(max(F, 1e-9) / S) / T, -0.02, 0.10))
        else:
            q = 0.0
        g = g.copy(); g["T"] = T; g["q"] = q
        rows.append(g)
    a = pd.concat(rows, ignore_index=True)
    a["moneyness"] = a.Strike / S
    itm = a[(a.cp < 0) & (a.moneyness > 1.03)].copy()
    eu = P.bs_price(S, itm.Strike.values, itm["T"].values, R_EXT, itm["q"].values,
                    itm.MidImpliedVol.values, itm.cp.values)
    am = A.american_price(S, itm.Strike.values, itm["T"].values, R_EXT, itm["q"].values,
                          itm.MidImpliedVol.values, itm.cp.values, steps=800)
    de = np.abs(eu - itm.MidTheoPrice.values)
    da = np.abs(am - itm.MidTheoPrice.values)
    print(f"  ITM puts ({len(itm)}) vs vendor American-BSM theo price:")
    print(f"    European med|Δ|={np.median(de):.4f}   American med|Δ|={np.median(da):.4f}"
          f"   -> {'CLOSES the gap' if np.median(da) < np.median(de) else 'no improvement'}")


if __name__ == "__main__":
    print("1) no-early-exercise reduction:"); test_no_early_exercise()
    print("2) step convergence:"); test_convergence()
    print("3) real Algoseek ITM puts:"); test_algoseek_itm_puts()
