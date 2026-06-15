"""Proper deflation with EFFECTIVE independent trials.

The 2790 (archetype x instrument) strategies are heavily correlated — equity volatility
is ~one common factor, so short-vol on VIX, HYG, EWZ, single-names etc. co-move. Treating
all 2790 as INDEPENDENT trials in the Deflated Sharpe over-deflates (expected-max-Sharpe
under the null is far too high -> ~0 survivors, an artifact). We instead estimate the
effective number of independent trials from the eigenvalue spectrum of the strategy
monthly-return correlation matrix:  N_eff = (Σλ)² / Σλ²  (participation ratio / "effective
number of bets"), then recompute the Deflated Sharpe with N_eff. We report BOTH the
ultra-conservative raw-N DSR and the N_eff DSR for transparency.
"""

from optengine.config import FINDINGS

import os, glob
import numpy as np, pandas as pd
from scipy.stats import norm

OUT = FINDINGS


def monthly_panel():
    files = glob.glob(os.path.join(OUT, "ledgers", "*.parquet"))
    led = pd.concat([pd.read_parquet(p, columns=["instrument", "archetype", "exit", "net", "premium"])
                     for p in files], ignore_index=True)
    led["sid"] = led["instrument"] + "|" + led["archetype"]
    led["ret"] = led["net"] / led["premium"].abs().replace(0, np.nan)
    led["m"] = pd.to_datetime(led["exit"].astype(str), format="%Y%m%d").dt.to_period("M").astype(str)
    return led.pivot_table(index="m", columns="sid", values="ret", aggfunc="mean")


def n_eff(panel, min_obs=24):
    """Effective independent trials = participation ratio (Σλ)²/Σλ² of the strategy
    return-correlation matrix. Computed via the m×m Gram matrix of column-standardized
    returns — N_eff = n²/‖C‖_F², with ‖C‖_F² = ‖ZZ^T/m‖_F² — so we NEVER materialize the
    n×n correlation matrix (RAM-safe at 20k+ strategies; a 15k×15k C would be ~2GB + a
    minutes-long eigendecomposition). Verified bit-identical (36.07) to the prior eigvalsh
    path on the 2,220-cell corpus."""
    p = panel.loc[:, panel.notna().sum() >= min_obs].fillna(0.0).values
    m, n = p.shape
    if n < 2 or m < 2:
        return float(n), n
    Z = (p - p.mean(0)) / (p.std(0, ddof=0) + 1e-12)
    S = Z @ Z.T / m                                  # m×m (months×months) Gram — tiny
    fro2 = float(np.sum(S * S))                       # = ‖C‖_F²
    return float(n * n / fro2), n


def dsr(sr, sk, ku, n, M):
    M = max(M, 2.0)
    v = np.nanvar(sr, ddof=1)
    emc = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - emc) * norm.ppf(1 - 1.0 / M) + emc * norm.ppf(1 - 1.0 / (M * np.e)))
    den = np.sqrt(np.maximum(1 - sk * sr + (ku - 1) / 4.0 * sr * sr, 1e-9))
    return norm.cdf((sr - sr0) * np.sqrt(np.maximum(n - 1, 1)) / den), sr0


def main():
    s = pd.read_csv(os.path.join(OUT, "corpus_strategies.csv"))
    panel = monthly_panel()
    neff, ncols = n_eff(panel)
    raw_M = len(s)
    print(f"panel strategies={ncols}  raw trials={raw_M}  N_eff (participation ratio)={neff:.1f}", flush=True)
    sr = s["sr_trade"].to_numpy(); sk = s["skew"].to_numpy(); ku = s["kurt"].to_numpy(); n = s["n"].to_numpy()
    d_raw, sr0_raw = dsr(sr, sk, ku, n, raw_M)
    d_eff, sr0_eff = dsr(sr, sk, ku, n, neff)
    s["dsr_raw"], s["dsr_eff"] = d_raw, d_eff
    s["dsr_eff_pass"] = s["dsr_eff"] > 0.95
    s["robust_sig"] = s["fdr_pass"] & s["dsr_eff_pass"]
    s.sort_values("t", ascending=False).to_csv(os.path.join(OUT, "corpus_deflated.csv"), index=False)
    print(f"sr0_raw(N={raw_M})={sr0_raw:.3f} -> DSR>.95: {int((d_raw>0.95).sum())}")
    print(f"sr0_eff(N_eff={neff:.0f})={sr0_eff:.3f} -> DSR_eff>.95: {int(s['dsr_eff_pass'].sum())}   FDR & DSR_eff: {int(s['robust_sig'].sum())}")
    print("\nTOP 25 by DSR_eff:")
    cols = ["instrument", "archetype", "n", "oos_net", "win", "t", "sharpe_ann", "dsr_eff", "fdr_pass"]
    print(s.sort_values("dsr_eff", ascending=False).head(25)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
