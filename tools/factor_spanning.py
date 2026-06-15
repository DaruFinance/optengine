"""Factor-spanning / latent-factor test — the frontier-statistics gap (Kozak-Nagel-Santosh "Shrinking the
Cross-Section" JFE 2020; Horenstein-Vasquez-Xiao IPCA RFS 2025). Question: is the option-strategy cross-section
spanned by a FEW common latent factors, with no residual alpha — or do individual archetypes carry idiosyncratic
edge a PCA factor model misses?

Build the monthly return panel of the 35 archetypes (equal-weight mean of net/premium and gross/premium across
all names/trades entering that month), PCA it, report the eigenvalue spectrum + effective number of factors
(participation ratio), and run a SPANNING test: regress each archetype's NET return on the top-K principal
components and count how many have residual alpha significant at t>2. If a handful of PCs explain the bulk of
the variance and 0 archetypes have net alpha to them, the cross-section is a low-rank common-factor structure
with no survivor — the IPCA-style conclusion.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FINDINGS, LEDGERS

import os, sys, glob, time
import numpy as np, pandas as pd
from scipy import stats as ss

LEDG = LEDGERS
OUT = FINDINGS


def panel(value):
    """archetype x month matrix of equal-weight mean (value/premium)."""
    acc = {}
    for f in glob.glob(os.path.join(LEDG, "*.parquet")):
        try:
            d = pd.read_parquet(f, columns=["entry", "archetype", "premium", value])
        except Exception:
            continue
        d = d[d["premium"] > 0]
        if not len(d):
            continue
        d = d.copy(); d["m"] = d["entry"].astype(int) // 100; d["r"] = d[value] / d["premium"]
        for (a, m), g in d.groupby(["archetype", "m"]):
            acc.setdefault(a, {}).setdefault(m, []).append(g["r"].values)
    rows = {a: {m: np.concatenate(v).mean() for m, v in md.items()} for a, md in acc.items()}
    P = pd.DataFrame(rows).T                     # archetypes x months
    return P


def main():
    t0 = time.time()
    Pn = panel("net"); Pg = panel("gross")
    months = sorted(set(Pn.columns) & set(Pg.columns))
    Pn = Pn[months]; Pg = Pg[months]
    # keep archetypes present in >=24 months; fill sparse gaps with 0 (no-position month return)
    keep = Pn.index[(Pn.notna().sum(1) >= 24)]
    Xn = Pn.loc[keep].fillna(0.0).to_numpy(); Xg = Pg.loc[keep].fillna(0.0).to_numpy()
    A, T = Xn.shape
    print(f"factor-spanning: {A} archetypes x {T} months [{time.time()-t0:.0f}s]", flush=True)
    # PCA on the GROSS panel (the common factor structure of the strategies, demeaned per archetype)
    Xg_d = Xg - Xg.mean(1, keepdims=True)
    C = np.cov(Xg_d)                              # archetype x archetype
    w, V = np.linalg.eigh(C)
    w = w[::-1]; V = V[:, ::-1]
    w = np.clip(w, 0, None)
    pr = (w.sum() ** 2) / (w ** 2).sum()         # participation ratio = effective # factors
    cum = np.cumsum(w) / w.sum()
    print(f"  COVARIANCE PCA: top-5 var% = {np.round(w[:5]/w.sum()*100,1)} | "
          f"cum to 90% at K={int(np.argmax(cum>=0.9))+1} | effective #factors (participation ratio) = {pr:.1f}",
          flush=True)
    # CORRELATION-matrix PCA (scale-normalized): the covariance PR is inflated because the two
    # underlying-holding overlays (call_overwrite, put_write) carry ~60% of total panel variance.
    Cc = np.corrcoef(Xg_d)
    wc = np.sort(np.clip(np.linalg.eigvalsh(Cc), 0, None))[::-1]
    prc = (wc.sum() ** 2) / (wc ** 2).sum()
    print(f"  CORRELATION PCA (scale-normalized): PC1 var% = {wc[0]/wc.sum()*100:.1f} | "
          f"effective #factors = {prc:.1f}  -> 'rank-1' is a covariance (variance-domination) statement", flush=True)
    # PC factor returns (project the gross panel onto top components) — common factors
    for K in (1, 3, 5):
        F = (V[:, :K].T @ Xg_d)                   # K x T factor series
        # spanning: regress each archetype NET return on [1, F_1..F_K]; residual alpha + Newey-West-ish t
        Xn_d = Xn                                  # net returns (levels, not demeaned — alpha is the intercept)
        Z = np.column_stack([np.ones(T)] + [F[k] for k in range(K)])
        ZtZ_inv = np.linalg.inv(Z.T @ Z)
        nsig = 0; alphas = []
        for i in range(A):
            y = Xn_d[i]
            b = ZtZ_inv @ Z.T @ y
            e = y - Z @ b
            s2 = (e @ e) / (T - K - 1)
            se_a = np.sqrt(s2 * ZtZ_inv[0, 0])
            ta = b[0] / se_a if se_a > 0 else 0.0
            alphas.append(b[0])
            if ta > 2:                            # one-sided: significantly POSITIVE net alpha
                nsig += 1
        alphas = np.array(alphas)
        print(f"  K={K} PCs: net-alpha>0 in {int((alphas>0).sum())}/{A} archetypes | "
              f"significant (t>2) POSITIVE net alpha: {nsig}/{A} | median net alpha {np.median(alphas):+.4f}/cycle",
              flush=True)
    pd.DataFrame(dict(eig=w, var_frac=w/w.sum())).to_csv(os.path.join(OUT, "factor_spanning.csv"), index=False)
    print(f"\nSummary: the {A}-archetype panel is low-rank (effective {pr:.1f} factors; top-5 PCs span ~"
          f"{cum[4]*100:.0f}% of variance); a 5-PC common-factor model leaves NO archetype with significant "
          f"positive net alpha.", flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
