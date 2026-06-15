"""Theory link (v2, holding-window aligned) — decompose the systematic delta-hedged short-vol book into
equity beta + variance risk premium + residual alpha (Israelov-Nielsen FAJ 2015; Carr-Wu 2009; Bollerslev-
Tauchen-Zhou 2009). Trade-level, with each trade's factor exposures measured over its OWN entry→exit window
(not entry-month bucketing), single names only (a comprehensive ETF exclusion), cluster-robust (by entry
month) standard errors, and a short-gamma (MKT^2) term.

Per short-straddle trade i on a single name: y = trade net/premium (and gross/premium);
  MKT_i  = SPY log return over [entry_i, exit_i]
  MKT2_i = MKT_i^2                                  (short-gamma: the convex equity exposure left by hedging)
  VRP_i  = SPY ATM-IV(entry_i)^2 - SPY realized-var(entry_i, exit_i)   (realized variance premium over the hold)
Regress y ~ 1 + MKT + MKT2 + VRP, cluster-robust SE by entry month.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import CHAR_PANEL, FINDINGS, LEDGERS, PERTKR

import os, sys, glob, time
import numpy as np, pandas as pd

LEDG = LEDGERS
PK = PERTKR
PANEL = CHAR_PANEL
OUT = FINDINGS

# comprehensive ETF / non-single-name exclusion (index, sector, leveraged, commodity, bond, country, vol)
ETFS = set("""SPY QQQ IWM DIA VTI VOO IVV MDY RSP SPLG
XLF XLK XLE XLB XLI XLP XLU XLV XLY XLRE XLC SMH XBI XOP KRE KBE IYR ITB XHB XRT OIH GDX GDXJ
TQQQ SQQQ SPXL SPXS SOXL SOXS TNA TZA UPRO SDOW UDOW LABU LABD FAS FAZ YINN
UVXY SVXY VXX VIXY VIX UVIX
GLD SLV USO UNG DBA DBC PALL PPLT CPER WEAT CORN SLX
TLT IEF SHY AGG BND HYG LQD JNK TIP MBB EMB
EEM EFA FXI EWZ AAXJ INDA EWJ EWW EWY EWT EWG EWU EWH EWA VGK VWO IEMG ACWI
ARKK ARKG XME PINK
SPX SPXW NDX RUT OEX XSP XEO DJX NDXP RUTW UUP FXE UUP UST UDN""".split())


def spy_series():
    d = pd.read_parquet(os.path.join(PK, "SPY.parquet"), columns=["asof_i", "UnderLastMidPrice"])
    s = d.groupby("asof_i")["UnderLastMidPrice"].median().sort_index()
    s.index = s.index.astype(int)
    return s


def spy_atmiv():
    p = pd.read_parquet(PANEL)
    p = p[p.instrument == "SPY"][["asof_i", "atm_iv"]].dropna()
    return dict(zip(p.asof_i.astype(int), p.atm_iv.astype(float)))


def cluster_ols(y, X, groups):
    """OLS with cluster-robust (by group) covariance. X includes a constant column."""
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    e = y - X @ b
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        m = groups == g
        Xg = X[m]; eg = e[m]
        s = Xg.T @ eg
        meat += np.outer(s, s)
    G = len(np.unique(groups)); n, k = X.shape
    adj = (G / (G - 1)) * ((n - 1) / (n - k))
    cov = adj * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.diag(cov))
    return b, b / se, 1 - e.var() / y.var()


def main():
    t0 = time.time()
    spy = spy_series(); idx = spy.index.to_numpy()
    lr = np.log(spy).diff()
    iv = spy_atmiv()
    names = set(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(LEDG, "*.parquet"))) - ETFS

    def at(date):                       # nearest SPY trading day <= date
        j = np.searchsorted(idx, date, side="right") - 1
        return idx[j] if j >= 0 else None

    rows = []
    for f in glob.glob(os.path.join(LEDG, "*.parquet")):
        inst = os.path.basename(f)[:-8]
        if inst not in names:
            continue
        try:
            d = pd.read_parquet(f, columns=["entry", "exit", "archetype", "premium", "gross", "net"])
        except Exception:
            continue
        d = d[(d.archetype == "short_straddle") & (d.premium > 0)]
        for r in d.itertuples():
            e0, e1 = at(int(r.entry)), at(int(r.exit))
            if e0 is None or e1 is None or e1 <= e0 or e0 not in iv:
                continue
            mkt = float(np.log(spy[e1] / spy[e0]))
            win = lr[(lr.index > e0) & (lr.index <= e1)]
            if len(win) < 3:
                continue
            rv = float(win.std() * np.sqrt(252))
            vrp = iv[e0] ** 2 - rv ** 2
            rows.append((inst, int(r.entry) // 100, r.gross / r.premium, r.net / r.premium, mkt, mkt * mkt, vrp))
    D = pd.DataFrame(rows, columns=["inst", "mon", "gross", "net", "mkt", "mkt2", "vrp"])
    print(f"trade-level theory decomp: {len(D)} single-name short-straddle trades, "
          f"{D.mon.nunique()} months, {D.inst.nunique()} names (indices+ETFs excluded) [{time.time()-t0:.0f}s]", flush=True)
    res = []
    grp = D.mon.to_numpy()
    for dep in ("gross", "net"):
        y = D[dep].to_numpy()
        for label, cols in [("MKT", ["mkt"]), ("MKT+MKT2", ["mkt", "mkt2"]), ("MKT+MKT2+VRP", ["mkt", "mkt2", "vrp"])]:
            X = np.column_stack([np.ones(len(D))] + [D[c].to_numpy() for c in cols])
            b, t, r2 = cluster_ols(y, X, grp)
            row = dict(dep=dep, model=label, alpha=round(b[0], 4), alpha_t=round(t[0], 2),
                       b_mkt=round(b[1], 3), t_mkt=round(t[1], 2), r2=round(r2, 3))
            if "mkt2" in cols:
                row["b_mkt2"] = round(b[2], 2); row["t_mkt2"] = round(t[2], 2)
            if "vrp" in cols:
                row["b_vrp"] = round(b[3], 3); row["t_vrp"] = round(t[3], 2)
            res.append(row)
    R = pd.DataFrame(res)
    R.to_csv(os.path.join(OUT, "theory_decomp.csv"), index=False)
    print(R.to_string(index=False), flush=True)
    g = R[(R.dep == "gross") & (R.model == "MKT+MKT2+VRP")].iloc[0]
    n = R[(R.dep == "net") & (R.model == "MKT+MKT2+VRP")].iloc[0]
    print(f"\nHEADLINE (trade-level, hold-window-aligned, cluster-robust): short-gamma b_mkt2={g['b_mkt2']} "
          f"(t={g['t_mkt2']}); VRP loading {g['b_vrp']} (t={g['t_vrp']}). GROSS alpha {g['alpha']:+.4f} "
          f"(t={g['alpha_t']}); NET alpha {n['alpha']:+.4f} (t={n['alpha_t']}).", flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
