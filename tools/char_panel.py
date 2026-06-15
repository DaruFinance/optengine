"""Cross-sectional characteristic panel — the causal, per-(name, day) signals the conditioned
long-short test sorts on. Each is known at the 15:59-ET decision time t (contemporaneous chain or a
trailing window), so sorting on it at entry t introduces no lookahead.

Characteristics (the canonical option-return predictors from the literature):
  atm_iv  : ATM ~30d implied vol (median MidImpliedVol, |K/F-1|<0.05, 20-45 DTE)
  rv21    : trailing 21d realized vol of the underlying (annualised)
  ivrv    : atm_iv - rv21   (Goyal-Saretto 2008 vol-mispricing signal)
  skew25  : 25-delta put IV - 25-delta call IV  (risk-reversal / skew)
  volvol  : trailing 21d std of atm_iv  (vol-of-vol; Ruan)
  mom21   : trailing 21d underlying return  (momentum/reversal)
  ivts    : atm_iv(20-45d) - atm_iv(60-120d)  (IV term-structure slope)
All vectorised groupby — no per-day Python loop. Parallel across names on cores 0-15.
Output: findings/char_panel.parquet [instrument, asof_i, <chars>].
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import CHAR_PANEL, PERTKR

import os, sys, glob, time
import multiprocessing as mp
import numpy as np, pandas as pd

PK = PERTKR
OUT = CHAR_PANEL


def one_name(tk):
    fp = os.path.join(PK, f"{tk}.parquet")
    try:
        d = pd.read_parquet(fp, columns=["asof_i", "cp", "Strike", "DaysToMaturity",
                                         "MidImpliedVol", "MidDelta", "UnderLastMidPrice", "converged"])
    except Exception:
        return None
    d = d[d["converged"].to_numpy(bool) & (d["MidImpliedVol"].to_numpy(float) > 0)
          & (d["UnderLastMidPrice"].to_numpy(float) > 0) & (d["Strike"].to_numpy(float) > 0)]
    if len(d) < 5000:
        return None
    d["u"] = d["UnderLastMidPrice"].astype(float)
    d["mny"] = np.abs(d["Strike"].to_numpy(float) / d["u"].to_numpy() - 1.0)
    d["ad"] = np.abs(d["MidDelta"].to_numpy(float))
    dte = d["DaysToMaturity"].to_numpy(float)
    near = (dte >= 20) & (dte <= 45)
    far = (dte >= 60) & (dte <= 120)
    # underlying daily series
    u = d.groupby("asof_i")["u"].median().sort_index()
    lr = np.log(u).diff()
    rv21 = lr.rolling(21).std() * np.sqrt(252)
    mom21 = u.pct_change(21)
    # ATM IV (near & far tenor)
    atm_near = d[near & (d["mny"] < 0.05)].groupby("asof_i")["MidImpliedVol"].median()
    atm_far = d[far & (d["mny"] < 0.05)].groupby("asof_i")["MidImpliedVol"].median()
    # 25-delta put/call IV (delta band 0.18-0.32, near tenor)
    band = near & (d["ad"] >= 0.18) & (d["ad"] <= 0.32)
    pir = d[band & (d["cp"] == -1)].groupby("asof_i")["MidImpliedVol"].median()
    cir = d[band & (d["cp"] == 1)].groupby("asof_i")["MidImpliedVol"].median()
    panel = pd.DataFrame({"atm_iv": atm_near, "atm_far": atm_far,
                          "put25": pir, "call25": cir}).reindex(u.index)
    panel["rv21"] = rv21
    panel["mom21"] = mom21
    panel["ivrv"] = panel["atm_iv"] - panel["rv21"]
    panel["skew25"] = panel["put25"] - panel["call25"]
    panel["ivts"] = panel["atm_iv"] - panel["atm_far"]
    panel["volvol"] = panel["atm_iv"].rolling(21).std()
    panel = panel.reset_index().rename(columns={"index": "asof_i"})
    panel["instrument"] = tk
    return panel[["instrument", "asof_i", "atm_iv", "rv21", "ivrv", "skew25", "volvol", "mom21", "ivts"]]


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    names = sorted(os.path.basename(f)[:-8] for f in glob.glob(os.path.join(PK, "*.parquet")))
    print(f"char_panel: {len(names)} names, {a.workers} workers", flush=True)
    t0 = time.time(); out = []
    with mp.Pool(a.workers, maxtasksperchild=20) as pool:
        for i, p in enumerate(pool.imap_unordered(one_name, names)):
            if p is not None:
                out.append(p)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(names)} [{time.time()-t0:.0f}s]", flush=True)
    panel = pd.concat(out, ignore_index=True)
    panel.to_parquet(OUT, compression="zstd", index=False)
    print(f"DONE {time.time()-t0:.0f}s: {len(panel)} (name,day) rows, {panel.instrument.nunique()} names -> {OUT}", flush=True)
    print(panel[["atm_iv", "rv21", "ivrv", "skew25", "volvol", "mom21", "ivts"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
