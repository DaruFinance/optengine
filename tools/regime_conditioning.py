"""Regime-conditional deflation — does a volatility regime hide a survivor the pooled test misses?

Practitioner folklore: short-vol "works" in calm regimes and you cut size in stress. So we condition every
OOS trade on the entry-date volatility regime (terciles of a VIX proxy = SPY 30-day ATM implied vol) and
re-run the BH-FDR + DSR deflation WITHIN each regime. If short-vol genuinely earns in the low-vol regime net
of costs, a low-VIX-conditional survivor should appear. We also report the short-vol vs long-vol aggregate
net by regime (the carry-vs-crash asymmetry). Pure post-process on the mega-run OOS ledgers.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FINDINGS, LEDGERS, PERTKR

import os, sys, glob, time
import numpy as np, pandas as pd
from scipy import stats as ss
from corpus import dsr_pvalue, MIN_TRADES
from romano_wolf import SHORT_VEGA

PK = PERTKR
LEDG = LEDGERS
OUT = FINDINGS


def vix_proxy():
    """date(YYYYMMDD str) -> SPY ~30-DTE ATM implied vol (a clean, self-constructed VIX proxy)."""
    df = pd.read_parquet(os.path.join(PK, "SPY.parquet"),
                         columns=["asof_i", "Strike", "DaysToMaturity", "MidImpliedVol",
                                  "UnderLastMidPrice", "converged"])
    df = df[df["converged"].to_numpy(bool) & (df["MidImpliedVol"].to_numpy(float) > 0)
            & (df["DaysToMaturity"].to_numpy(float).clip(0) > 0)]
    df["m"] = np.abs(df["Strike"].to_numpy(float) / df["UnderLastMidPrice"].to_numpy(float) - 1.0)
    df["dd"] = np.abs(df["DaysToMaturity"].to_numpy(float) - 30.0)
    df = df[(df["m"] < 0.03) & (df["dd"] < 12)]
    s = df.groupby("asof_i")["MidImpliedVol"].median()
    return {str(int(k)): float(v) for k, v in s.items()}


def _deflate(rows):
    s = pd.DataFrame(rows)
    s = s[(s["n"] >= MIN_TRADES) & (s["sd"] > 0)].copy()
    if not len(s):
        return 0, 0, 0
    s["p"] = 2 * ss.t.sf(s["t"].abs(), s["n"] - 1)
    m = len(s); order = np.argsort(s["p"].to_numpy()); ps = s["p"].to_numpy()[order]
    bh = ps <= (np.arange(1, m + 1) / m) * 0.10
    kmax = (np.where(bh)[0].max() + 1) if bh.any() else 0
    fp = np.zeros(m, bool)
    if kmax:
        fp[order[:kmax]] = True
    trial = s["sr_trade"].to_numpy()
    dsr = np.array([dsr_pvalue(r.sr_trade, r.skew, r.kurt, r.n, trial)[0] for r in s.itertuples()])
    surv = int((fp & (dsr > 0.95)).sum())
    return len(s), int((s["oos_net"] > 0).sum()), surv


def main():
    t0 = time.time()
    vix = vix_proxy()
    lv = np.array(sorted(vix.values()))
    q1, q2 = np.percentile(lv, [33.3, 66.7])
    print(f"VIX proxy (SPY 30d ATM IV): {len(vix)} days, terciles at {q1*100:.1f}% / {q2*100:.1f}%", flush=True)

    def regime(d):
        v = vix.get(d)
        if v is None:
            return None
        return "low" if v <= q1 else ("high" if v > q2 else "mid")

    files = sorted(glob.glob(os.path.join(LEDG, "*.parquet")))
    # per (regime) -> per (inst,arch) -> list of trade nets ; plus short/long aggregate
    buckets = {r: {} for r in ("low", "mid", "high")}
    svlv = {r: {"sv": 0.0, "lv": 0.0, "svn": 0, "lvn": 0} for r in ("low", "mid", "high")}
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["entry", "net", "archetype"])
        except Exception:
            continue
        inst = os.path.basename(f)[:-8]
        df["reg"] = df["entry"].astype(str).map(regime)
        df = df[df["reg"].notna()]
        for (arch, reg), g in df.groupby(["archetype", "reg"], sort=False):
            buckets[reg].setdefault((inst, arch), []).extend(g["net"].tolist())
            tgt = svlv[reg]
            if arch in SHORT_VEGA:
                tgt["sv"] += float(g["net"].sum()); tgt["svn"] += len(g)
            else:
                tgt["lv"] += float(g["net"].sum()); tgt["lvn"] += len(g)
    print(f"\nregime-conditional deflation (FDR+DSR within each regime):", flush=True)
    out = []
    for reg in ("low", "mid", "high"):
        rows = []
        for (inst, arch), nets in buckets[reg].items():
            x = np.array(nets, float)
            if len(x) < 3:
                continue
            mu = x.mean(); sd = x.std(ddof=1) if len(x) > 1 else 0.0
            rows.append(dict(instrument=inst, archetype=arch, n=len(x), oos_net=float(x.sum()),
                             sd=sd, t=(mu / (sd / np.sqrt(len(x)))) if sd > 0 else 0.0,
                             sr_trade=(mu / sd) if sd > 0 else 0.0,
                             skew=float(ss.skew(x)) if len(x) > 2 else 0.0,
                             kurt=float(ss.kurtosis(x)) if len(x) > 3 else 0.0))
        nstrat, npos, surv = _deflate(rows)
        a = svlv[reg]
        print(f"  {reg.upper():4s} vol: {nstrat} strategies eval | net+ {npos}/{nstrat} | "
              f"FDR&DSR survivors={surv} | short-vol net ${a['sv']:,.0f} ({a['svn']} tr) | "
              f"long-vol net ${a['lv']:,.0f} ({a['lvn']} tr)", flush=True)
        out.append(dict(regime=reg, strategies=nstrat, net_positive=npos, survivors=surv,
                        short_vol_net=round(a["sv"]), long_vol_net=round(a["lv"])))
    pd.DataFrame(out).to_csv(os.path.join(OUT, "regime_conditioning.csv"), index=False)
    print(f"\nDONE {time.time()-t0:.0f}s -> regime_conditioning.csv", flush=True)


if __name__ == "__main__":
    main()
