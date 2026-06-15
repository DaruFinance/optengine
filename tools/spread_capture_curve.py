"""Spread-capture sensitivity / breakeven curve — answers the referee question the binary
gross-vs-net split cannot: *at what execution quality would these strategies survive?*

The fill is mid + side·capture·half_spread, so a trade's P&L is LINEAR in `capture` and commission
(0.65·ncon) does not scale with it. From each stored OOS trade's (gross, net, ncon) we reconstruct
net at ANY capture c WITHOUT re-running a single backtest:

    spread_full = (gross − net) − 0.65·ncon          # the full-taker half-spread cost
    net_c       = gross − c·spread_full − 0.65·ncon   # c=1 → net, c=0 → gross − commission

Then we re-run the EXACT deflation (BH-FDR q=0.10 + Bailey-López de Prado DSR) at a grid of capture
levels and count survivors, and report each strategy's breakeven capture c* (net_c = 0). This turns
"0 survivors at full taker" into a curve: how far below the quoted spread you would have to execute
before anything clears — the capacity/edge-vs-microstructure frontier.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FINDINGS, LEDGERS

import os, sys, glob, time
import numpy as np, pandas as pd
from scipy import stats as ss
from corpus import dsr_pvalue, MIN_TRADES

LEDG = LEDGERS
OUT = FINDINGS
CAP_GRID = [0.0, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 1.0]
COMM = 0.65


def _agg_one(g):
    """Per-(instrument,archetype) OOS aggregate, vectorised across the capture grid.
    Returns dict[c] -> stat row, plus breakeven c*."""
    gross = g["gross"].to_numpy(float); net = g["net"].to_numpy(float)
    ncon = g["ncon"].to_numpy(float)
    spread_full = (gross - net) - COMM * ncon                    # full-taker half-spread cost ($)
    n = len(g)
    rows = {}
    for c in CAP_GRID:
        x = gross - c * spread_full - COMM * ncon                # net P&L per trade at capture c
        mu = float(x.mean()); sd = float(x.std(ddof=1)) if n > 1 else 0.0
        rows[c] = dict(n=n, oos_net=float(x.sum()), mean=mu, sd=sd,
                       t=(mu / (sd / np.sqrt(n))) if sd > 0 else 0.0,
                       sr_trade=(mu / sd) if sd > 0 else 0.0,
                       skew=float(ss.skew(x)) if n > 2 else 0.0,
                       kurt=float(ss.kurtosis(x, fisher=True)) if n > 3 else 0.0)
    # breakeven capture: net_c = 0  ->  c* = (gross_sum - comm_sum) / spread_full_sum
    gs = float(gross.sum()); cs = COMM * float(ncon.sum()); sf = float(spread_full.sum())
    cstar = (gs - cs) / sf if sf > 0 else np.nan
    return rows, cstar


def _deflate(rows, key):
    """rows: list of per-strategy stat dicts (already at one capture level). Apply BH-FDR + DSR."""
    s = pd.DataFrame([r for r in rows if r["n"] >= MIN_TRADES and r["sd"] > 0]).copy()
    if not len(s):
        return 0, 0, 0.0
    s["p"] = 2 * ss.t.sf(s["t"].abs(), s["n"] - 1)
    m = len(s); order = np.argsort(s["p"].to_numpy()); ps = s["p"].to_numpy()[order]
    bh = ps <= (np.arange(1, m + 1) / m) * 0.10
    kmax = (np.where(bh)[0].max() + 1) if bh.any() else 0
    fp = np.zeros(m, bool)
    if kmax:
        fp[order[:kmax]] = True
    s["fdr_pass"] = fp
    trial = s["sr_trade"].to_numpy()
    dsr = np.array([dsr_pvalue(r.sr_trade, r.skew, r.kurt, r.n, trial)[0] for r in s.itertuples()])
    s["dsr_pass"] = dsr > 0.95
    surv = int((s["fdr_pass"] & s["dsr_pass"]).sum())
    npos = int((s["oos_net"] > 0).sum())
    return npos, surv, float(s["oos_net"].sum())


def main():
    files = sorted(glob.glob(os.path.join(LEDG, "*.parquet")))
    print(f"spread-capture curve: {len(files)} instrument ledgers, capture grid {CAP_GRID}", flush=True)
    t0 = time.time()
    per_cap = {c: [] for c in CAP_GRID}        # capture -> list of per-strategy stat rows
    cstars = []
    nstrat = 0
    for k, f in enumerate(files):
        try:
            df = pd.read_parquet(f, columns=["gross", "net", "ncon", "archetype"])
        except Exception:
            continue
        inst = os.path.basename(f)[:-8]
        for arch, g in df.groupby("archetype", sort=False):
            if len(g) < 3:
                continue
            rows, cstar = _agg_one(g)
            for c in CAP_GRID:
                r = dict(rows[c]); r["instrument"] = inst; r["archetype"] = arch
                per_cap[c].append(r)
            if np.isfinite(cstar):
                cstars.append(cstar)
            nstrat += 1
        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(files)} files [{time.time()-t0:.0f}s]", flush=True)
    print(f"aggregated {nstrat} strategies [{time.time()-t0:.0f}s]; deflating per capture level...", flush=True)
    out = []
    for c in CAP_GRID:
        npos, surv, pnet = _deflate(per_cap[c], c)
        out.append(dict(capture=c, n_strategies=len(per_cap[c]), net_positive=npos,
                        fdr_dsr_survivors=surv, portfolio_net=round(pnet)))
        print(f"  capture={c:.2f}: net+ {npos}/{len(per_cap[c])}  FDR&DSR survivors={surv}  "
              f"portfolio_net=${pnet:,.0f}", flush=True)
    cdf = pd.DataFrame(out); cdf.to_csv(os.path.join(OUT, "spread_capture_curve.csv"), index=False)
    cs = np.array(cstars); cs = cs[np.isfinite(cs)]
    # only strategies with a gross edge (c* in (0,1]) have a meaningful breakeven
    valid = cs[(cs > 0) & (cs <= 2)]
    print(f"\nbreakeven capture c* (net_c=0), strategies with gross edge: n={len(valid)}", flush=True)
    if len(valid):
        for q in (10, 25, 50, 75, 90):
            print(f"  p{q}: {np.percentile(valid, q):.3f}", flush=True)
        print(f"  frac needing better-than-half-spread (c*<0.5): {100*(valid<0.5).mean():.0f}%", flush=True)
        print(f"  frac that survive even at full taker (c*>=1): {100*(cs>=1).mean():.0f}%", flush=True)
    pd.DataFrame(dict(cstar=cs)).to_csv(os.path.join(OUT, "breakeven_capture.csv"), index=False)
    print(f"\nDONE {time.time()-t0:.0f}s -> spread_capture_curve.csv + breakeven_capture.csv", flush=True)


if __name__ == "__main__":
    main()
