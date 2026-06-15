"""Power analysis + DSR benchmark-calibration audit, answering reviewer issues #1 and #3.

#1 Positive control / power: does the DSR gate have power to certify a genuine net edge at our
   per-cell sample sizes? We (a) report the trades-per-cell distribution, (b) compute the minimum
   true annualized Sharpe a clean (Gaussian) strategy needs to clear DSR>0.95 under the AS-IMPLEMENTED
   empirical-dispersion sr0, and (c) inject synthetic strategies of known Sharpe and report pass rates.
#3 Benchmark calibration: the empirical cross-sectional sr0 is inflated by structurally-doomed cells
   (heterogeneous TRUE Sharpes), which raises the bar (biases toward the null). Recompute survivors under
   a noise-only null dispersion and a robust (MAD) dispersion, and run a trial-family granularity panel.

Reads corpus_deflated.csv (equity, 18,766 cells) + fop_corpus_strategies.csv (cross-venue, 550).
Single-threaded, tiny RAM. No new fit; re-deflates stored per-cell stats.
"""

from optengine.config import FINDINGS

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np, pandas as pd
from scipy.stats import norm
from scipy.optimize import brentq

OUT = FINDINGS
GAMMA = 0.5772156649015329


def best_of_n_multiple(N):
    """Bailey-LdP expected-max-of-N-standard-normals multiplier."""
    return (1 - GAMMA) * norm.ppf(1 - 1.0 / N) + GAMMA * norm.ppf(1 - 1.0 / (N * np.e))


def dsr(sr, n, skew, kurt, sr0):
    """Deflated Sharpe (probabilistic Sharpe vs sr0). sr = per-trade Sharpe."""
    denom = np.sqrt(np.clip(1 - skew * sr + (kurt - 1) / 4.0 * sr * sr, 1e-9, None))
    return norm.cdf((sr - sr0) * np.sqrt(np.clip(n - 1, 1, None)) / denom)


def emp_sr0(sr_trades, N=None):
    """As-implemented benchmark: cross-sectional std of trial per-trade Sharpes x best-of-N multiple."""
    sr_trades = np.asarray(sr_trades, float); sr_trades = sr_trades[np.isfinite(sr_trades)]
    N = len(sr_trades) if N is None else N
    return np.std(sr_trades, ddof=1) * best_of_n_multiple(N), np.std(sr_trades, ddof=1)


def load():
    eq = pd.read_csv(os.path.join(OUT, "corpus_deflated.csv"))
    eq = eq[np.isfinite(eq["sr_trade"]) & np.isfinite(eq["n"]) & (eq["n"] >= 12)].copy()
    fo = pd.read_csv(os.path.join(OUT, "fop_corpus_strategies.csv"))
    fo = fo[np.isfinite(fo["sr_trade"]) & np.isfinite(fo["n"]) & (fo["n"] >= 12)].copy()
    return eq, fo


def main():
    eq, fo = load()
    print("=" * 78)
    print("CONFIRM DSR formula reproduces stored column (equity)")
    s = eq.iloc[0]
    rec = dsr(s["sr_trade"], s["n"], s["skew"], s["kurt"], s["sr0"])
    print(f"  {s['instrument']}/{s['archetype']}: stored dsr={s['dsr']:.3e}  recomputed={rec:.3e}")
    chk = dsr(eq["sr_trade"].values, eq["n"].values, eq["skew"].values, eq["kurt"].values, eq["sr0"].values)
    print(f"  max |recomputed - stored| over {len(eq)} cells = {np.nanmax(np.abs(chk - eq['dsr'].values)):.2e}")

    # ---------- #1a  trades-per-cell distribution ----------
    print("=" * 78); print("ISSUE #1a  TRADES-PER-CELL DISTRIBUTION")
    for name, d in [("equity (18,766)", eq), ("cross-venue (550)", fo)]:
        n = d["n"].values
        qs = np.percentile(n, [5, 10, 25, 50, 75, 90, 95])
        print(f"  {name:22s} min={n.min():.0f}  p5={qs[0]:.0f} p10={qs[1]:.0f} p25={qs[2]:.0f} "
              f"p50={qs[3]:.0f} p75={qs[4]:.0f} p90={qs[5]:.0f} p95={qs[6]:.0f} max={n.max():.0f}  mean={n.mean():.0f}")
        for thr in (50, 100, 250):
            print(f"      cells with n>={thr:4d}: {(n>=thr).sum():6d} ({100*(n>=thr).mean():.1f}%)")

    # ---------- #1b  empirical benchmark sr0 ----------
    print("=" * 78); print("ISSUE #3 (setup)  EMPIRICAL BENCHMARK sr0 (as implemented)")
    eq_sr0, eq_std = emp_sr0(eq["sr_trade"].values)
    fo_sr0, fo_std = emp_sr0(fo["sr_trade"].values)
    print(f"  equity:      cross-sectional std(sr_trade)={eq_std:.3f}  N={len(eq)}  multiple={best_of_n_multiple(len(eq)):.2f}  sr0={eq_sr0:.3f}  (stored {eq['sr0'].iloc[0]:.3f})")
    print(f"  cross-venue: cross-sectional std(sr_trade)={fo_std:.3f}  N={len(fo)}  multiple={best_of_n_multiple(len(fo)):.2f}  sr0={fo_sr0:.3f}")
    print(f"  max per-trade Sharpe present: equity {eq['sr_trade'].max():.2f}  cross-venue {fo['sr_trade'].max():.2f}")
    print(f"  best stored DSR: equity {eq['dsr'].max():.3f}  cross-venue {fo['dsr'].max():.3f}")

    # ---------- #1c  POSITIVE CONTROL: min true Sharpe to clear DSR>0.95 under empirical sr0 ----------
    print("=" * 78); print("ISSUE #1  POSITIVE CONTROL: can the gate certify a genuine edge?")
    print("  Clean Gaussian strategy (skew 0, kurt 3). Minimum per-trade & annualized Sharpe to clear DSR>0.95")
    print("  under the AS-IMPLEMENTED empirical sr0, by sample size n (tpy=median trades/yr of the arm):")
    for name, d, sr0v in [("equity", eq, eq_sr0), ("cross-venue", fo, fo_sr0)]:
        tpy = np.median(d["tpy"].values) if "tpy" in d else np.median(d["n"].values) / 4.0
        print(f"  --- {name}: empirical sr0={sr0v:.2f}, median tpy={tpy:.0f} ---")
        for n in [int(np.percentile(d['n'],25)), int(np.percentile(d['n'],50)), int(np.percentile(d['n'],90)), int(d['n'].max())]:
            # solve (sr - sr0)*sqrt(n-1)/sqrt(1+0.5 sr^2) = z_.95  for sr > sr0
            z = norm.ppf(0.95)
            f = lambda sr: (sr - sr0v) * np.sqrt(n - 1) / np.sqrt(1 + 0.5 * sr * sr) - z
            try:
                sr_need = brentq(f, sr0v + 1e-6, sr0v + 50)
                print(f"      n={n:5d}:  per-trade Sharpe >= {sr_need:5.2f}   => annualized Sharpe >= {sr_need*np.sqrt(tpy):6.1f}")
            except Exception as e:
                print(f"      n={n:5d}:  no solution ({e})")

    # inject synthetic strategies of known annualized Sharpe at the actual n-distribution
    print("  Injection test: synthetic Gaussian strategies at the arm's own n-distribution, DSR-pass rate")
    rng = np.random.default_rng(7)
    for name, d, sr0v in [("equity", eq, eq_sr0), ("cross-venue", fo, fo_sr0)]:
        tpy = np.median(d["tpy"].values)
        ns = d["n"].values.astype(int)
        print(f"  --- {name} (sr0={sr0v:.2f}) ---")
        for S_ann in [1, 2, 3, 5, 8, 12]:
            sr = S_ann / np.sqrt(tpy)            # per-trade Sharpe of a true-Sharpe-S_ann strategy
            d_inj = dsr(sr, ns, 0.0, 3.0, sr0v)  # clean Gaussian, evaluated at each cell's n
            print(f"      true annualized Sharpe {S_ann:2d} (per-trade {sr:.2f}):  DSR>0.95 in {100*(d_inj>0.95).mean():5.1f}% of cells")

    # ---------- #3  BENCHMARK ROBUSTNESS: noise-only & robust dispersion, survivor recount ----------
    print("=" * 78); print("ISSUE #3  BENCHMARK CALIBRATION ROBUSTNESS (survivor recount)")
    print("  A survivor = net-positive (oos_net>0) AND fdr_pass AND DSR>0.95. Recompute sr0 three ways:")
    for name, d in [("equity", eq), ("cross-venue", fo)]:
        srt = d["sr_trade"].values; n = d["n"].values
        netpos = d["oos_net"].values > 0
        fdr = d["fdr_pass"].astype(str).str.lower().isin(["true", "1", "1.0"]).values
        mult = best_of_n_multiple(len(d))
        # (a) empirical dispersion (as implemented)
        sr0_emp = np.std(srt, ddof=1) * mult
        # (b) noise-only null dispersion: per-cell null Sharpe SE ~ 1/sqrt(n); use the median as the null spread
        sr0_noise = np.median(1.0 / np.sqrt(n)) * mult
        # (c) robust dispersion: MAD-based std (de-weights the doomed tails) x multiple
        mad = np.median(np.abs(srt - np.median(srt))) * 1.4826
        sr0_mad = mad * mult
        print(f"  --- {name} (N={len(d)}, multiple={mult:.2f}) ---")
        for lbl, s0 in [("(a) empirical  ", sr0_emp), ("(c) robust MAD ", sr0_mad), ("(b) noise-only ", sr0_noise)]:
            dd = dsr(srt, n, d["skew"].values, d["kurt"].values, s0)
            surv = netpos & fdr & (dd > 0.95)
            print(f"      sr0={s0:5.2f}  {lbl}: DSR>0.95 cells={int((dd>0.95).sum()):5d}   net-pos+fdr+DSR survivors={int(surv.sum()):4d}")
            if surv.sum() and surv.sum() <= 25:
                top = d[surv].sort_values("sr_trade", ascending=False)
                for _, r in top.head(15).iterrows():
                    print(f"          {r['instrument']:8s} {r['archetype']:18s} n={int(r['n']):4d} "
                          f"srT={r['sr_trade']:.2f} ann={r['sharpe_ann']:.2f} skew={r['skew']:+.1f} net=${r['oos_net']:,.0f}")

    # ---------- #3  TRIAL-FAMILY GRANULARITY PANEL (equity) ----------
    print("=" * 78); print("ISSUE #3  TRIAL-FAMILY GRANULARITY PANEL (equity): sr0 and survivors vs family definition")
    FAM = {"short_straddle":"shortvol","short_strangle":"shortvol","wide_strangle":"shortvol","short_put_atm":"shortvol",
           "short_call_atm":"shortvol","put_write":"shortvol","call_overwrite":"shortvol","iron_condor":"shortvol",
           "iron_butterfly":"shortvol","put_credit_spread":"shortvol","call_credit_spread":"shortvol",
           "broken_wing_condor":"shortvol","broken_wing_put_fly":"shortvol","jade_lizard":"shortvol","big_lizard":"shortvol",
           "put_ratio":"shortvol","call_ratio":"shortvol"}
    eq["vega"] = eq["archetype"].map(lambda a: FAM.get(a, "other"))
    netpos = eq["oos_net"].values > 0
    fdr = eq["fdr_pass"].astype(str).str.lower().isin(["true","1","1.0"]).values
    def survivors_within(mask_groups):
        tot = 0
        for _, idx in mask_groups:
            sub = eq.iloc[idx]
            if len(sub) < 3: continue
            s0 = np.std(sub["sr_trade"].values, ddof=1) * best_of_n_multiple(len(sub))
            dd = dsr(sub["sr_trade"].values, sub["n"].values, sub["skew"].values, sub["kurt"].values, s0)
            np_ = sub["oos_net"].values > 0
            fd = sub["fdr_pass"].astype(str).str.lower().isin(["true","1","1.0"]).values
            tot += int((np_ & fd & (dd > 0.95)).sum())
        return tot
    idxall = [("all", np.arange(len(eq)))]
    idxvega = [(g, np.where(eq["vega"].values == g)[0]) for g in eq["vega"].unique()]
    idxarch = [(g, np.where(eq["archetype"].values == g)[0]) for g in eq["archetype"].unique()]
    idxinst = [(g, np.where(eq["instrument"].values == g)[0]) for g in eq["instrument"].unique()]
    print(f"  whole corpus as one family (N={len(eq)}):     survivors = {survivors_within(idxall)}")
    print(f"  by vega class ({len(idxvega)} families):                  survivors = {survivors_within(idxvega)}")
    print(f"  by archetype ({len(idxarch)} families):                 survivors = {survivors_within(idxarch)}")
    print(f"  by instrument ({len(idxinst)} families):               survivors = {survivors_within(idxinst)}")
    print("  (finer trial family => smaller N => lower benchmark => more 'survivors'; this is the researcher DoF.)")
    print("=" * 78)


if __name__ == "__main__":
    main()
