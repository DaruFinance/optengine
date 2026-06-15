"""Survivor-PORTFOLIO test — addresses the reviewers' core gap: the positive thesis was
asserted at the portfolio level but never constructed. Build the equal-weight monthly
portfolio of the survivor cells (delta-hedged, return-on-premium per trade), and report:
annualized Sharpe + t-stat, max drawdown, CVaR(5%), the drawdown through the in-sample
vol spikes (Feb-2018 Volmageddon, Mar-2020 COVID, 2022), a PORTFOLIO-level deflated
significance (Sharpe vs the expected-max under N_eff independent trials), the market beta
of the hedged portfolio (direction-neutrality), and an OI-based capacity proxy for the
survivor instruments. Reports for two survivor sets: t>2.5 and the 1.5x-cost-robust subset.
"""

from optengine.config import FINDINGS, SURFACE_PANEL

import os, glob
import numpy as np, pandas as pd
from scipy.stats import norm, t as tdist

OUT = FINDINGS
PANEL = SURFACE_PANEL
N_EFF = 36.0


def ledger():
    led = pd.concat([pd.read_parquet(p, columns=["instrument", "archetype", "exit", "net", "premium"])
                     for p in glob.glob(os.path.join(OUT, "ledgers", "*.parquet"))], ignore_index=True)
    led["ret"] = led["net"] / led["premium"].abs().replace(0, np.nan)
    led["m"] = pd.to_datetime(led["exit"].astype(str), format="%Y%m%d").dt.to_period("M")
    led["sid"] = list(zip(led["instrument"], led["archetype"]))
    return led


def portfolio_monthly(led, cells):
    sub = led[led["sid"].isin(cells)].dropna(subset=["ret"])
    cell_m = sub.groupby(["sid", "m"])["ret"].mean().reset_index()    # per-cell monthly
    return cell_m.groupby("m")["ret"].mean().sort_index()             # equal-weight across cells


def spy_monthly():
    sp = pd.read_parquet(PANEL, columns=["date", "underlying", "under_px"])
    spy = sp[sp["underlying"] == "SPY"].copy()
    spy["m"] = pd.to_datetime(spy["date"]).dt.to_period("M")
    return spy.groupby("m")["under_px"].last().pct_change()


def report(name, port, mkt):
    r = port.values
    n = len(r)
    mu, sd = r.mean(), r.std(ddof=1)
    ann = np.sqrt(12) * mu / sd
    tstat = mu / (sd / np.sqrt(n))
    eq = (1 + port).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    k = max(1, int(0.05 * n))
    cvar = float(np.sort(r)[:k].mean())
    # deflated: portfolio per-month Sharpe vs expected-max of N_eff independent trials
    sr_m = mu / sd
    emc = 0.5772156649
    sr0 = (1.0 / np.sqrt(n)) * ((1 - emc) * norm.ppf(1 - 1 / N_EFF) + emc * norm.ppf(1 - 1 / (N_EFF * np.e)))
    dsr = float(norm.cdf((sr_m - sr0) * np.sqrt(n - 1)))
    j = pd.DataFrame({"p": port, "mkt": mkt}).dropna()
    beta = float(np.polyfit(j["mkt"].values, j["p"].values, 1)[0]) if len(j) > 5 else np.nan
    def mret(pstr):
        try: return float(port.get(pd.Period(pstr), np.nan))
        except Exception: return np.nan
    print(f"\n[{name}] {n} months")
    print(f"  ann Sharpe={ann:.2f}  t={tstat:.2f}  mean_monthly={mu:.2%}  maxDD={dd:.1%}  CVaR5%={cvar:.2%}")
    print(f"  portfolio Deflated-Sharpe (N_eff={N_EFF:.0f})={dsr:.3f}  market-beta(vs SPY)={beta:+.2f}")
    print(f"  vol-spike months: Feb2018={mret('2018-02'):.2%}  Mar2020={mret('2020-03')*1:.2%}  "
          f"worst-month={r.min():.2%} ({port.idxmin()})")


def capacity(cells):
    insts = sorted({c[0] for c in cells})
    sp = pd.read_parquet(PANEL, columns=["date", "underlying", "total_oi", "atm_spread_bp"])
    g = sp[sp["underlying"].isin(insts)].groupby("underlying").agg(
        med_total_oi=("total_oi", "median"), med_spread_bp=("atm_spread_bp", "median"))
    print("\n[capacity proxy] survivor instruments — median total option OI (contracts) & ATM spread:")
    print(g.round(0).to_string())
    print("  (OI x $100 mult x ~a few % daily turnover bounds deployable size; VIX/HYG/EWZ are")
    print("   liquid but finite — a single desk's clip, not a strategy of unbounded capacity.)")


def main():
    led = ledger()
    s = pd.read_csv(os.path.join(OUT, "corpus_deflated.csv"))
    t25 = set(zip(s[s["t"] > 2.5]["instrument"], s[s["t"] > 2.5]["archetype"]))
    rb = pd.read_csv(os.path.join(OUT, "robustness.csv"))
    robust = set(zip(rb[rb["robust"] == True]["instrument"], rb[rb["robust"] == True]["archetype"]))
    mkt = spy_monthly()
    report("t>2.5 survivors (%d cells)" % len(t25), portfolio_monthly(led, t25), mkt)
    report("1.5x-cost-robust subset (%d cells)" % len(robust), portfolio_monthly(led, robust), mkt)
    capacity(robust)


if __name__ == "__main__":
    main()
