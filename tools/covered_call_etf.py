"""Covered-call / income-ETF arm (JEPI/QYLD/XYLD) — Israelov-Nze-Ndong "Devil's Bargain" test.

The ~$80B systematic call-overwrite ETF complex (QYLD=ATM on NDX, XYLD=ATM on SPX, JEPI=S&P overwrite) is the
biggest real-money short-vol vehicle of 2023-26 and was untested. Model the covered call directly: LONG the
index + SHORT a monthly call (ATM and 2%-OTM), un-hedged (the long index carries the delta), real bid/ask on
the call, multiplier 100. The buy-write per-cycle return (per $1 of index notional) is
    bw = index_return + short_call_pnl / (index_level * 100).
The decisive question (Israelov-Nze-Ndong FAJ 2023): does the overwrite add ALPHA over a beta-matched
equity/cash benchmark, or is the option "income" just disguised equity-beta reduction plus an uncompensated
upside give-up? We regress the buy-write on the index (beta), report Jensen's alpha vs the beta-matched mix,
and decompose income (call premium) vs upside give-up (call payoff), GROSS (structural) and NET (real spread).
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FINDINGS, PERTKR

import os, sys, glob, time
import numpy as np, pandas as pd
from optengine.position import LegSpec, run_structure
from optengine.data import greeks_store as G
from optengine.costs import OptionCostModel, MULT

PK = PERTKR
OUT = FINDINGS
INDEXES = ["SPX", "QQQ", "SPY", "IWM", "DIA"]
RF_M = 0.0                              # risk-free per cycle for the cash leg (0 = conservative; see ODTE caveat)


def short_call(kind):
    if kind == "ATM":
        return [LegSpec(+1, -1, "atm", dte=(25, 35))]
    return [LegSpec(+1, -1, "moneyness", 1.02, dte=(25, 35))]      # 2% OTM


def alpha_beta(bw, idx):
    """Jensen regression bw ~ a + b*idx (excess of RF). Returns a (per-cycle), b, t(a), and the
    beta-matched benchmark mean = b*mean(idx) + (1-b)*RF; alpha = mean(bw) - that."""
    x = idx - RF_M; y = bw - RF_M
    b = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    a = y.mean() - b * x.mean()
    resid = y - (a + b * x)
    se_a = np.sqrt(np.var(resid, ddof=2) / len(x) * (1 + x.mean() ** 2 / np.var(x, ddof=1)))
    return a, b, (a / se_a if se_a > 0 else 0.0)


def main():
    G._PK_DIR = PK
    cost = OptionCostModel()
    dates = sorted(d for y in range(2017, 2027) for d in G.available_dates(y))
    rows = []
    for idx in INDEXES:
        if not os.path.exists(os.path.join(PK, f"{idx}.parquet")):
            continue
        G.clear_cache(); fr = G.build_array_frames(idx)
        for kind in ("ATM", "2%OTM"):
            tr = run_structure(idx, dates, short_call(kind), entry_gap=20, exit_dte=2, cost=cost,
                               delta_hedge=False, frames=fr)
            if tr is None or len(tr) < 12:
                continue
            tr = tr[(tr["under_entry"] > 0) & (tr["premium"] > 0)].copy()
            notional = tr["under_entry"].to_numpy() * MULT
            idx_ret = (tr["under_exit"].to_numpy() / tr["under_entry"].to_numpy()) - 1.0
            cc_net = tr["net"].to_numpy() / notional               # short-call pnl per $ index (real spread)
            cc_gross = tr["gross"].to_numpy() / notional           # per $ index (mid)
            bw_net = idx_ret + cc_net; bw_gross = idx_ret + cc_gross
            premium = tr["premium"].to_numpy() / notional          # income collected per $ index
            # decomposition
            a_n, b_n, ta_n = alpha_beta(bw_net, idx_ret)
            a_g, b_g, ta_g = alpha_beta(bw_gross, idx_ret)
            n = len(tr)
            rows.append(dict(index=idx, overwrite=kind, n=n,
                             idx_sharpe=round(idx_ret.mean() / idx_ret.std(ddof=1) * np.sqrt(12), 2),
                             bw_net_sharpe=round(bw_net.mean() / bw_net.std(ddof=1) * np.sqrt(12), 2),
                             beta=round(b_n, 2), income_ann=round(premium.mean() * 12, 4),
                             alpha_gross_ann=round(a_g * 12, 4), alpha_gross_t=round(ta_g, 2),
                             alpha_net_ann=round(a_n * 12, 4), alpha_net_t=round(ta_n, 2),
                             call_leg_net_gross=round((cc_gross.mean() - premium.mean()) * 12, 4)))   # payoff net of premium
            print(f"{idx:4s} {kind:5s}: n={n} idxSh={rows[-1]['idx_sharpe']} bwSh={rows[-1]['bw_net_sharpe']} "
                  f"beta={b_n:.2f} income={premium.mean()*12*100:.1f}%/yr "
                  f"alpha_gross={a_g*12*100:+.1f}%(t={ta_g:.1f}) alpha_net={a_n*12*100:+.1f}%(t={ta_n:.1f})", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(OUT, "covered_call_etf.csv"), index=False)
    if len(R):
        print(f"\n--- Devil's Bargain summary ({len(R)} index x overwrite books) ---", flush=True)
        print(f"  GROSS alpha vs beta-matched: median {R.alpha_gross_ann.median()*100:+.1f}%/yr, "
              f"# with t>2: {int((R.alpha_gross_t>2).sum())}/{len(R)}", flush=True)
        print(f"  NET alpha vs beta-matched: median {R.alpha_net_ann.median()*100:+.1f}%/yr, "
              f"# net-positive: {int((R.alpha_net_ann>0).sum())}/{len(R)} | # with t>2: {int((R.alpha_net_t>2).sum())}/{len(R)}", flush=True)
        print(f"  median buy-write beta to index: {R.beta.median():.2f} (covered call = beta-reduced equity)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
