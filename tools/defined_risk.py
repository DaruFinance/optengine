"""Defined- vs undefined-risk short-vol: does capping the tail rescue significance?

Motivated by the crypto natural experiment (BTC iron_butterfly, long wings, survives FDR at t=7.2 with the
2021/FTX tail in-sample, while the naked BTC short_straddle collapses to t=0.41 / skew −10.5). Generalize it
to the whole 16-root corpus: split the short-vol archetypes by OPTION STRUCTURE into

  DEFINED-risk (long wings cap the short legs): iron_condor, iron_butterfly, put_credit_spread,
      call_credit_spread, broken_wing_condor, broken_wing_put_fly
  UNDEFINED-risk (naked short legs): short_straddle, short_strangle, wide_strangle, short_put_atm,
      short_call_atm, put_write, call_overwrite
  SEMI (one side capped): jade_lizard, big_lizard, put_ratio, call_ratio

and compare per-trade skew/kurtosis, t, max DSR, FDR/DSR survivors — overall and per asset class. The retail
'income' complex (iron condors, credit spreads) IS the defined-risk bucket, so this directly answers "do the
safe, capped, most-popular income structures beat the deflation bar where naked selling does not?"
Reads findings/fop_corpus_strategies.csv (+ dsr_pvalue for a defined-risk-only trial family). No new fit.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FINDINGS

import os, sys
import numpy as np, pandas as pd
from corpus import dsr_pvalue

OUT = FINDINGS
ACLASS = {"ES": "equity", "RTM": "equity", "OG": "metals", "PAO": "metals", "LO": "energy", "ON": "energy",
          "OZN": "rates", "OZB": "rates", "OZF": "rates", "OZT": "rates", "UB1": "rates",
          "BTC": "crypto", "ETH": "crypto", "OZC": "ags", "OZS": "ags", "OZW": "ags"}
DEFINED = {"iron_condor", "iron_butterfly", "put_credit_spread", "call_credit_spread",
           "broken_wing_condor", "broken_wing_put_fly"}
UNDEFINED = {"short_straddle", "short_strangle", "wide_strangle", "short_put_atm", "short_call_atm",
             "put_write", "call_overwrite"}
SEMI = {"jade_lizard", "big_lizard", "put_ratio", "call_ratio"}


def grp(a):
    return "defined" if a in DEFINED else ("undefined" if a in UNDEFINED else ("semi" if a in SEMI else "other"))


def main():
    s = pd.read_csv(os.path.join(OUT, "fop_corpus_strategies.csv"))
    s["risk"] = s["archetype"].map(grp)
    s["ac"] = s["instrument"].map(ACLASS)
    sv = s[s["risk"].isin(["defined", "undefined", "semi"])].copy()
    print("=== short-vol cells by risk structure (16-root corpus) ===")
    print(f"{'risk':10s} {'cells':>5s} {'med_t':>6s} {'max_t':>6s} {'med_skew':>9s} {'med_kurt':>9s} "
          f"{'maxDSR':>7s} {'FDR&net+':>9s} {'net+':>5s} {'surv':>5s}")
    for r in ("undefined", "semi", "defined"):
        g = sv[sv["risk"] == r]
        surv = int((g["fdr_pass"] & g["dsr_pass"]).sum())
        print(f"{r:10s} {len(g):5d} {g['t'].median():6.2f} {g['t'].max():6.2f} {g['skew'].median():9.2f} "
              f"{g['kurt'].median():9.1f} {g['dsr'].max():7.3f} {int((g['fdr_pass']&(g['oos_net']>0)).sum()):9d} "
              f"{int((g['oos_net']>0).sum()):5d} {surv:5d}")
    # tail magnitude: fraction with skew < -1 and kurt > 10
    print("\n=== tail severity (the capped-vs-naked contrast) ===")
    for r in ("undefined", "semi", "defined"):
        g = sv[sv["risk"] == r]
        print(f"  {r:10s} skew<-1: {int((g['skew']<-1).sum())}/{len(g)} ({(g['skew']<-1).mean()*100:.0f}%) | "
              f"kurt>10: {int((g['kurt']>10).sum())}/{len(g)} | mean|worst-trade|/premium proxy via kurt mean={g['kurt'].mean():.1f}")
    # per asset class: defined vs undefined max t and survivors
    print("\n=== per asset class: best t (defined | undefined) and survivors ===")
    for ac in sorted(sv["ac"].dropna().unique()):
        d = sv[(sv["ac"] == ac) & (sv["risk"] == "defined")]; u = sv[(sv["ac"] == ac) & (sv["risk"] == "undefined")]
        print(f"  {ac:8s} defined best_t={d['t'].max():5.2f} (surv {int((d['fdr_pass']&d['dsr_pass']).sum())}) | "
              f"undefined best_t={u['t'].max():5.2f} (surv {int((u['fdr_pass']&u['dsr_pass']).sum())})")
    # focused deflation: best DEFINED-risk cell vs a defined-risk-only trial family
    print("\n=== focused DSR: does restricting to defined-risk rescue the best defined cell? ===")
    dfm = sv[sv["risk"] == "defined"]
    top = dfm.sort_values("t", ascending=False).iloc[0]
    fam = dfm.dropna(subset=["sr_trade"])["sr_trade"].values
    d, sr0 = dsr_pvalue(top["sr_trade"], top["skew"], top["kurt"], int(top["n"]), fam)
    allfam = s.dropna(subset=["sr_trade"])["sr_trade"].values
    d_all, _ = dsr_pvalue(top["sr_trade"], top["skew"], top["kurt"], int(top["n"]), allfam)
    print(f"  best defined cell: {top['instrument']} {top['archetype']} t={top['t']:.2f} sr_trade={top['sr_trade']:.2f} "
          f"skew={top['skew']:.2f}")
    print(f"    DSR vs all-corpus family (N={len(allfam)}): {d_all:.3f} | vs defined-risk-only family "
          f"(N={len(fam)}, sr0={sr0:.2f}): {d:.3f}  -> {'PASSES' if d>0.95 else 'still fails'}")
    sv[["instrument", "archetype", "ac", "risk", "n", "oos_net", "t", "skew", "kurt", "dsr", "fdr_pass", "dsr_pass"]
       ].to_csv(os.path.join(OUT, "defined_risk.csv"), index=False)
    nd = int((sv["risk"] == "defined").sum()); ndsurv = int((sv[sv.risk == "defined"]["fdr_pass"] & sv[sv.risk == "defined"]["dsr_pass"]).sum())
    print(f"\nSUMMARY: {nd} defined-risk short-vol cells, {ndsurv} survivors; "
          f"{int((sv.risk=='undefined').sum())} undefined, {int((sv[sv.risk=='undefined']['fdr_pass']&sv[sv.risk=='undefined']['dsr_pass']).sum())} survivors.")
    print("DONE")


if __name__ == "__main__":
    main()
