"""Short-vol tail compendium by asset class — on the PER-CELL RETURN-skew the Deflated Sharpe actually consumes
(corpus `skew` = skew of per-trade net/premium), NOT pooled dollar-skew (which mixes heterogeneous-multiplier
roots and diverges from the per-cell statistic). Question: is the short-vol left tail universal across the six
cross-venue classes, and is it what the deflation rejects? Answer: the tail is universal (negative median
per-cell return-skew in all 6 classes), but the near-survivors (top-t cells) have BENIGN tails, so the binding
gate is multiplicity (CROSS_VENUE_v4.md), not the tail. Reads findings/fop_corpus_strategies.csv (+ ledgers for
the dollar realization). Writes findings/tail_compendium.csv.
"""

from optengine.config import FINDINGS

import os, glob
import numpy as np, pandas as pd

OUT = FINDINGS
LEDG = os.path.join(OUT, "fop_ledgers")
AC = {"ES": "equity", "RTM": "equity", "OG": "metals", "PAO": "metals", "LO": "energy", "ON": "energy",
      "OZN": "rates", "OZB": "rates", "OZF": "rates", "OZT": "rates", "UB1": "rates",
      "BTC": "crypto", "ETH": "crypto", "OZC": "ags", "OZS": "ags", "OZW": "ags"}
SHORT = {"short_straddle", "short_strangle", "wide_strangle", "iron_condor", "iron_butterfly",
         "put_credit_spread", "call_credit_spread", "put_write", "call_overwrite", "short_put_atm",
         "short_call_atm", "jade_lizard", "big_lizard", "broken_wing_condor", "broken_wing_put_fly",
         "put_ratio", "call_ratio"}
LONG = {"long_straddle", "long_strangle", "long_put_tail", "long_call", "put_backspread", "call_backspread"}


def main():
    s = pd.read_csv(os.path.join(OUT, "fop_corpus_strategies.csv"))
    s["ac"] = s["instrument"].map(AC)
    sv = s[s["archetype"].isin(SHORT)]
    print("=== PER-CELL RETURN-skew (corpus `skew`, the DSR statistic) — short-vol, by class ===")
    print(f"{'class':8s} {'cells':>5s} {'med_skew':>8s} {'frac<0':>6s} {'frac<-1':>7s} {'med_kurt':>8s}  best-t cell (its skew)")
    rows = []
    for ac, g in sv.groupby("ac"):
        bt = g.sort_values("t", ascending=False).iloc[0]
        d = dict(ac=ac, cells=len(g), med_skew=round(g["skew"].median(), 2), frac_neg=round((g["skew"] < 0).mean(), 2),
                 frac_lt1=round((g["skew"] < -1).mean(), 2), med_kurt=round(g["kurt"].median(), 1),
                 best_t=round(bt["t"], 2), best_arch=bt["archetype"], best_skew=round(bt["skew"], 2))
        rows.append(d)
        print(f"{ac:8s} {len(g):5d} {g['skew'].median():8.2f} {(g['skew']<0).mean()*100:5.0f}% {(g['skew']<-1).mean()*100:6.0f}% "
              f"{g['kurt'].median():8.1f}  {bt['archetype']:14s} ({bt['skew']:+.2f})")
    print(f"ALL short-vol: cells={len(sv)} med_skew={sv['skew'].median():.2f} frac<0={(sv['skew']<0).mean()*100:.0f}% "
          f"frac<-1={(sv['skew']<-1).mean()*100:.0f}%")
    top = sv.sort_values("t", ascending=False).head(8)
    print(f"\n  near-survivors (top-8 by t): median return-skew {top['skew'].median():+.2f} vs all short-vol "
          f"{sv['skew'].median():+.2f} — the best cells are far LESS left-tailed; they die on multiplicity not tail.")
    # dollar realization (ledger) + the single-root share of each class's pooled $ variance (the scale caveat)
    R = [pd.read_parquet(f, columns=["entry", "net", "instrument", "archetype"]) for f in glob.glob(os.path.join(LEDG, "*.parquet"))]
    L = pd.concat(R, ignore_index=True); L["ac"] = L["instrument"].map(AC); L["mo"] = L["entry"].astype(str).str[:6]
    lsv = L[L["archetype"].isin(SHORT)]
    print("\n=== dollar realization (worst calendar month) + top-root variance share (scale caveat) ===")
    for ac, g in lsv.groupby("ac"):
        wm = g.groupby("mo")["net"].sum().min()
        vr = g.groupby("instrument")["net"].var()
        share = (vr.max() / vr.sum()) if vr.sum() > 0 else float("nan")
        topr = vr.idxmax()
        for r in rows:
            if r["ac"] == ac:
                r["worst_month"] = round(float(wm)); r["top_root"] = topr; r["top_root_var_share"] = round(float(share), 2)
        print(f"  {ac:8s} worst_month=${wm:,.0f} | {topr} = {share*100:.0f}% of pooled $ variance")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tail_compendium.csv"), index=False)
    # long side: net-negative everywhere; skew is right at aggregate (backspreads), per-class is a scale artifact
    lv = L[L["archetype"].isin(LONG)]
    print(f"\nlong-vol: net ${lv['net'].sum():,.0f} over {len(lv)} trades, win {(lv['net']>0).mean()*100:.0f}%; "
          f"net-negative in all 6 classes: {all(lv[lv.ac==a]['net'].sum()<0 for a in lv['ac'].dropna().unique())}")
    print("DONE")


if __name__ == "__main__":
    main()
