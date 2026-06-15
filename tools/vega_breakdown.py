"""Vega-sign breakdown of the mega-corpus — the dimension the 35-archetype run adds over
the short-vol-only base study. Splits the cross-section into clean SHORT-vol (sell premium,
short gamma/vega) vs LONG-vol/convexity (buy premium) vs term-structure (calendars/diagonals,
net long back-month vega) vs pin/fly vs directional, and reports net-of-cost survival,
multiplicity-deflated survivors, and the RETURN-SKEW sign per bucket (a validation: premium
selling should be negatively skewed — pennies then a crash — and long convexity positively
skewed). Reads findings/corpus_deflated.csv (post-deflate). RAM-light, read-only.
"""

from optengine.config import FINDINGS

import os
import numpy as np, pandas as pd

OUT = FINDINGS

BUCKET = {
    # SHORT vol — sell premium, short gamma/vega (the base-study thesis set)
    "short_straddle": "short_vol", "short_strangle": "short_vol", "wide_strangle": "short_vol",
    "iron_condor": "short_vol", "iron_butterfly": "short_vol", "put_credit_spread": "short_vol",
    "call_credit_spread": "short_vol", "put_write": "short_vol", "call_overwrite": "short_vol",
    "short_put_atm": "short_vol", "short_call_atm": "short_vol", "put_ratio": "short_vol",
    "call_ratio": "short_vol", "jade_lizard": "short_vol", "big_lizard": "short_vol",
    "broken_wing_condor": "short_vol", "broken_wing_put_fly": "short_vol",
    # LONG vol / convexity — buy premium, long gamma/vega
    "long_straddle": "long_vol", "long_strangle": "long_vol", "long_put_tail": "long_vol",
    "long_call": "long_vol", "put_backspread": "long_vol", "call_backspread": "long_vol",
    # TERM structure — calendars / diagonals (net long back-month vega)
    "put_calendar": "term", "call_calendar": "term", "double_calendar": "term",
    "put_diagonal": "term", "call_diagonal": "term",
    # PIN / fly — long debit butterfly (short realized-vol / pin bet, small vega)
    "call_butterfly": "pin_fly", "put_butterfly": "pin_fly",
    # DIRECTIONAL / skew — delta-dominant, mixed vega
    "put_debit_spread": "directional", "call_debit_spread": "directional", "collar": "directional",
    "risk_reversal": "directional", "reverse_risk_reversal": "directional",
}
ORDER = ["short_vol", "long_vol", "term", "pin_fly", "directional"]


def main():
    import sys
    csv = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUT, "corpus_deflated.csv")
    s = pd.read_csv(csv if os.path.isabs(csv) else os.path.join(OUT, csv))
    s["bucket"] = s["archetype"].map(BUCKET).fillna("unmapped")
    surv_col = "robust_sig" if "robust_sig" in s.columns else None
    print(f"corpus_deflated: {len(s)} cells across {s['instrument'].nunique()} instruments, "
          f"{s['archetype'].nunique()} archetypes\n")
    rows = []
    for b in ORDER + [x for x in s["bucket"].unique() if x not in ORDER]:
        g = s[s["bucket"] == b]
        if not len(g):
            continue
        rows.append(dict(
            bucket=b, n_cells=len(g), archetypes=g["archetype"].nunique(),
            pct_net_pos=round(100 * (g["oos_net"] > 0).mean(), 1),
            med_oos_net=round(float(g["oos_net"].median())),
            med_t=round(float(g["t"].median()), 2),
            med_skew=round(float(g["skew"].median()), 2) if "skew" in g else np.nan,
            fdr=int(g["fdr_pass"].sum()) if "fdr_pass" in g else 0,
            dsr_eff=int(g["dsr_eff_pass"].sum()) if "dsr_eff_pass" in g else 0,
            survivors=int(g[surv_col].sum()) if surv_col else 0,
        ))
    rep = pd.DataFrame(rows)
    print("=== BY VEGA BUCKET ===")
    print(rep.to_string(index=False))
    print("\nSKEW VALIDATION: short_vol med_skew should be NEGATIVE (pennies then crash); "
          "long_vol POSITIVE (pay premium, occasional big win).")
    print("\n=== PER-ARCHETYPE (sorted by %net-positive) ===")
    pa = s.groupby(["bucket", "archetype"]).agg(
        n=("oos_net", "size"), pct_pos=("oos_net", lambda x: round(100 * (x > 0).mean(), 1)),
        med_net=("oos_net", "median"), med_t=("t", "median"),
        med_skew=("skew", "median") if "skew" in s else ("oos_net", "size")).reset_index()
    pa = pa.sort_values("pct_pos", ascending=False)
    print(pa.to_string(index=False))
    rep.to_csv(os.path.join(OUT, "vega_breakdown.csv"), index=False)
    print(f"\nwrote {OUT}/vega_breakdown.csv")


if __name__ == "__main__":
    main()
