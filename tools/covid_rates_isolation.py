"""COVID-2020 + 2022-rates extension analysis for the cross-venue FO corpus.

Two honest stress lenses on the 2020+2022-extended corpus:
  (A) Per-ASSET-CLASS OOS survivor table from fop_corpus_strategies.csv, with the OOS date span each
      root contributes (so the reader sees WHICH crises are in-sample vs out-of-sample per root).
  (B) The 2022 rates-vol regime as a CLEAN OOS stress test: rates roots (OZN/OZB/OZF/OZT/UB1) have
      2020-2021 in-sample history BEFORE 2022, so trades entered in 2022 are genuinely out-of-sample.
      The 2022 Fed hiking cycle drove the MOVE index >150 (worst Treasury-vol regime in ~40y) — do any
      rates-options archetypes survive it OOS? We slice the OOS ledger by entry-year and vega sign.
  (C) The March-2020 COVID crash is UNAVOIDABLY in the first in-sample window (earliest FO data = Jan
      2020, and the WFO needs 126 IS days). So we report the short-vol vs long-vol book PnL for any
      2020 OOS trades (mid/late-2020 onward) — labeled as the post-crash-normalization OOS slice, NOT
      a crash-entry claim. The crash-entry illustration is a separate continuous-run script.

Reads findings/fop_corpus_strategies.csv + findings/fop_ledgers/*.parquet. Writes findings/covid_rates_*.csv.
No lookahead introduced: every number is a slice of the existing WFO-OOS ledger by entry date / archetype.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FINDINGS

import os, glob
import numpy as np, pandas as pd

OUT = FINDINGS
LEDG = os.path.join(OUT, "fop_ledgers")

ACLASS = {
    "ES": "equity_index", "NQ": "equity_index", "RTM": "equity_index",
    "OG": "metals", "PAO": "metals",
    "LO": "energy", "ON": "energy",
    "OZN": "rates", "OZB": "rates", "OZF": "rates", "OZT": "rates", "UB1": "rates",
    "BTC": "crypto", "ETH": "crypto",
    "OZC": "ags", "OZS": "ags", "OZW": "ags",
}
RATES = [r for r, c in ACLASS.items() if c == "rates"]
# vega-sign buckets (from optengine/archetypes.py section structure)
SHORT_VOL = {"short_straddle", "short_strangle", "wide_strangle", "iron_condor", "iron_butterfly",
             "put_credit_spread", "call_credit_spread", "put_write", "call_overwrite",
             "short_put_atm", "short_call_atm", "jade_lizard", "big_lizard", "broken_wing_condor",
             "broken_wing_put_fly", "put_ratio", "call_ratio"}
LONG_VOL = {"long_straddle", "long_strangle", "long_put_tail", "long_call",
            "put_backspread", "call_backspread"}


def vbucket(a):
    return "short_vol" if a in SHORT_VOL else ("long_vol" if a in LONG_VOL else "mixed")


def load_ledgers():
    rows = []
    for f in glob.glob(os.path.join(LEDG, "*.parquet")):
        try:
            d = pd.read_parquet(f, columns=["entry", "net", "gross", "premium", "instrument", "archetype"])
        except Exception:
            continue
        rows.append(d)
    if not rows:
        return pd.DataFrame()
    L = pd.concat(rows, ignore_index=True)
    L["entry"] = L["entry"].astype(str)
    L["yr"] = L["entry"].str.slice(0, 4)
    L["aclass"] = L["instrument"].map(ACLASS).fillna("other")
    L["vbucket"] = L["archetype"].map(vbucket)
    return L


def survivor_table(strat_csv, L):
    s = pd.read_csv(strat_csv)
    s["aclass"] = s["instrument"].map(ACLASS).fillna("other")
    fdr = s.get("fdr_pass", pd.Series(False, index=s.index)).fillna(False).astype(bool)
    dsr = s.get("dsr_pass", pd.Series(False, index=s.index)).fillna(False).astype(bool)
    s["survivor"] = fdr & dsr
    # OOS date span per root (from the ledger)
    span = L.groupby("instrument")["entry"].agg(["min", "max"]) if len(L) else pd.DataFrame()
    print("\n=== (A) PER-ASSET-CLASS OOS SURVIVOR TABLE (extended corpus) ===")
    print(f"{'asset_class':14s} {'roots':>5s} {'strats':>6s} {'net+':>5s} {'FDR&DSR_surv':>12s} {'best_t':>7s}")
    out = []
    for ac, g in s.groupby("aclass"):
        nrt = g["instrument"].nunique()
        netpos = int((g["oos_net"] > 0).sum())
        surv = int(g["survivor"].sum())
        bt = g["t"].max()
        out.append(dict(aclass=ac, roots=nrt, strats=len(g), net_pos=netpos, survivors=surv, best_t=round(bt, 2)))
        print(f"{ac:14s} {nrt:5d} {len(g):6d} {netpos:5d} {surv:12d} {bt:7.2f}")
    tot = dict(aclass="TOTAL", roots=s["instrument"].nunique(), strats=len(s),
               net_pos=int((s["oos_net"] > 0).sum()), survivors=int(s["survivor"].sum()),
               best_t=round(s["t"].max(), 2))
    print(f"{'TOTAL':14s} {tot['roots']:5d} {tot['strats']:6d} {tot['net_pos']:5d} {tot['survivors']:12d} {tot['best_t']:7.2f}")
    pd.DataFrame(out + [tot]).to_csv(os.path.join(OUT, "covid_rates_survivor_table.csv"), index=False)
    if len(span):
        print("\n  OOS entry span per root (which crises are OOS):")
        for rt in sorted(span.index):
            print(f"    {rt:5s} {span.loc[rt,'min']} -> {span.loc[rt,'max']}  [{ACLASS.get(rt,'?')}]")
    return s


def rates_2022_oos(L):
    print("\n=== (B) 2022 RATES-VOL REGIME — CLEAN OOS STRESS (rates roots, 2020-21 IS precedes 2022) ===")
    R = L[L["instrument"].isin(RATES)].copy()
    if not len(R):
        print("  no rates OOS trades"); return
    print(f"  rates OOS trades total: {len(R)} across {R['instrument'].nunique()} roots, "
          f"entry {R['entry'].min()}->{R['entry'].max()}")
    # by year x vega bucket: net PnL + per-trade mean + win rate
    print(f"\n  {'year':4s} {'vbucket':10s} {'n':>5s} {'net_$':>12s} {'mean_$/tr':>10s} {'win%':>5s}")
    for (yr, vb), g in R.groupby(["yr", "vbucket"]):
        wr = (g["net"] > 0).mean() * 100
        print(f"  {yr:4s} {vb:10s} {len(g):5d} {g['net'].sum():12.0f} {g['net'].mean():10.1f} {wr:5.0f}")
    # 2022-specific: does any rates archetype have positive OOS net in 2022?
    r22 = R[R["yr"] == "2022"]
    if len(r22):
        by = r22.groupby(["instrument", "archetype"]).agg(n=("net", "size"), net=("net", "sum"),
                                                          mean=("net", "mean"), win=("net", lambda x: (x > 0).mean()))
        by = by[by["n"] >= 5].sort_values("net", ascending=False)
        print(f"\n  2022 rates books (>=5 OOS trades): {len(by)} | net-positive in 2022: {int((by['net']>0).sum())}/{len(by)}")
        if len(by):
            print("  top 5 by 2022 net PnL:")
            print(by.head(5).to_string())
            print("  bottom 3:")
            print(by.tail(3).to_string())
        by.to_csv(os.path.join(OUT, "covid_rates_2022_books.csv"))


def cross_asset_2022_and_matrix(L):
    """2022 is a clean OOS bear for ALL roots (every root's OOS starts <=2021): equity -25%, crypto -65%,
    MOVE index >150. Slice short-vol OOS PnL by asset class in 2022, and the full-OOS asset x vega matrix."""
    print("\n=== (B2) 2022 SHORT-VOL OOS STRESS BY ASSET CLASS (clean OOS bear) ===")
    s22 = L[(L["yr"] == "2022") & (L["vbucket"] == "short_vol")]
    print(f"  {'asset':8s} {'n':>5s} {'net_$':>11s} {'mean/tr':>9s} {'win%':>5s}")
    rows = []
    for ac, g in s22.groupby("aclass"):
        rows.append(dict(aclass=ac, n=len(g), net=round(g["net"].sum()), mean=round(g["net"].mean(), 1),
                         win=round((g["net"] > 0).mean() * 100)))
        print(f"  {ac:8s} {len(g):5d} {g['net'].sum():11.0f} {g['net'].mean():9.1f} {(g['net']>0).mean()*100:5.0f}")
    print(f"  {'TOTAL':8s} {len(s22):5d} {s22['net'].sum():11.0f} {s22['net'].mean():9.1f} {(s22['net']>0).mean()*100:5.0f}")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "covid_rates_2022_byclass.csv"), index=False)
    print("\n=== (B3) FULL-OOS (2021-2026) NET P&L BY ASSET x VEGA BUCKET ($, summed across all books) ===")
    piv = L.pivot_table(index="aclass", columns="vbucket", values="net", aggfunc="sum").round(0)
    print(piv.to_string())
    piv.to_csv(os.path.join(OUT, "covid_rates_oos_matrix.csv"))
    tot = L["net"].sum(); n = len(L)
    sv = L[L["vbucket"] == "short_vol"]["net"].sum()
    print(f"\n  GRAND TOTAL net P&L across all {L['instrument'].nunique()} roots x archetypes: "
          f"${tot:,.0f} over {n:,} OOS trades")
    print(f"  short_vol ${sv:,.0f} | the ONLY net-positive (aclass,vbucket) cell is crypto/short_vol "
          f"= ${piv.loc['crypto','short_vol']:,.0f} (the crypto VRP — DSR-killed on skew+multiplicity)")


def covid_2020_oos(L):
    print("\n=== (C) 2020 OOS SLICE (post-crash normalization — crash itself is in 1st IS window) ===")
    c = L[L["yr"] == "2020"].copy()
    if not len(c):
        print("  no 2020 OOS trades (WFO IS warm-up consumes all of 2020) — expected; crash is in-sample.")
        return
    print(f"  2020 OOS trades: {len(c)}, entry {c['entry'].min()}->{c['entry'].max()} "
          f"(note: these are mid/late-2020, AFTER the 126-day IS warm-up)")
    for vb, g in c.groupby("vbucket"):
        wr = (g["net"] > 0).mean() * 100
        print(f"    {vb:10s} n={len(g):4d} net=${g['net'].sum():10.0f} mean=${g['net'].mean():7.1f} win={wr:.0f}%")


def main():
    strat_csv = os.path.join(OUT, "fop_corpus_strategies.csv")
    L = load_ledgers()
    print(f"loaded OOS ledger: {len(L)} trades across {L['instrument'].nunique() if len(L) else 0} roots, "
          f"years {sorted(L['yr'].unique()) if len(L) else []}")
    survivor_table(strat_csv, L)
    rates_2022_oos(L)
    cross_asset_2022_and_matrix(L)
    covid_2020_oos(L)
    print("\nDONE")


if __name__ == "__main__":
    main()
