"""One-shot number report for the definitive (dense-2020) cross-venue cut — every figure CROSS_VENUE_v4
cites, recomputed from the regenerated corpus + ledgers, including the NEW 2020-H2 OOS slice the dense-2020
re-finalize creates (the artifact that round-2 review correctly flagged). Run AFTER fop_corpus.py reruns.
Reads findings/fop_corpus_strategies.csv + findings/fop_ledgers/*.parquet. Pure post-hoc slicing, no lookahead.
"""

from optengine.config import FINDINGS

import os, glob
import numpy as np, pandas as pd
from scipy.stats import norm

OUT = FINDINGS
LEDG = os.path.join(OUT, "fop_ledgers")
ACLASS = {"ES": "equity", "NQ": "equity", "RTM": "equity", "OG": "metals", "PAO": "metals",
          "LO": "energy", "ON": "energy", "OZN": "rates", "OZB": "rates", "OZF": "rates", "OZT": "rates",
          "UB1": "rates", "BTC": "crypto", "ETH": "crypto", "OZC": "ags", "OZS": "ags", "OZW": "ags"}
SHORT = {"short_straddle", "short_strangle", "wide_strangle", "iron_condor", "iron_butterfly",
         "put_credit_spread", "call_credit_spread", "put_write", "call_overwrite", "short_put_atm",
         "short_call_atm", "jade_lizard", "big_lizard", "broken_wing_condor", "broken_wing_put_fly",
         "put_ratio", "call_ratio"}
LONG = {"long_straddle", "long_strangle", "long_put_tail", "long_call", "put_backspread", "call_backspread"}
vb = lambda a: "short_vol" if a in SHORT else ("long_vol" if a in LONG else "mixed")


def load_ledger():
    R = []
    for f in glob.glob(os.path.join(LEDG, "*.parquet")):
        R.append(pd.read_parquet(f, columns=["entry", "net", "gross", "premium", "instrument", "archetype"]))
    L = pd.concat(R, ignore_index=True)
    L["entry"] = L["entry"].astype(str); L["yr"] = L["entry"].str[:4]
    L["ac"] = L["instrument"].map(ACLASS); L["vb"] = L["archetype"].map(vb)
    return L


def main():
    s = pd.read_csv(os.path.join(OUT, "fop_corpus_strategies.csv"))
    L = load_ledger()
    print("=" * 70)
    print("CORPUS COUNTS")
    surv = (s["fdr_pass"] & s["dsr_pass"]).sum()
    fdrp = s["fdr_pass"].sum()
    fdr_losers = ((s["fdr_pass"]) & (s["t"] < 0)).sum(); fdr_win = ((s["fdr_pass"]) & (s["t"] > 0)).sum()
    print(f"  books(n>=12)={len(s)} | net-positive={(s['oos_net']>0).sum()} | FDR&DSR survivors={surv}")
    print(f"  FDR-pass={fdrp} (winners t>0: {fdr_win} | LOSERS t<0: {fdr_losers}) | max DSR={s['dsr'].max():.4f}")
    print(f"  best t={s['t'].max():.2f} | worst t={s['t'].min():.2f} | books t<-2: {(s['t']<-2).sum()}")
    # sr0 benchmark
    tr = s.dropna(subset=["sr_trade"])["sr_trade"].values
    M = len(tr); v = np.var(tr, ddof=1); emc = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - emc) * norm.ppf(1 - 1.0 / M) + emc * norm.ppf(1 - 1.0 / (M * np.e)))
    print(f"  DSR benchmark sr0={sr0:.4f} (M={M}, std(trial)={np.sqrt(v):.4f}, max sr_trade={tr.max():.4f})")
    print("\nPER-ASSET-CLASS")
    s["ac"] = s["instrument"].map(ACLASS)
    for ac, g in s.groupby("ac"):
        print(f"  {ac:8s} roots={g['instrument'].nunique()} books={len(g)} net+={(g['oos_net']>0).sum()} "
              f"surv={int((g['fdr_pass']&g['dsr_pass']).sum())} best_t={g['t'].max():.2f}")
    print("\nMATRIX (full-OOS net $ by asset x vega) + grand total")
    piv = L.pivot_table(index="ac", columns="vb", values="net", aggfunc="sum").round(0)
    print(piv.to_string())
    print(f"  GRAND TOTAL net={L['net'].sum():,.0f} over {len(L):,} trades, {len(L.groupby(['instrument','archetype']))} cells")
    pos = [(a, vbk) for a in piv.index for vbk in piv.columns if piv.loc[a, vbk] > 0]
    print(f"  net-positive cells: {pos}")
    print("\nOOS YEAR COVERAGE (the NEW 2020-H2 slice the dense pass creates)")
    print(f"  earliest OOS entry across all roots: {L['entry'].min()} | years: {sorted(L['yr'].unique())}")
    for r in sorted(L["instrument"].unique()):
        g = L[L["instrument"] == r]
        n20 = (g["yr"] == "2020").sum()
        print(f"    {r:4s} first_OOS={g['entry'].min()} 2020_OOS_trades={n20}")
    h2 = L[L["yr"] == "2020"]
    if len(h2):
        print(f"  >> 2020 OOS slice now NON-empty: {len(h2)} trades, entry {h2['entry'].min()}->{h2['entry'].max()}")
        for vbk, g in h2.groupby("vb"):
            print(f"       {vbk:10s} n={len(g)} net=${g['net'].sum():,.0f} win={(g['net']>0).mean()*100:.0f}%")
    print("\n2022 SHORT-VOL OOS STRESS BY CLASS")
    s22 = L[(L["yr"] == "2022") & (L["vb"] == "short_vol")]
    for ac, g in s22.groupby("ac"):
        print(f"  {ac:8s} n={len(g)} net=${g['net'].sum():,.0f} win={(g['net']>0).mean()*100:.0f}%")
    print(f"  TOTAL n={len(s22)} net=${s22['net'].sum():,.0f} win={(s22['net']>0).mean()*100:.0f}%")
    print("\nCRYPTO short-vol standalone + cell-level skew + regime/root breakdown")
    crs = s[(s["instrument"].isin(["BTC", "ETH"])) & (s["archetype"].isin(SHORT))]
    sk = crs["skew"]
    print(f"  crypto short-vol CELLS={len(crs)}: skew neg={int((sk<0).sum())}/{len(crs)} mean={sk.mean():.2f} "
          f"median={sk.median():.2f} | <-1:{int((sk<-1).sum())} <-5:{int((sk<-5).sum())}")
    for r in ("BTC", "ETH"):
        g = crs[crs["instrument"] == r]
        print(f"    {r} cells={len(g)} skew_neg={int((g['skew']<0).sum())}/{len(g)} mean_skew={g['skew'].mean():.2f} "
              f"mean_kurt={g['kurt'].mean():.1f}")
    cl = L[(L["ac"] == "crypto") & (L["vb"] == "short_vol")].copy()
    net = cl["net"].to_numpy()
    print(f"  POOLED crypto short-vol book: n={len(cl)} net=${net.sum():,.0f} mean=${net.mean():,.0f} "
          f"skew(realized)={pd.Series(net).skew():.2f} win={(net>0).mean()*100:.0f}%")
    cl["mo"] = cl["entry"].str[:6]
    eq = cl.sort_values("entry").groupby("entry")["net"].sum().cumsum()
    dd = (eq - eq.cummax()).min()
    print(f"    max DD (daily-agg entry-order)=${dd:,.0f} ({dd/net.sum()*-100:.1f}% of total) | "
          f"worst month={cl.groupby('mo')['net'].sum().idxmin()} ${cl.groupby('mo')['net'].sum().min():,.0f}")
    print(f"    by ROOT: BTC ${cl[cl.instrument=='BTC']['net'].sum():,.0f} ({len(cl[cl.instrument=='BTC'])} tr) | "
          f"ETH ${cl[cl.instrument=='ETH']['net'].sum():,.0f} ({len(cl[cl.instrument=='ETH'])} tr)")
    print("    by YEAR:")
    for yr, g in cl.groupby("yr"):
        print(f"      {yr}: ${g['net'].sum():,.0f} ({len(g)} tr)")
    print("\nRATES short-vol by OOS year")
    rt = L[(L["ac"] == "rates") & (L["vb"] == "short_vol")]
    for yr, g in rt.groupby("yr"):
        print(f"  {yr}: ${g['net'].sum():,.0f} ({len(g)} tr, win {(g['net']>0).mean()*100:.0f}%)")
    print("DONE")


if __name__ == "__main__":
    main()
