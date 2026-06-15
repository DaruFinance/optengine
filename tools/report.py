"""Cross-sectional analysis of the corpus run, for the writeup Results section.
Reads findings/corpus_{raw,strategies}.csv; prints the survival summary + per-archetype
and per-instrument-class breakdowns + the top deflated survivors."""

from optengine.config import FINDINGS

import os, glob
import numpy as np, pandas as pd

OUT = FINDINGS
IDX_ETF = {"SPXW", "SPX", "NDX", "RUT", "SPY", "QQQ", "IWM", "DIA", "XLF", "XLE", "XLK",
           "XLV", "XLI", "XLU", "XLP", "XLY", "XLB", "XLRE", "XLC", "TLT", "HYG", "LQD",
           "GLD", "SLV", "USO", "UNG", "EEM", "FXI", "VXX", "UVXY", "SMH", "XBI", "KRE",
           "GDX", "EWZ", "XOP", "ARKK", "TQQQ", "SQQQ"}


def cls(t):
    return "index/ETF" if t in IDX_ETF else "single-name"


def main():
    pd.set_option("display.width", 200)
    raw = pd.read_csv(os.path.join(OUT, "corpus_raw.csv"))
    s = pd.read_csv(os.path.join(OUT, "corpus_strategies.csv"))
    s["cls"] = s["instrument"].map(cls)
    has = lambda c: c in s.columns

    print("=== CORPUS SUMMARY ===")
    print(f"strategies attempted: {len(raw)}   with >=12 OOS trades: {len(s)}")
    if has("fdr_pass"):
        print(f"  survive BH-FDR (q=.10):        {int(s['fdr_pass'].sum())}")
    if has("dsr_pass"):
        print(f"  survive Deflated-Sharpe (>.95): {int(s['dsr_pass'].sum())}")
    if has("fdr_pass") and has("dsr_pass"):
        print(f"  survive BOTH:                   {int((s['fdr_pass'] & s['dsr_pass']).sum())}")
    print(f"  median net (all strategies): ${s['oos_net'].median():,.0f}   "
          f"% net-positive: {(s['oos_net'] > 0).mean():.0%}")

    agg = {"n": ("t", "size"), "med_t": ("t", "median"),
           "pos_net": ("oos_net", lambda x: (x > 0).mean())}
    if has("dsr_pass"):
        agg["dsr_pass"] = ("dsr_pass", "sum")
    print("\n=== per archetype (sorted by median t) ===")
    print(s.groupby("archetype").agg(**agg).sort_values("med_t", ascending=False).round(3).to_string())
    print("\n=== per instrument class ===")
    print(s.groupby("cls").agg(**agg).round(3).to_string())

    # --- gross-vs-net spread tax, from the per-trade ledgers ---
    lp = glob.glob(os.path.join(OUT, "ledgers", "*.parquet"))
    if lp:
        led = pd.concat([pd.read_parquet(p, columns=["instrument", "archetype", "gross", "net", "premium"])
                         for p in lp], ignore_index=True)
        led["cls"] = led["instrument"].map(cls)
        led["tax"] = led["gross"] - led["net"]
        print(f"\n=== gross-vs-net spread tax ({len(led):,} OOS trades) — per instrument class ===")
        g = led.groupby("cls").agg(trades=("net", "size"), gross=("gross", "sum"),
                                   net=("net", "sum"), tax=("tax", "sum"))
        g["net/gross"] = (g["net"] / g["gross"]).round(2)
        g["tax/trade"] = (g["tax"] / g["trades"]).round(0)
        print(g.round(0).to_string())
        print("\n=== net/gross retention by archetype (1.0=costless, <0 flips sign) ===")
        ga = led.groupby("archetype").agg(gross=("gross", "sum"), net=("net", "sum"))
        ga["net/gross"] = (ga["net"] / ga["gross"]).round(2)
        print(ga.sort_values("net/gross").round(0).to_string())

    print("\n=== TOP 25 strategies by t-stat (with deflation flags) ===")
    cols = [c for c in ["instrument", "archetype", "cls", "n", "oos_net", "win", "t",
                        "sharpe_ann", "dsr", "fdr_pass", "worst"] if c in s.columns]
    print(s.sort_values("t", ascending=False).head(25)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
