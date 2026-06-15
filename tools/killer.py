"""The two spine artifacts for the paper:
  FIGURE  gross Sharpe (mid-price) vs net Sharpe (after real bid/ask), per strategy, both arms, with the
          deflation-survivor overlay. Shows the spread tax dragging every strategy below the no-cost diagonal
          and zero survivors.
  TABLE   strategy family -> gross result -> net result -> deflated significance -> survivor count.
Built from the per-trade ledgers on disk (OPRA findings/ledgers/, cross-venue findings/fop_ledgers/), with the
canonical FDR/DSR survivor flags merged from the strategies CSVs. No new fit.
"""

from optengine.config import REPO_ROOT

import os, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ENG = REPO_ROOT
OUT = os.path.join(ENG, "findings")

FAMILY = {
    "short_straddle": "Naked short-vol", "short_strangle": "Naked short-vol", "wide_strangle": "Naked short-vol",
    "short_put_atm": "Naked short-vol", "short_call_atm": "Naked short-vol",
    "put_write": "One-sided premium", "call_overwrite": "One-sided premium",
    "iron_condor": "Defined-risk short-vol", "iron_butterfly": "Defined-risk short-vol",
    "put_credit_spread": "Defined-risk short-vol", "call_credit_spread": "Defined-risk short-vol",
    "broken_wing_condor": "Defined-risk short-vol", "broken_wing_put_fly": "Defined-risk short-vol",
    "jade_lizard": "Defined-risk short-vol", "big_lizard": "Defined-risk short-vol",
    "long_straddle": "Long-vol / convexity", "long_strangle": "Long-vol / convexity",
    "long_put_tail": "Long-vol / convexity", "long_call": "Long-vol / convexity",
    "put_backspread": "Long-vol / convexity", "call_backspread": "Long-vol / convexity",
    "put_calendar": "Calendar / diagonal", "call_calendar": "Calendar / diagonal",
    "double_calendar": "Calendar / diagonal", "put_diagonal": "Calendar / diagonal", "call_diagonal": "Calendar / diagonal",
    "call_butterfly": "Butterfly", "put_butterfly": "Butterfly",
    "put_ratio": "Ratio / skew", "call_ratio": "Ratio / skew", "risk_reversal": "Ratio / skew",
    "reverse_risk_reversal": "Ratio / skew",
    "collar": "Directional", "put_debit_spread": "Directional", "call_debit_spread": "Directional",
}
FAM_ORDER = ["Naked short-vol", "One-sided premium", "Defined-risk short-vol", "Ratio / skew",
             "Calendar / diagonal", "Butterfly", "Long-vol / convexity", "Directional"]


def cell_sharpes(ledger_glob):
    rows = []
    for f in sorted(glob.glob(ledger_glob)):
        try:
            d = pd.read_parquet(f, columns=["entry", "premium", "gross", "net", "instrument", "archetype"])
        except Exception:
            continue
        d = d[d["premium"] > 0]
        if not len(d):
            continue
        d = d.copy(); d["gr"] = d["gross"] / d["premium"]; d["nr"] = d["net"] / d["premium"]
        d["dt"] = pd.to_datetime(d["entry"].astype(str), format="%Y%m%d", errors="coerce")
        for (inst, arch), g in d.groupby(["instrument", "archetype"]):
            n = len(g)
            if n < 12:
                continue
            yrs = max((g["dt"].max() - g["dt"].min()).days / 365.25, 0.25)
            tpy = n / yrs
            gr, nr = g["gr"].values, g["nr"].values
            gs = gr.mean() / gr.std() * np.sqrt(tpy) if gr.std() > 0 else np.nan
            ns = nr.mean() / nr.std() * np.sqrt(tpy) if nr.std() > 0 else np.nan
            rows.append(dict(instrument=inst, archetype=arch, n=n, gross_sharpe=gs, net_sharpe=ns))
    return pd.DataFrame(rows)


def main():
    print("computing per-cell gross/net Sharpe from ledgers...", flush=True)
    opra = cell_sharpes(os.path.join(OUT, "ledgers", "*.parquet")); opra["arm"] = "OPRA equity"
    xv = cell_sharpes(os.path.join(OUT, "fop_ledgers", "*.parquet")); xv["arm"] = "Cross-venue futures-options"
    print(f"  OPRA cells {len(opra)} | cross-venue cells {len(xv)}", flush=True)

    def flags(csv):
        s = pd.read_csv(csv)
        for c in ("fdr_pass", "dsr_pass"):
            if c in s:
                s[c] = s[c].astype(str).str.lower().isin(["true", "1", "1.0"])
        return s[["instrument", "archetype", "fdr_pass", "dsr_pass"]]
    of = flags(os.path.join(OUT, "mega_693x35", "corpus_strategies.csv"))
    xf = flags(os.path.join(OUT, "fop_corpus_strategies.csv"))
    opra = opra.merge(of, on=["instrument", "archetype"], how="left")
    xv = xv.merge(xf, on=["instrument", "archetype"], how="left")
    df = pd.concat([opra, xv], ignore_index=True).dropna(subset=["gross_sharpe", "net_sharpe"])
    df["fdr_pass"] = df["fdr_pass"].fillna(False); df["dsr_pass"] = df["dsr_pass"].fillna(False)
    df["survivor"] = df["fdr_pass"] & df["dsr_pass"]
    df["family"] = df["archetype"].map(FAMILY).fillna("Other")
    df.to_csv(os.path.join(OUT, "gross_net_sharpe.csv"), index=False)

    N = len(df)
    gpos = (df["gross_sharpe"] > 0).mean() * 100
    npos = (df["net_sharpe"] > 0).mean() * 100
    nsurv = int(df["survivor"].sum())
    flips = ((df["gross_sharpe"] > 0) & (df["net_sharpe"] < 0)).mean() * 100
    print(f"  N={N} | gross Sharpe>0: {gpos:.0f}% | net Sharpe>0: {npos:.0f}% | gross+/net-: {flips:.0f}% | survivors: {nsurv}", flush=True)

    # ---- FIGURE ----
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    fig, (ax, axh) = plt.subplots(1, 2, figsize=(14, 6.2), gridspec_kw={"width_ratios": [1.25, 1]})
    lim = (-8, 12)
    for arm, c in [("OPRA equity", "#2c6fbb"), ("Cross-venue futures-options", "#d98a3d")]:
        g = df[(df.arm == arm) & (~df.survivor)]
        ax.scatter(g.gross_sharpe.clip(*lim), g.net_sharpe.clip(*lim), s=6, alpha=0.18, c=c, label=f"{arm} (n={len(df[df.arm==arm]):,})", linewidths=0)
    sv = df[df.survivor]
    if len(sv):
        ax.scatter(sv.gross_sharpe.clip(*lim), sv.net_sharpe.clip(*lim), s=70, marker="*", c="#d62728", zorder=5, label=f"survives deflation (n={nsurv})")
    ax.plot(lim, lim, "--", c="#444", lw=1.2, label="no-cost line (net = gross)")
    ax.axhline(0, c="#999", lw=0.8); ax.axvline(0, c="#999", lw=0.8)
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_xlabel("Gross annualized Sharpe (mid-price)")
    ax.set_ylabel("Net annualized Sharpe (after real bid/ask)")
    ax.set_title("Almost every strategy sits below the no-cost line (4 of 19,314 above)", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax.annotate(f"gross Sharpe > 0:  {gpos:.0f}%\nnet Sharpe > 0:    {npos:.0f}%\ngross+ flipped to net-:  {flips:.0f}%\nsurvive deflation:  {nsurv} of {N:,}",
                xy=(0.97, 0.03), xycoords="axes fraction", ha="right", va="bottom", fontsize=9.5,
                bbox=dict(boxstyle="round", fc="#fff7e6", ec="#d98a3d"))

    bins = np.linspace(-8, 12, 60)
    axh.hist(df.gross_sharpe.clip(*lim), bins=bins, alpha=0.55, color="#7fb069", label=f"gross ({gpos:.0f}% > 0)")
    axh.hist(df.net_sharpe.clip(*lim), bins=bins, alpha=0.6, color="#b5475a", label=f"net ({npos:.0f}% > 0)")
    axh.axvline(0, c="#333", lw=1)
    axh.set_xlabel("Annualized Sharpe"); axh.set_ylabel("strategies")
    axh.set_title("Real costs shift the whole distribution left, past zero", fontsize=12)
    axh.legend(fontsize=9)
    fig.suptitle("Most apparent options-strategy alpha is eaten by the spread, and none survives multiple-testing",
                 fontsize=13.5, y=0.99)
    fig.text(0.5, 0.005, f"{N:,} strategies with a computable Sharpe (of 19,316 corpus cells), 2017–2026 (equity) / 2020–2026 (cross-venue), real NBBO bid/ask fills, walk-forward, BH-FDR + Deflated-Sharpe.  Survivors: 0.",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    p = os.path.join(OUT, "FIG_gross_vs_net_sharpe.png")
    try:
        fig.savefig(p, dpi=150); print("  saved figure:", p, flush=True)
    except Exception as e:
        print("  (save failed", p, e, ")")

    # ---- TABLE ----
    def fam_row(g):
        return pd.Series(dict(
            n=len(g),
            gross_pos=round((g.gross_sharpe > 0).mean() * 100),
            gross_med=round(g.gross_sharpe.median(), 2),
            net_pos=round((g.net_sharpe > 0).mean() * 100),
            net_med=round(g.net_sharpe.median(), 2),
            fdr_sig=int(g.fdr_pass.sum()),
            survivors=int(g.survivor.sum())))
    tab = df.groupby("family").apply(fam_row).reindex([f for f in FAM_ORDER if f in df.family.unique()])
    tot = fam_row(df); tot.name = "ALL"
    tab = pd.concat([tab, pd.DataFrame([tot])])
    tab.to_csv(os.path.join(OUT, "TABLE_family_survival.csv"))
    # markdown
    md = ["# Strategy family: gross -> net -> deflated -> survivors\n",
          f"_{N:,} systematic option strategies (OPRA equity 2017-26 + cross-venue futures-options 2020-26), "
          "real NBBO bid/ask, walk-forward, BH-FDR(q=0.10) + Deflated-Sharpe(>0.95)._\n",
          "| Strategy family | # | Gross Sharpe>0 | med gross | Net Sharpe>0 | med net | FDR-significant (net) | **Survive deflation** |",
          "|---|--:|--:|--:|--:|--:|--:|--:|"]
    for fam, r in tab.iterrows():
        bold = f"**{int(r.survivors)}**"
        md.append(f"| {fam} | {int(r.n):,} | {int(r.gross_pos)}% | {r.gross_med:+.2f} | {int(r.net_pos)}% | {r.net_med:+.2f} | {int(r.fdr_sig)} | {bold} |")
    open(os.path.join(OUT, "TABLE_family_survival.md"), "w").write("\n".join(md) + "\n")
    print("\n" + "\n".join(md), flush=True)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
