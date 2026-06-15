"""Cross-sectional characteristic-conditioned long-short — confronts the option-return-predictability
literature (Goyal-Saretto 2008; Zhan-Han-Cao-Tong RFS 2022; Bali-Beckmeyer-Moerke-Weigert RFS 2023)
under this engine's real-quoted-spread + deflation discipline.

CONSTRUCTION (corrected — level-premium-free):
  - For one delta-hedged option position type (the literature's object), each name/entry has a realised
    return = trade net / entry premium, GROSS (mid) and NET (real spread); cost = rgross - rnet >= 0.
  - The cross-sectional long-short is formed from the SAME single return series so the common vol-premium
    LEVEL cancels in the difference and only the characteristic's cross-sectional signal remains:
        LS_gross(month) = mean(rgross | top-quintile) - mean(rgross | bottom-quintile)
        LS_net          = LS_gross - mean(cost | top) - mean(cost | bottom)   [spread paid on both legs]
    (Using a long ledger for the long leg and a SEPARATELY-simulated short ledger for the short leg would
    smuggle the short-vol level premium into the "signal" — a random characteristic would then score ~2.7
    Sharpe. This single-series construction makes a placebo score ~0, isolating the characteristic.)
  - A PLACEBO characteristic (deterministic pseudo-random per name/month) is run as the noise floor.
  - Two position types: delta-hedged STRADDLE (long_straddle) and delta-hedged SINGLE CALL (long_call, the
    Zhan/Bali object). Both directions, gross+net, deflated across all (char x dir x struct) trials, + a
    spread-capture sweep (at what execution quality, if any, does a conditioned L-S survive?).
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import CHAR_PANEL, FINDINGS, LEDGERS

import os, sys, glob, time, hashlib
import numpy as np, pandas as pd
from scipy import stats as ss
from corpus import dsr_pvalue

LEDG = LEDGERS
PANEL = CHAR_PANEL
OUT = FINDINGS
CHARS = ["ivrv", "atm_iv", "rv21", "skew25", "volvol", "mom21", "ivts", "placebo"]
STRUCTS = {"straddle": "long_straddle", "single_call": "long_call"}
QUINT = 0.20
MINQ = 5


def load_returns(archetype):
    rows = []
    for f in glob.glob(os.path.join(LEDG, "*.parquet")):
        try:
            d = pd.read_parquet(f, columns=["entry", "archetype", "premium", "gross", "net", "instrument"])
        except Exception:
            continue
        d = d[(d["archetype"] == archetype) & (d["premium"] > 0)]
        if len(d):
            rows.append(d)
    L = pd.concat(rows, ignore_index=True)
    L["entry"] = L["entry"].astype(int)
    L["rgross"] = L["gross"] / L["premium"]
    L["rnet"] = L["net"] / L["premium"]
    for c in ("rgross", "rnet"):
        lo, hi = L[c].quantile([0.005, 0.995]); L[c] = L[c].clip(lo, hi)
    L["cost"] = L["rgross"] - L["rnet"]                 # full-taker spread+commission, per premium (>=0)
    return L[["instrument", "entry", "rgross", "rnet", "cost"]]


def _ph(s):
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


def xs_ls(M, char, capture=1.0):
    """Monthly cross-sectional long-short from a single return series. Returns gross series, net series
    (direction A=high-minus-low and B=low-minus-high), at the given spread capture."""
    Ag, An, Bn = [], [], []
    for mon, g in M.groupby("month", sort=True):
        g = g.dropna(subset=[char])
        if g["instrument"].nunique() < 2 * MINQ:
            continue
        g = g.drop_duplicates("instrument")
        hi = g[char].quantile(1 - QUINT); lo = g[char].quantile(QUINT)
        top = g[g[char] >= hi]; bot = g[g[char] <= lo]
        if len(top) < MINQ or len(bot) < MINQ:
            continue
        lsg = top["rgross"].mean() - bot["rgross"].mean()        # LEVEL CANCELS (same series)
        cc = capture * (top["cost"].mean() + bot["cost"].mean())  # spread paid on both legs
        Ag.append(lsg); An.append(lsg - cc); Bn.append(-lsg - cc)
    return np.array(Ag), np.array(An), np.array(Bn)


def stat(x):
    x = x[np.isfinite(x)]
    if len(x) < 12:
        return None
    mu = x.mean(); sd = x.std(ddof=1)
    if sd <= 0:
        return None
    return dict(n=len(x), sharpe=(mu / sd) * np.sqrt(12), t=mu / (sd / np.sqrt(len(x))),
                sr_trade=mu / sd, skew=float(ss.skew(x)), kurt=float(ss.kurtosis(x, fisher=False)))


def deflate(df):
    if not len(df):
        return df
    trial = df["sr_trade"].to_numpy()
    df = df.copy()
    df["p"] = ss.t.sf(df["t"], df["n"] - 1)              # one-sided: significantly PROFITABLE only
    m = len(df); order = np.argsort(df["p"].to_numpy()); ps = df["p"].to_numpy()[order]
    bh = ps <= (np.arange(1, m + 1) / m) * 0.10
    kmax = (np.where(bh)[0].max() + 1) if bh.any() else 0
    fp = np.zeros(m, bool)
    if kmax:
        fp[order[:kmax]] = True
    df["fdr_pass"] = fp
    df["dsr"] = [dsr_pvalue(r.sr_trade, r.skew, r.kurt, r.n, trial)[0] for r in df.itertuples()]
    df["dsr_pass"] = df["dsr"] > 0.95
    return df


def main():
    t0 = time.time()
    panel = pd.read_parquet(PANEL).rename(columns={"asof_i": "entry"})
    rows, capture_rows = [], []
    for sname, arch in STRUCTS.items():
        L = load_returns(arch)
        M = L.merge(panel, on=["instrument", "entry"], how="left")
        M["month"] = M["entry"] // 100
        M["placebo"] = [_ph(f"{i}_{m}") for i, m in zip(M["instrument"], M["month"])]
        print(f"\n=== {sname} ({arch}): {len(M)} trades, {M.instrument.nunique()} names, {M.month.nunique()} months [{time.time()-t0:.0f}s] ===", flush=True)
        net_recs = []
        for ch in CHARS:
            Ag, An, Bn = xs_ls(M, ch)
            sg = stat(Ag)
            for dname, xn in [("A_hi_minus_lo", An), ("B_lo_minus_hi", Bn)]:
                sn = stat(xn)
                if sg and sn:
                    rec = dict(struct=sname, char=ch, direction=dname, gross_sharpe=round(sg["sharpe"], 2),
                               gross_t=round(sg["t"], 2), net_sharpe=round(sn["sharpe"], 2),
                               net_t=round(sn["t"], 2), t=sn["t"], n=sn["n"], sr_trade=sn["sr_trade"],
                               skew=sn["skew"], kurt=sn["kurt"])
                    rows.append(rec)
                    if ch != "placebo":
                        net_recs.append(rec)
        nd = deflate(pd.DataFrame(net_recs))
        # placebo noise floor as a DISTRIBUTION over 50 random seeds (not a single draw)
        pl_abs = []
        for seed in range(50):
            M["_pl"] = [_ph(f"{i}_{m}_{seed}") for i, m in zip(M["instrument"], M["month"])]
            Ag, _, _ = xs_ls(M, "_pl")
            s = stat(Ag)
            if s:
                pl_abs.append(abs(s["sharpe"]))
        pl_abs = np.array(pl_abs)
        print(f"  PLACEBO gross |Sharpe| over 50 seeds: mean {pl_abs.mean():.2f}, p95 {np.percentile(pl_abs,95):.2f}, max {pl_abs.max():.2f} (noise floor ~0)", flush=True)
        # best real characteristic by gross Sharpe MAGNITUDE across both directions
        gmag = {}
        for ch in [c for c in CHARS if c != "placebo"]:
            a = next((r for r in rows if r["struct"] == sname and r["char"] == ch and r["direction"].startswith("A")), None)
            if a:
                gmag[ch] = abs(a["gross_sharpe"])
        if gmag:
            bch = max(gmag, key=gmag.get)
            ba = next(r for r in rows if r["struct"] == sname and r["char"] == bch and r["direction"].startswith("A"))
            print(f"  best real-char gross |Sharpe|: {bch} {gmag[bch]:.2f} (t={abs(ba['gross_t']):.2f}) -- vs placebo p95 {np.percentile(pl_abs,95):.2f}", flush=True)
        print(f"  {'char':8s} {'grossSh':>7s} {'gross_t':>7s} {'netSh(A)':>8s} {'netSh(B)':>8s} {'bestNet_t':>9s} {'dsr':>6s}", flush=True)
        for ch in [c for c in CHARS if c != "placebo"]:
            a = next((r for r in rows if r["struct"] == sname and r["char"] == ch and r["direction"].startswith("A")), None)
            b = next((r for r in rows if r["struct"] == sname and r["char"] == ch and r["direction"].startswith("B")), None)
            if not a:
                continue
            ndr = nd[(nd.char == ch)] if len(nd) else pd.DataFrame()
            bestdsr = ndr["dsr"].max() if len(ndr) else float("nan")
            bestnt = ndr["net_t"].abs().max() if len(ndr) else float("nan")
            print(f"  {ch:8s} {a['gross_sharpe']:7.2f} {a['gross_t']:7.2f} {a['net_sharpe']:8.2f} {b['net_sharpe']:8.2f} {bestnt:9.2f} {bestdsr:6.3f}", flush=True)
        if len(nd):
            surv = int((nd.fdr_pass & nd.dsr_pass).sum())
            print(f"  NET conditioned L-S: {len(nd)} trials | net-positive Sharpe: {int((nd.net_sharpe>0).sum())} | FDR&DSR survivors: {surv}", flush=True)
        # capture sweep (direction A)
        for c in [0.0, 0.25, 0.5, 0.75, 1.0]:
            recs = []
            for ch in [x for x in CHARS if x != "placebo"]:
                Ag, An, Bn = xs_ls(M, ch, capture=c)
                s = stat(An)
                if s:
                    recs.append(dict(sr_trade=s["sr_trade"], skew=s["skew"], kurt=s["kurt"], n=s["n"],
                                     t=s["t"], net_sharpe=s["sharpe"]))
            rc = deflate(pd.DataFrame(recs))
            sv = int((rc.fdr_pass & rc.dsr_pass).sum()) if len(rc) else 0
            capture_rows.append(dict(struct=sname, capture=c, pos=int((rc.net_sharpe > 0).sum()) if len(rc) else 0,
                                     survivors=sv, best=round(float(rc.net_sharpe.max()), 2) if len(rc) else None))
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cross_sectional_strategies.csv"), index=False)
    pd.DataFrame(capture_rows).to_csv(os.path.join(OUT, "cross_sectional_capture.csv"), index=False)
    print("\ncapture sweep (direction A, net survivors by execution quality):", flush=True)
    for r in capture_rows:
        print(f"  {r['struct']:11s} c={r['capture']:.2f}  pos={r['pos']}  survivors={r['survivors']}  best_netSharpe={r['best']}", flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
