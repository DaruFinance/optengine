"""Intraday 0DTE-ES study — the resolution the daily engine cannot reach.

From the intraday ES short-dated panel (odte_raw/<root>.parquet, a 14-point ET ladder per contract per day),
select 0-DTE contracts (expiry == quote date, by empirical expiry), reconstruct per-(date, snap) the
parity-forward F and Black-76 IV/greeks with the true intraday time-to-settlement T, and backtest short-dated
structures: enter at a chosen ET time crossing the real 1-min bid/ask (ONE-SIDED entry spread + fee),
delta-hedge intraday with the ES future, then HOLD TO SETTLEMENT and settle each leg at INTRINSIC (no closing
spread/fee) using F at the last liquid (~14:30 ET) snap — the canonical 0DTE income trade. ES multiplier 50,
CME fee. Tests short straddle/strangle/iron-condor + the long side, multiple hedge frequencies, deflated.
NOTE: liquid quotes end ~14:30 ET while CME settle is 16:00 ET, so the held-to-settlement result is an UPPER
bound (banks full theta on the last clean F, the unhedged 14:30->16:00 hour is unobservable) — see ODTE.md.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FINDINGS, ODTE_RAW

import os, sys, glob, time
import numpy as np, pandas as pd
from scipy import stats as ss
from optengine.pricing import forward_from_parity, implied_vol, black76_greeks
from corpus import dsr_pvalue

OUT = FINDINGS
MULT, FEE, SLIP = 50.0, 1.5, 5e-5
LADDER = ["09:35", "10:00", "10:30", "11:00", "11:30", "12:00", "12:30",
          "13:00", "13:30", "14:00", "14:30"]   # liquid quotes end ~14:30 ET; 15:00+ are degraded/stale
SETTLE = 16 * 60                      # carry-forwards. True CME ES weekly settle is 16:00 ET (3pm CT); the
                                      # 14:30->16:00 final stretch is unobservable in the data (see ODTE.md caveat)


def _tm(s):
    return int(s[:2]) * 60 + int(s[3:5])


def finalize(df):
    df = df.copy()
    df["mid"] = 0.5 * (df.bid + df.ask)
    df["cpn"] = np.where(df.cp == "C", 1, -1)
    # expiry = last LISTED date of the (Month, ExpYear) contract group (each E-root group is one expiry,
    # listed ~6 weeks then settling; this is robust to per-strike illiquidity quote-gaps).
    grp = df.groupby(["Month", "ExpYear"]).date.max().rename("expiry").reset_index()
    df = df.merge(grp, on=["Month", "ExpYear"])
    gmax = int(df.date.max())
    df = df[df.expiry < gmax]                                  # drop groups still live at the window edge
    df["dte"] = [(pd.Timestamp(str(e)) - pd.Timestamp(str(d))).days for e, d in zip(df.expiry, df.date)]
    return df


def panel_0dte(df):
    """Per (date, snap) among same-day-expiry contracts: parity F + Black-76 iv/delta with intraday T."""
    z = df[df.dte == 0]
    rows = []
    for (d, s), g in z.groupby(["date", "snap"], sort=False):
        ca = g[g.cpn == 1].groupby("Strike").mid.median()
        pu = g[g.cpn == -1].groupby("Strike").mid.median()
        common = ca.index.intersection(pu.index)
        if len(common) < 4:
            continue
        F, _ = forward_from_parity(common.values.astype(float), ca.loc[common].values,
                                   pu.loc[common].values, float(np.median(common)))
        if not (np.isfinite(F) and F > 0):
            continue
        T = max((SETTLE - _tm(s)) / (60 * 24 * 365.0), 1e-6)
        K = g.Strike.values.astype(float); cpn = g.cpn.values
        iv = implied_vol(g.mid.values.astype(float), float(F), K, T, 0.0, 0.0, cpn, model="black76")
        gk = black76_greeks(float(F), K, T, 0.0, np.nan_to_num(iv, nan=0.5), cpn)
        sub = pd.DataFrame(dict(date=int(d), snap=s, tmin=_tm(s), cpn=cpn, Strike=K, F=float(F),
                                iv=iv, delta=gk["delta"], bid=g.bid.values, ask=g.ask.values, mid=g.mid.values))
        rows.append(sub)
    return pd.concat(rows, ignore_index=True) if rows else None


def _pick(chain, cpn, target_delta):
    c = chain[chain.cpn == cpn]
    if not len(c):
        return None
    if target_delta is None:                     # ATM by |delta|->0.5 i.e. nearest strike to F
        j = (c.Strike - c.F.iloc[0]).abs().idxmin()
    else:
        j = (c.delta.abs() - target_delta).abs().idxmin()
    return chain.loc[j]


def run_day(panel_d, legs, entry_snap, hedge_every, capture=1.0):
    """One 0DTE-day. legs: list of (cpn, qty, target_delta). qty<0 short. Returns net P&L or None."""
    snaps = [s for s in LADDER if s in set(panel_d.snap)]
    if entry_snap not in snaps:
        return None
    order = snaps[snaps.index(entry_snap):]
    if len(order) < 2:
        return None
    ch0 = panel_d[panel_d.snap == entry_snap]
    picks = []
    for cpn, qty, td in legs:
        p = _pick(ch0, cpn, td)
        if p is None:
            return None
        picks.append((qty, float(p.Strike), int(p.cpn)))
    def leg_quote(ch, K, cpn):
        r = ch[(ch.Strike == K) & (ch.cpn == cpn)]
        return (float(r.bid.iloc[0]), float(r.ask.iloc[0]), float(r.mid.iloc[0]), float(r.delta.iloc[0])) if len(r) else None
    # entry fills (cross spread by capture)
    net = 0.0; nd = 0.0; prevF = float(ch0.F.iloc[0]); entry = {}
    for qty, K, cpn in picks:
        q = leg_quote(ch0, K, cpn)
        if q is None:
            return None
        b, a, m, dl = q
        fill = m + np.sign(qty) * capture * (a - b) / 2.0          # buy->ask, sell->bid
        net -= qty * fill * MULT                                    # pay to open (qty>0 buy: cash out)
        net -= FEE * abs(qty)
        entry[(K, cpn)] = fill
        nd += qty * dl
    hpnl = 0.0; last_hedge = 0
    for i, s in enumerate(order[1:], 1):
        ch = panel_d[panel_d.snap == s]
        if not len(ch):
            continue
        F = float(ch.F.iloc[0])
        hpnl += (-nd) * (F - prevF) * MULT; prevF = F
        if i - last_hedge >= hedge_every:
            d_now = 0.0; ok = True
            for qty, K, cpn in picks:
                q = leg_quote(ch, K, cpn)
                if q is None:
                    ok = False; break
                d_now += qty * q[3]
            if ok:
                hpnl -= abs(nd - d_now) * F * MULT * SLIP
                nd = d_now; last_hedge = i
    # SETTLE at intrinsic at expiry: a 0DTE option held to settlement exercises / cash-settles to intrinsic
    # with NO closing bid/ask spread and NO closing fee. (Buying every leg back at the stale ~15:00 ask was the
    # load-bearing error that inverted the result — 0DTE income trades pay the entry spread, not a round trip.)
    # Settlement price = F at the last genuine snap (~15:00). The ITM leg is paid/received at intrinsic.
    Fset = prevF
    for qty, K, cpn in picks:
        net += qty * max(cpn * (Fset - K), 0.0) * MULT
    return net + hpnl


STRUCTS = {
    "short_straddle": [(1, -1, None), (-1, -1, None)],
    "short_strangle": [(1, -1, 0.25), (-1, -1, 0.25)],
    "iron_condor": [(1, -1, 0.25), (1, 1, 0.10), (-1, -1, 0.25), (-1, 1, 0.10)],
    "long_straddle": [(1, 1, None), (-1, 1, None)],
}


def main():
    import glob as _g
    t0 = time.time()
    args = sys.argv[1:] if len(sys.argv) > 1 else [f"{ODTE_RAW}/*.parquet"]
    files = []
    for a in args:
        files += sorted(_g.glob(a)) if "*" in a else [a]
    panels = []
    for f in files:
        try:
            P = panel_0dte(finalize(pd.read_parquet(f)))
        except Exception:
            P = None
        if P is not None and len(P):
            P = P.copy(); P["root"] = os.path.basename(f)[:-8]
            panels.append(P)
    if not panels:
        print("no 0DTE panel"); return
    P = pd.concat(panels, ignore_index=True)
    P["rd"] = P["root"] + "_" + P["date"].astype(str)         # unique (root,date) key
    days = sorted(P["rd"].unique())
    print(f"0DTE panel: {len(days)} expiry-days across {P.root.nunique()} roots, F range {P.F.min():.0f}-{P.F.max():.0f} [{time.time()-t0:.0f}s]", flush=True)
    recs = []
    for sname, legs in STRUCTS.items():
        for entry in ["10:00", "13:00"]:
            for hedge in [1, 4, 99]:                # hedge every step / every 4 steps / never
                pnl = []
                for d in days:
                    pd_d = P[P.rd == d]
                    r = run_day(pd_d, legs, entry, hedge)
                    if r is not None and np.isfinite(r):
                        pnl.append(r)
                if len(pnl) < 20:
                    continue
                x = np.array(pnl); mu = x.mean(); sd = x.std(ddof=1)
                if sd <= 0:
                    continue
                recs.append(dict(struct=sname, entry=entry, hedge=hedge, n=len(x), net=round(x.sum()),
                                 mean=round(mu, 1), t=round(mu / (sd / np.sqrt(len(x))), 2),
                                 sharpe_ann=round((mu / sd) * np.sqrt(52), 2), win=round((x > 0).mean(), 2),
                                 sr_trade=mu / sd, skew=round(float(ss.skew(x)), 2),
                                 kurt=round(float(ss.kurtosis(x, fisher=False)), 2), worst=round(x.min())))
    R = pd.DataFrame(recs)
    if len(R):
        trial = R["sr_trade"].to_numpy()
        R["dsr"] = [dsr_pvalue(r.sr_trade, r.skew, r.kurt, r.n, trial)[0] for r in R.itertuples()]
        R["p"] = 2 * ss.t.sf(R["t"].abs(), R["n"] - 1)
        m = len(R); order = np.argsort(R["p"].to_numpy()); ps = R["p"].to_numpy()[order]
        bh = ps <= (np.arange(1, m + 1) / m) * 0.10
        kmax = (np.where(bh)[0].max() + 1) if bh.any() else 0
        fp = np.zeros(m, bool)
        if kmax:
            fp[order[:kmax]] = True
        R["fdr_pass"] = fp; R["dsr_pass"] = R["dsr"] > 0.95
        R = R.sort_values("t", ascending=False)
    R.to_csv(os.path.join(OUT, "odte_strategies.csv"), index=False)
    print(R[["struct", "entry", "hedge", "n", "net", "t", "sharpe_ann", "win", "skew", "dsr"]].to_string(index=False), flush=True)
    if len(R):
        print(f"\nnet-positive: {int((R.net>0).sum())}/{len(R)} | FDR&DSR survivors: {int((R.fdr_pass&R.dsr_pass).sum())}", flush=True)
    print(f"DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
