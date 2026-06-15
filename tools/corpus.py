"""Rigorous corpus run: every archetype x instrument, full-history walk-forward,
delta-hedged, real bid/ask costs, no-lookahead. Per-strategy OOS stats with
return-on-premium normalization, then cross-sectional multiple-testing correction
(Benjamini-Hochberg FDR + Bailey-Lopez de Prado Deflated Sharpe; effective trials =
number of strategies tested, NOT windows x combos). Full per-trade ledger written per
instrument. Task = one instrument (loads its per-ticker parquet once, clears RAM after).
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import REPO_ROOT

import sys, os, glob, time, argparse
import multiprocessing as mp
import numpy as np, pandas as pd
from scipy import stats as ss
from scipy.stats import norm
from optengine.wfo import wfo
from optengine import archetypes as A
from optengine.data import greeks_store as G

BASE = REPO_ROOT
OUT = os.path.join(BASE, "findings")
LEDG = os.path.join(OUT, "ledgers")
GRID = {"dte": [(25, 35), (40, 55)]}
EXITS = (2, 5, 10)
MIN_TRADES = 12


def instruments():
    return sorted(os.path.basename(p)[:-8] for p in glob.glob(os.path.join(BASE, "pertkr", "*.parquet")))


def _stats(inst, arch, oos, wins):
    if oos is None or not len(oos):
        return dict(instrument=inst, archetype=arch, n=0, oos_net=0.0)
    net = oos["net"].to_numpy(float)
    prem = np.abs(oos["premium"].to_numpy(float))
    prem = np.where(prem < 1e-6, np.nan, prem)
    ret = (net / prem)
    ret = ret[np.isfinite(ret)]
    n = int(len(ret))
    base = dict(instrument=inst, archetype=arch, n=n, oos_net=round(float(net.sum())),
                win=round(float((net > 0).mean()), 3) if len(net) else 0.0,
                worst=round(float(net.min())) if len(net) else 0.0,
                med_prem=round(float(np.nanmedian(prem))) if n else 0.0)
    if n < 2:
        return base
    mu = float(np.mean(ret)); sd = float(np.std(ret, ddof=1))
    yrs = 0.5
    if len(wins) > 1:
        d0 = pd.Timestamp(str(wins["oos_start"].iloc[0]))
        d1 = pd.Timestamp(str(wins["oos_start"].iloc[-1]))
        yrs = max((d1 - d0).days / 365.25, 0.25)
    tpy = n / yrs
    sr_trade = mu / sd if sd > 0 else 0.0
    base.update(mean_ret=round(mu, 4), sd_ret=round(sd, 4),
                t=round(mu / (sd / np.sqrt(n)), 2) if sd > 0 else 0.0,
                sharpe_ann=round(sr_trade * np.sqrt(tpy), 2),
                sr_trade=round(sr_trade, 4),
                skew=round(float(ss.skew(ret)), 3), kurt=round(float(ss.kurtosis(ret, fisher=False)), 2),
                rrr=round(mu / (abs(float(np.min(ret))) + 1e-9), 3),
                tpy=round(tpy, 1))
    return base


def _task(args):
    inst, dates = args
    G.clear_cache()
    frames = G.build_array_frames(inst)          # fast vectorized per-day array-frames
    rows, ledgers = [], []
    for arch in A.ALL:
        try:
            oos, wins, _nc = wfo(inst, dates, getattr(A, arch), GRID, exit_grid=EXITS,
                                 is_len=126, oos_len=63, obj="rrr", delta_hedge=True, frames=frames)
        except Exception as e:
            rows.append(dict(instrument=inst, archetype=arch, n=0, error=repr(e)[:80]))
            continue
        rows.append(_stats(inst, arch, oos, wins))
        if oos is not None and len(oos):
            o = oos.copy(); o["instrument"] = inst; o["archetype"] = arch
            ledgers.append(o)
    if ledgers:
        try:
            pd.concat(ledgers, ignore_index=True).to_parquet(
                os.path.join(LEDG, f"{inst}.parquet"), compression="zstd", index=False)
        except Exception:
            pass
    G.clear_cache()
    return rows


def dsr_pvalue(sr, skew, kurt, n, trial_srs):
    """Bailey-Lopez de Prado Deflated Sharpe: P(true SR>0) given M trials. sr & trial_srs
    are per-observation (per-trade) Sharpes."""
    M = len(trial_srs)
    v = float(np.var(trial_srs, ddof=1))
    if v <= 0 or M < 2 or n < 2:
        return np.nan, np.nan
    emc = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - emc) * norm.ppf(1 - 1.0 / M) + emc * norm.ppf(1 - 1.0 / (M * np.e)))
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4.0 * sr * sr, 1e-9))
    dsr = float(norm.cdf((sr - sr0) * np.sqrt(n - 1) / denom))
    return dsr, float(sr0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--limit", type=int, default=0, help="cap #instruments (0=all)")
    ap.add_argument("--ram_floor", type=float, default=8.0)
    a = ap.parse_args()
    os.makedirs(LEDG, exist_ok=True)
    insts = instruments()
    if a.limit:
        insts = insts[:a.limit]
    dates = sorted(d for y in range(2017, 2027) for d in G.available_dates(y))
    from optengine.fan import ram_available_gb
    av = ram_available_gb()
    print(f"corpus: {len(insts)} instruments x {len(A.ALL)} archetypes = {len(insts)*len(A.ALL)} strategies, "
          f"{len(dates)} days, workers={a.workers}, RAM avail {av:.1f}GB", flush=True)
    if av < a.ram_floor:
        raise SystemExit(f"RAM {av:.1f} < floor {a.ram_floor}; abort")
    sizes = {i: os.path.getsize(os.path.join(BASE, "pertkr", f"{i}.parquet")) for i in insts}
    # 3-phase RAM-safe split: only the true-big index/mega-cap chains (>100MB parquet ->
    # ~3.4GB frames) need the tight 3-worker cap; mid single-names (25-100MB -> <2GB frames)
    # run wide. Peak RAM: mid 10x~2GB=20GB, big 3x~3.4GB=10GB — both well under avail.
    small = [i for i in insts if sizes[i] <= 25_000_000]
    mid = [i for i in insts if 25_000_000 < sizes[i] <= 100_000_000]
    big = [i for i in insts if sizes[i] > 100_000_000]
    print(f"  {len(small)} small @ w={a.workers}  +  {len(mid)} mid @ w={min(10, a.workers)}  +  "
          f"{len(big)} big @ w={min(3, a.workers)}  (RAM-safe phasing)", flush=True)
    t0 = time.time()
    allrows = []

    def _runpool(subset, w):
        if not subset:
            return
        with mp.Pool(w, maxtasksperchild=1) as pool:
            for k, rows in enumerate(pool.imap_unordered(_task, [(i, dates) for i in subset])):
                allrows.extend(rows)
                if (k + 1) % 10 == 0 or k + 1 == len(subset):
                    print(f"  {len(allrows)//len(A.ALL)}/{len(insts)} instr [{time.time()-t0:.0f}s, "
                          f"RAM {ram_available_gb():.1f}GB]", flush=True)

    _runpool(small, a.workers)
    _runpool(mid, min(10, a.workers))
    _runpool(big, min(3, a.workers))
    df = pd.DataFrame(allrows)
    df.to_csv(os.path.join(OUT, "corpus_raw.csv"), index=False)

    # ---- multiple-testing correction across the cross-section ----
    s = df[(df["n"] >= MIN_TRADES) & df.get("sr_trade", pd.Series(dtype=float)).notna()].copy()
    s["p"] = 2 * ss.t.sf(s["t"].abs(), s["n"] - 1)
    order = np.argsort(s["p"].to_numpy())
    p_sorted = s["p"].to_numpy()[order]
    m = len(s)
    bh = p_sorted <= (np.arange(1, m + 1) / m) * 0.10        # BH-FDR q=0.10
    kmax = np.where(bh)[0].max() + 1 if bh.any() else 0
    fdr_pass = np.zeros(m, bool)
    if kmax:
        fdr_pass[order[:kmax]] = True
    s["fdr_pass"] = fdr_pass
    trial = s["sr_trade"].to_numpy()
    dvals = [dsr_pvalue(r.sr_trade, r.skew, r.kurt, r.n, trial) for r in s.itertuples()]
    s["dsr"] = [d[0] for d in dvals]
    s["sr0"] = [d[1] for d in dvals]
    s["dsr_pass"] = s["dsr"] > 0.95
    s = s.sort_values("t", ascending=False)
    s.to_csv(os.path.join(OUT, "corpus_strategies.csv"), index=False)
    print(f"\nDONE {time.time()-t0:.0f}s. {len(df)} strategies attempted, {m} with >={MIN_TRADES} trades.")
    print(f"  survive BH-FDR(q=.10): {int(s['fdr_pass'].sum())}   survive Deflated-Sharpe(>.95): {int(s['dsr_pass'].sum())}")
    print(f"  both: {int((s['fdr_pass'] & s['dsr_pass']).sum())}")
    print(s.head(20)[["instrument", "archetype", "n", "oos_net", "win", "t", "sharpe_ann", "dsr", "fdr_pass"]].to_string(index=False))


if __name__ == "__main__":
    main()
