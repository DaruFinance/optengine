"""Second multiple-testing control + tail-window isolation — robustness for the deflation in STUDY.md.

(1) ROMANO-WOLF stepdown (FWER, bootstrap of the max studentized statistic). BH-FDR controls the false
    discovery RATE; DSR deflates a single Sharpe for trials. Romano-Wolf controls the family-wise error
    rate while accounting for CROSS-STRATEGY CORRELATION (the index/sector overlap FDR ignores), by
    bootstrapping the joint distribution of the max t across strategies. If RW also returns 0, the null is
    not an artifact of the multiplicity method. Strategies aligned on a common calendar-month P&L grid
    (0 in no-position months — a real monthly-frequency series); months resampled JOINTLY across all
    strategies so the bootstrap preserves their dependence.

(2) TAIL-WINDOW isolation: short-vol P&L in Feb-2018 (Volmageddon / XIV) and Mar-2020 (COVID), to show the
    short-volatility left tail explicitly rather than only through skew/kurtosis.

Pure post-process on the mega-run OOS ledgers — no backtest re-run.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FINDINGS, LEDGERS

import os, sys, glob, time
import numpy as np, pandas as pd
from corpus import MIN_TRADES

LEDG = LEDGERS
OUT = FINDINGS
SHORT_VEGA = {"short_straddle", "short_strangle", "iron_condor", "iron_butterfly", "put_write",
              "call_overwrite", "put_credit_spread", "call_credit_spread", "jade_lizard",
              "reverse_jade_lizard", "broken_wing_call", "broken_wing_put", "ratio_call_spread",
              "ratio_put_spread", "short_logstrip", "surface_short_rich_strangle", "surface_sell_rich_call",
              "surface_sell_rich_put"}
B = 1000          # bootstrap reps
ALPHA = 0.10      # FWER


def load_monthly():
    """Stream ledgers -> strategy x month net-P&L matrix (0 = no position that month)."""
    files = sorted(glob.glob(os.path.join(LEDG, "*.parquet")))
    cells = {}        # (inst,arch) -> {month -> net sum}, plus trade count
    cnt = {}
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["entry", "net", "archetype"])
        except Exception:
            continue
        inst = os.path.basename(f)[:-8]
        df["m"] = df["entry"].astype(str).str.slice(0, 6)
        for (arch, mon), g in df.groupby(["archetype", "m"], sort=False):
            key = (inst, arch)
            cells.setdefault(key, {})[mon] = cells.get(key, {}).get(mon, 0.0) + float(g["net"].sum())
            cnt[key] = cnt.get(key, 0) + len(g)
    months = sorted({m for d in cells.values() for m in d})
    midx = {m: i for i, m in enumerate(months)}
    keys = [k for k in cells if cnt[k] >= MIN_TRADES and len(cells[k]) >= 24]   # >=24 active months
    M = np.zeros((len(keys), len(months)), float)
    for r, k in enumerate(keys):
        for mon, v in cells[k].items():
            M[r, midx[mon]] = v
        # winsorize each strategy's monthly series at 2/98 pct (tame single-month blow-ups that
        # otherwise destabilise the studentized bootstrap variance)
        lo, hi = np.percentile(M[r], [2, 98])
        M[r] = np.clip(M[r], lo, hi)
    return keys, np.array(months), M


def romano_wolf(M, two_sided=True):
    S, T = M.shape
    mu = M.mean(1); sd = M.std(1, ddof=1); se = sd / np.sqrt(T)
    t_obs = np.where(se > 0, mu / se, 0.0)
    stat = np.abs(t_obs) if two_sided else t_obs
    R = M - mu[:, None]                                  # center under H0: E[monthly P&L]=0
    # bootstrapped studentized stats: t*[b,i] (joint month resample, seeded for reproducibility)
    rng = np.random.default_rng(12345)
    tstar = np.empty((B, S), float)
    for b in range(B):
        Rb = R[:, rng.integers(0, T, T)]                 # same month-resample applied to ALL strategies
        mb = Rb.mean(1); sb = Rb.std(1, ddof=1)
        tb = np.where(sb > 0, mb / (sb / np.sqrt(T)), 0.0)
        tstar[b] = np.abs(tb) if two_sided else tb
    # stepdown: reject stat_i >= (1-alpha) quantile of max over ACTIVE set; remove; repeat
    active = np.ones(S, bool); rejected = np.zeros(S, bool)
    while True:
        cols = np.where(active)[0]
        if not len(cols):
            break
        maxb = tstar[:, cols].max(1)
        c = np.quantile(maxb, 1 - ALPHA)
        newrej = active & (stat >= c)
        if not newrej.any():
            break
        rejected |= newrej; active &= ~newrej
    return rejected, t_obs, c


def stress_window(peak, label):
    """OOS trades HELD THROUGH the peak date (entry<=peak<=exit) — positions actually open when the vol
    bomb went off — short-vol vs long-vol net and worst single trade."""
    files = sorted(glob.glob(os.path.join(LEDG, "*.parquet")))
    sv = {"n": 0, "net": 0.0, "worst": 0.0}; lv = {"n": 0, "net": 0.0, "worst": 0.0}
    for f in files:
        try:
            df = pd.read_parquet(f, columns=["entry", "exit", "net", "archetype"])
        except Exception:
            continue
        e = df["entry"].astype(str); x = df["exit"].astype(str)
        held = (e <= str(peak)) & (x >= str(peak))        # position open ON the peak date
        d = df[held]
        for arch, g in d.groupby("archetype", sort=False):
            tgt = sv if arch in SHORT_VEGA else lv
            tgt["n"] += len(g); tgt["net"] += float(g["net"].sum())
            tgt["worst"] = min(tgt["worst"], float(g["net"].min()) if len(g) else 0.0)
    return label, sv, lv


def main():
    t0 = time.time()
    print("loading monthly P&L matrix...", flush=True)
    keys, months, M = load_monthly()
    print(f"  {len(keys)} strategies x {len(months)} months [{time.time()-t0:.0f}s]", flush=True)
    rej, t_obs, c = romano_wolf(M, two_sided=False)      # one-sided: hunt POSITIVE surviving edges
    print(f"\nROMANO-WOLF stepdown (FWER={ALPHA}, B={B}, one-sided +edge, joint month resample):", flush=True)
    print(f"  survivors (FWER-significant POSITIVE edge): {int(rej.sum())} / {len(keys)}", flush=True)
    print(f"  max +t_obs: {t_obs.max():.2f}  | critical value c: {c:.2f}  "
          f"(best monthly edge {'clears' if t_obs.max()>=c else 'BELOW'} the FWER bar)", flush=True)
    top = np.argsort(-t_obs)[:8]
    print("  top +t strategies:", [(keys[i][0], keys[i][1], round(float(t_obs[i]), 2)) for i in top], flush=True)
    pd.DataFrame({"instrument": [keys[i][0] for i in range(len(keys))],
                  "archetype": [keys[i][1] for i in range(len(keys))],
                  "t_monthly": t_obs, "rw_reject": rej}).to_csv(os.path.join(OUT, "romano_wolf.csv"), index=False)
    print("\nTAIL-WINDOW isolation (short-vol vs long-vol, OOS trades HELD THROUGH the peak):", flush=True)
    for peak, lab in [(20180205, "Volmageddon peak 2018-02-05"), (20200316, "COVID peak 2020-03-16")]:
        _, sv, lv = stress_window(peak, lab)
        print(f"  {lab}:", flush=True)
        print(f"    SHORT-vol: {sv['n']} trades, net ${sv['net']:,.0f}, worst single ${sv['worst']:,.0f}", flush=True)
        print(f"    LONG-vol : {lv['n']} trades, net ${lv['net']:,.0f}, worst single ${lv['worst']:,.0f}", flush=True)
    print(f"\nDONE {time.time()-t0:.0f}s -> romano_wolf.csv", flush=True)


if __name__ == "__main__":
    main()
