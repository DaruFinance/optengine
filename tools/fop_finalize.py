"""Finalize raw FO snapshots -> engine-format daily chains (fop_pertkr/<root>.parquet).

Rule-free expiry: each contract's expiry = its LAST quote date in the data (exact for any
contract that expires within the pulled window; right-censored live contracts at the pull
frontier are dropped). Per (date, expiry): put-call-parity forward F (Black-76 measure),
T=(expiry-asof)/365, Black-76 implied vol + delta. Stores REAL points prices + real F; the
engine applies the per-root multiplier via run_structure(mult=ROOT_MULT[root]).

Prints a per-root ATM-IV SANITY check — a root only enters the study if its ATM IV lands in
the expected band (catches wrong price-units / multipliers, esp. the thin CME crypto roots).
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import FOP_PERTKR, FOP_RAW

import os, sys, argparse, datetime as dt
import numpy as np, pandas as pd
from optengine.pricing import forward_from_parity, implied_vol, black76_greeks

RAW = FOP_RAW
OUT = FOP_PERTKR
FORCE = False      # if True, write even roots that fail the ATM-IV sanity band (set via --force)

# CME contract multiplier ($/index-point) and a representative options fee ($/contract).
# 17 roots across 6 asset classes (CME contract multiplier $/point). Each root's units verified via
# parity-forward level + ATM-IV sanity (multiplier-independent).
ROOT_MULT = {"ES": 50, "NQ": 20, "RTM": 50,                       # equity index
             "OG": 100, "PAO": 100,                               # metals (gold, palladium)
             "LO": 1000, "ON": 10000,                             # energy (crude 1000bbl, natgas 10000 MMBtu)
             "OZN": 1000, "OZB": 1000, "OZF": 1000, "OZT": 2000, "UB1": 1000,  # rates
             "BTC": 5, "ETH": 50,                                 # crypto
             "OZC": 50, "OZS": 50, "OZW": 50}                     # ags (grains, 5000bu x $0.01/cent)
ROOT_FEE = {r: (2.50 if r in ("BTC", "ETH") else 1.50) for r in ROOT_MULT}
# expected ATM-IV band per asset class (sanity gate; wide on purpose)
IV_BAND = {"ES": (0.08, 0.45), "NQ": (0.10, 0.50), "RTM": (0.12, 0.55),
           "OG": (0.07, 0.40), "PAO": (0.10, 0.60),
           "LO": (0.20, 1.20), "ON": (0.25, 1.50),
           "OZN": (0.02, 0.20), "OZB": (0.03, 0.25), "OZF": (0.01, 0.15), "OZT": (0.004, 0.10), "UB1": (0.04, 0.25),
           "BTC": (0.30, 1.60), "ETH": (0.35, 1.80),
           "OZC": (0.10, 0.60), "OZS": (0.08, 0.50), "OZW": (0.12, 0.70)}
# per-root strike (KSCALE) & premium (PSCALE) rescales to a common unit, each verified via parity + ATM-IV:
#   crude strikes in cents (LO/100); rates strikes x10 + premium x100; natgas strikes x1000 + premium x0.1;
#   grains premium x100; CME BTC premium x100 (strike already USD). ES/NQ/RTM/OG/PAO/ETH already aligned.
KSCALE = {"LO": 100.0, "ON": 1000.0, "OZN": 10.0, "OZB": 10.0, "OZF": 10.0, "OZT": 10.0, "UB1": 10.0}
PSCALE = {"OZN": 100.0, "OZB": 100.0, "OZF": 100.0, "OZT": 100.0, "UB1": 100.0,
          "ON": 0.1, "BTC": 100.0, "OZC": 100.0, "OZS": 100.0, "OZW": 100.0}


def to_date(i):
    i = int(i); return dt.date(i // 10000, (i // 100) % 100, i % 100)


def finalize(root):
    import glob as _glob
    files = _glob.glob(os.path.join(RAW, root, "*.parquet"))      # per-date files from the flat pool
    if not files and os.path.exists(os.path.join(RAW, f"{root}.parquet")):
        files = [os.path.join(RAW, f"{root}.parquet")]            # legacy single-file fallback
    if not files:
        print(f"{root}: no raw"); return None
    r = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    r["K"] = r["K"] / KSCALE.get(root, 1.0)        # strike rescale
    ps = PSCALE.get(root, 1.0)
    r["bid"] = r["bid"] * ps; r["ask"] = r["ask"] * ps   # premium rescale (stored -> engine P&L uses it x mult)
    r["mid"] = 0.5 * (r["bid"] + r["ask"])
    gmax = int(r["asof_i"].max())
    frontier = int((to_date(gmax) - dt.timedelta(days=7)).strftime("%Y%m%d"))
    # empirical expiry per contract (mcode,ey) = last quote date; drop live frontier
    exp_map = r.groupby(["mcode", "ey"])["asof_i"].max().rename("exp_i").reset_index()
    r = r.merge(exp_map, on=["mcode", "ey"])
    r = r[r["exp_i"] < frontier]
    out = []
    for (di, mc, ey, exp_i), g in r.groupby(["asof_i", "mcode", "ey", "exp_i"]):
        T = (to_date(exp_i) - to_date(di)).days / 365.0
        if T <= 0:
            continue
        ca = g[g.cp == "C"].set_index("K")["mid"]; pu = g[g.cp == "P"].set_index("K")["mid"]
        common = ca.index.intersection(pu.index)
        if len(common) < 4:
            continue
        F, D = forward_from_parity(common.values.astype(float), ca.loc[common].values,
                                   pu.loc[common].values, float(np.median(common)))
        if not (np.isfinite(F) and F > 0):
            continue
        cpn = np.where(g.cp.values == "C", 1, -1).astype(int)
        K = g.K.values.astype(float)
        iv = implied_vol(g.mid.values.astype(float), float(F), K, T, 0.0, 0.0, cpn, model="black76")
        gk = black76_greeks(float(F), K, T, 0.0, np.nan_to_num(iv, nan=0.2), cpn)
        out.append(pd.DataFrame(dict(asof_i=int(di), cp=cpn, Strike=K, expiry_i=int(exp_i),
            DaysToMaturity=(to_date(exp_i) - to_date(di)).days, MidImpliedVol=iv, MidDelta=gk["delta"],
            LastBidPrice=g.bid.values, LastMidPrice=g.mid.values, LastAskPrice=g.ask.values,
            UnderLastMidPrice=float(F), converged=np.isfinite(iv))))
    if not out:
        print(f"{root}: nothing finalized"); return None
    d = pd.concat(out, ignore_index=True).sort_values(["asof_i", "expiry_i", "cp", "Strike"])
    # ---- per-root ATM-IV sanity gate (BEFORE writing): only validated-unit roots enter the study ----
    atm = d[(d.Strike / d.UnderLastMidPrice).between(0.97, 1.03) & np.isfinite(d.MidImpliedVol)]
    med_iv = float(atm.MidImpliedVol.median()) if len(atm) else np.nan
    lo, hi = IV_BAND.get(root, (0.0, 5.0))
    ok = (lo <= med_iv <= hi) or FORCE
    tag = "PASS" if (lo <= med_iv <= hi) else ("FORCED" if FORCE else "SKIP-UNITS")
    print(f"{root}: {len(d)} rows, {d.asof_i.nunique()} dates, {d.expiry_i.nunique()} expiries | "
          f"ATM IV {med_iv*100:.1f}% band [{lo*100:.0f},{hi*100:.0f}]% -> {tag} "
          f"| mult={ROOT_MULT.get(root,'?')} spread/mid={((d.LastAskPrice-d.LastBidPrice)/d.LastMidPrice.replace(0,np.nan)).median():.3f}",
          flush=True)
    if not ok:
        return None                       # raw kept in fop_raw; re-finalize after a per-root unit fix
    os.makedirs(OUT, exist_ok=True)
    d.to_parquet(os.path.join(OUT, f"{root}.parquet"), compression="zstd", index=False)
    return os.path.join(OUT, f"{root}.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--roots", default=""); ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    FORCE = a.force
    roots = a.roots.split(",") if a.roots else [os.path.basename(p)[:-8] for p in
              __import__("glob").glob(os.path.join(RAW, "*.parquet"))]
    for rt in roots:
        finalize(rt)
