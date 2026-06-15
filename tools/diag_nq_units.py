"""Diagnose NQ ATM-IV-by-year unit break (NQ failed the [10,50]% sanity gate at 112% after 2020 data
was added). Replicate fop_finalize's parity-forward + Black-76 IV, but bucket by YEAR, to see whether
the break is 2020-vintage-only (fixable by a year-scoped scale) or all-years (a deeper decode issue).
Read-only on fop_raw/NQ. Prints median ATM IV and median (mid/forward) per year for NQ vs ES (control).
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules

import os, sys, glob
import numpy as np, pandas as pd
import datetime as dt
from fop_finalize import KSCALE, PSCALE, forward_from_parity, implied_vol, to_date, RAW


def atm_iv_by_year(root):
    files = sorted(glob.glob(os.path.join(RAW, root, "*.parquet")))
    if not files:
        print(f"{root}: no raw"); return
    print(f"\n{root}: {len(files)} raw date-files")
    print(f"  {'year':4s} {'dates':>5s} {'ATM_IV%':>8s} {'mid/F_atm':>9s} {'n_atm':>7s}")
    by_year = {}
    for f in files:
        yr = os.path.basename(f)[:4]
        by_year.setdefault(yr, []).append(f)
    for yr in sorted(by_year):
        fs = by_year[yr]
        r = pd.concat([pd.read_parquet(x) for x in fs], ignore_index=True)
        r["K"] = r["K"] / KSCALE.get(root, 1.0)
        ps = PSCALE.get(root, 1.0)
        r["mid"] = 0.5 * (r["bid"] + r["ask"]) * ps
        ivs, ratios = [], []
        # sample up to ~12 (date,contract,expiry) groups per year for speed
        grps = list(r.groupby(["asof_i", "mcode", "ey"]))
        step = max(1, len(grps) // 60)
        for (di, mc, ey), g in grps[::step]:
            ca = g[g.cp == "C"].set_index("K")["mid"]; pu = g[g.cp == "P"].set_index("K")["mid"]
            common = ca.index.intersection(pu.index)
            if len(common) < 4:
                continue
            exp_i = int(g["asof_i"].max())  # crude: not used for T here
            # T from a forward-week proxy: use max asof in this contract as a stand-in expiry
            T = max((g["asof_i"].max() - di), 1) / 365.0
            F, D = forward_from_parity(common.values.astype(float), ca.loc[common].values,
                                       pu.loc[common].values, float(np.median(common)))
            if not (np.isfinite(F) and F > 0):
                continue
            atm = g[(g.K / F).between(0.97, 1.03)]
            if not len(atm):
                continue
            cpn = np.where(atm.cp.values == "C", 1, -1).astype(int)
            iv = implied_vol(atm.mid.values.astype(float), float(F), atm.K.values.astype(float),
                             T, 0.0, 0.0, cpn, model="black76")
            iv = iv[np.isfinite(iv)]
            if len(iv):
                ivs.extend(iv.tolist())
                ratios.extend((atm.mid.values / F).tolist())
        miv = np.median(ivs) * 100 if ivs else np.nan
        mr = np.median(ratios) if ratios else np.nan
        by_year[yr] = (len(fs), miv, mr, len(ivs))
        print(f"  {yr:4s} {len(fs):5d} {miv:8.1f} {mr:9.4f} {len(ivs):7d}")


if __name__ == "__main__":
    for root in (sys.argv[1:] or ["NQ", "ES"]):
        atm_iv_by_year(root)
