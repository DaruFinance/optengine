"""optengine.signals — causal daily conditioning features from the prebuilt surface
panel (ATM-IV term structure, 25d risk-reversal/skew, realized vol, GEX), per
(date, underlying). All features are SHIFTED one day so an entry on day t uses only
information known at t-1 close (strictly no-lookahead). Used to CONDITION entries,
e.g. sell vol only when implied > realized (positive VRP) and the term is in contango.
"""
from __future__ import annotations

from optengine.config import SURFACE_PANEL

import pandas as pd

PANEL = SURFACE_PANEL
_P = None


def _panel():
    global _P
    if _P is None:
        df = pd.read_parquet(PANEL)
        df["d"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d")
        _P = df
    return _P


def underlyings():
    return sorted(str(x) for x in _panel()["underlying"].dropna().unique())


def features(underlying):
    """Causal daily feature frame indexed by YYYYMMDD, or None. Columns:
    vrp30 (atm_iv_30 - realized_vol_30), term_slope, skew_rr (25d RR), gex, atm_iv, rv30."""
    p = _panel()
    s = p[p["underlying"].astype(str) == str(underlying)].sort_values("d")
    if len(s) < 30:
        return None
    out = pd.DataFrame(index=s["d"].to_numpy())
    out["vrp30"] = (s["atm_iv_30"].to_numpy() - s["realized_vol_30"].to_numpy())
    out["term_slope"] = s["term_slope"].to_numpy()
    out["skew_rr"] = s["rr_25d_30"].to_numpy()
    out["gex"] = s["gex"].to_numpy()
    out["atm_iv"] = s["atm_iv_30"].to_numpy()
    out["rv30"] = s["realized_vol_30"].to_numpy()
    return out.shift(1)   # decide day t using t-1 close features
