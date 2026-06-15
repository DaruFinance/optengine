"""optengine.position — multi-leg option structures + path P&L on real chains.

A Structure is a list of LegSpec (declarative strike/expiry selectors). run_structure
steps a ticker's daily chains, opens the structure on an entry trigger, marks it to the
real quotes, daily-delta-hedges, exits on a DTE/expiry rule, and emits per-trade GROSS
(mid) and NET (real bid/ask + commission) P&L.

HOT-LOOP: per-day selection / matching / marking run on numpy ARRAY-FRAMES (dict of
np arrays from to_arrays) — no pandas in the inner loop (≈20× faster than DataFrame ops).
pandas is used only to build frames and for rare per-trade calendar-day math.

Sign convention: leg qty is signed (+long, -short). To OPEN, +qty buys (toward ask),
-qty sells (toward bid); to CLOSE, the reverse. Leg P&L = qty·(close_fill-open_fill)·100.
"""
from __future__ import annotations
from dataclasses import dataclass
import datetime as _dt
import numpy as np, pandas as pd
from .data import greeks_store as _gs
from .costs import OptionCostModel, MULT

__all__ = ["LegSpec", "to_arrays", "run_structure"]


@dataclass
class LegSpec:
    cp: int                 # +1 call, -1 put
    qty: int                # signed: + long, - short
    sel: str = "atm"        # 'atm' | 'delta' | 'moneyness'
    sel_val: float = None
    dte: tuple = (25, 35)   # (lo, hi) days-to-expiry window


def _converged(ch):
    base = (ch["converged"] if "converged" in ch.columns
            else ch["ImpliedVolConvergence"].astype(str).str.startswith("Converged"))
    return ch[base & (ch["MidImpliedVol"] > 0) & (ch["LastMidPrice"] > 0)
              & (ch["LastBidPrice"] >= 0) & (ch["LastAskPrice"] > 0)]


def to_arrays(ch):
    """Converged chain DataFrame -> numpy array-frame (the hot-loop format), or None."""
    if ch is None or not len(ch):
        return None
    exp = (ch["expiry_i"].to_numpy(np.int32) if "expiry_i" in ch.columns
           else ch["expiry"].dt.strftime("%Y%m%d").astype("int32").to_numpy())
    return {
        "cp": ch["cp"].to_numpy(np.int8),
        "K": ch["Strike"].to_numpy(np.float32),
        "dte": ch["DaysToMaturity"].to_numpy(np.float32),     # vendor = calendar days to expiry
        "delta": ch["MidDelta"].to_numpy(np.float32),
        "bid": ch["LastBidPrice"].to_numpy(np.float32),
        "mid": ch["LastMidPrice"].to_numpy(np.float32),
        "ask": ch["LastAskPrice"].to_numpy(np.float32),
        "exp": exp,
        "under": float(np.median(ch["UnderLastMidPrice"].to_numpy(np.float64))),
    }


def _ddays(a, b):
    """Calendar days b-a for yyyymmdd ints (datetime.date.toordinal is C-fast)."""
    a, b = int(a), int(b)
    return (_dt.date(b // 10000, (b // 100) % 100, b % 100).toordinal()
            - _dt.date(a // 10000, (a // 100) % 100, a % 100).toordinal())


def _select(fr, spec):
    """Index of the contract matching LegSpec in array-frame fr, or -1."""
    lo, hi = spec.dte
    m = (fr["cp"] == spec.cp) & (fr["dte"] >= lo) & (fr["dte"] <= hi)
    idx = np.where(m)[0]
    if idx.size == 0:
        return -1
    mid_dte = 0.5 * (lo + hi)
    texp = fr["exp"][idx[np.argmin(np.abs(fr["dte"][idx] - mid_dte))]]
    idx2 = idx[fr["exp"][idx] == texp]
    if idx2.size == 0:
        return -1
    if spec.sel == "delta":
        j = np.argmin(np.abs(np.abs(fr["delta"][idx2]) - spec.sel_val))
    elif spec.sel == "moneyness":
        j = np.argmin(np.abs(fr["K"][idx2] / fr["under"] - spec.sel_val))
    elif spec.sel in ("rich", "cheap"):
        # surface-RV: pick the contract richest/cheapest vs the fitted SVI surface, within a
        # tradeable |delta| band (avoid illiquid deep wings). Needs fr["rich"] (precomputed).
        if "rich" not in fr:
            return -1
        band = (np.abs(fr["delta"][idx2]) >= 0.08) & (np.abs(fr["delta"][idx2]) <= 0.55)
        cand = idx2[band] if band.any() else idx2
        r = fr["rich"][cand]
        ok = np.isfinite(r)
        if not ok.any():
            return -1
        cand = cand[ok]; r = r[ok]
        j = int(np.argmax(r) if spec.sel == "rich" else np.argmin(r))
        return int(cand[j])
    else:
        j = np.argmin(np.abs(fr["K"][idx2] - fr["under"]))
    return int(idx2[j])


def _match(fr, cp, K, exp):
    w = np.where((fr["cp"] == cp) & (fr["K"] == K) & (fr["exp"] == exp))[0]
    return int(w[0]) if w.size else -1


def run_structure(ticker, dates, structure, entry_gap=7, exit_dte=2,
                  cost=None, capture=1.0, delta_hedge=False, hedge_slip=5e-5, frames=None,
                  entry_days=None, mult=MULT, commission=0.65, hedge_every=1):
    """Backtest one structure over `dates` (sorted YYYYMMDD strings). `frames`: optional
    {day: array-frame}. `mult` = contract multiplier ($/point); default 100 (OPRA), pass the
    per-root futures multiplier (ES=$50, GC=$100, …) for the futures-options venue.
    {day: array-frame} (from to_arrays); else chains load+converge+arrayify per day."""
    cost = cost or OptionCostModel(spread_capture=capture, commission=commission)
    pos = None
    last_entry_i = None
    trades = []
    for d in dates:
        if frames is not None:
            fr = frames.get(d)
        else:
            ch = _gs.load_greeks_day(d, ticker)
            fr = to_arrays(_converged(ch)) if ch is not None else None
        if fr is None or fr["cp"].size == 0:
            continue
        di = int(d)
        under = fr["under"]

        # ---- manage open position ----
        if pos is not None:
            idxs = [_match(fr, cp, K, exp) for (cp, K, exp, _m, _f) in pos["legs"]]
            if delta_hedge:
                # existing hedge marks daily; the hedge is RE-SET (and slip paid) only every
                # `hedge_every` days. hedge_every=1 reproduces daily hedging exactly.
                pos["hedge_pnl"] += (-pos["net_delta"]) * (under - pos["prev_under"]) * mult
                pos["prev_under"] = under
                if _ddays(pos["last_hedge_i"], di) >= hedge_every and all(i >= 0 for i in idxs):
                    nd = sum(spec.qty * float(fr["delta"][i]) for spec, i in zip(structure, idxs))
                    pos["hedge_slip_cost"] += abs(pos["net_delta"] - nd) * under * mult * hedge_slip
                    pos["net_delta"] = nd
                    pos["last_hedge_i"] = di
            dte_now = _ddays(di, pos["exp_i"])
            if dte_now <= exit_dte or any(i < 0 for i in idxs):
                g = n = 0.0
                ncon = 0
                for spec, (cp, K, exp, mid_e, fill_e), i in zip(structure, pos["legs"], idxs):
                    if i >= 0:
                        mid = float(fr["mid"][i]); bid = float(fr["bid"][i]); ask = float(fr["ask"][i])
                    else:
                        mid = bid = ask = max(cp * (under - K), 0.0)
                    xfill = cost.fill(mid, bid, ask, -np.sign(spec.qty))
                    g += spec.qty * (mid - mid_e) * mult
                    n += spec.qty * (xfill - fill_e) * mult - cost.commission_cost(spec.qty)
                    ncon += abs(spec.qty)
                trades.append(dict(entry=pos["entry"], exit=d, expiry=str(pos["exp_i"]),
                                   dte_held=pos["dte_held"], under_entry=pos["under"], under_exit=under,
                                   premium=pos["premium"], opt_gross=g, opt_net=n,
                                   hedge_pnl=pos["hedge_pnl"], gross=g + pos["hedge_pnl"],
                                   net=n + pos["hedge_pnl"] - pos["hedge_slip_cost"], ncon=ncon))
                pos = None

        # ---- open new position ----
        if (pos is None and (entry_days is None or d in entry_days)
                and (last_entry_i is None or _ddays(last_entry_i, di) >= entry_gap)):
            picks = [_select(fr, spec) for spec in structure]
            if all(i >= 0 for i in picks):
                legs = []
                prem = 0.0
                nd = 0.0
                for spec, i in zip(structure, picks):
                    cp = int(fr["cp"][i]); K = float(fr["K"][i]); exp = int(fr["exp"][i])
                    mid = float(fr["mid"][i]); bid = float(fr["bid"][i]); ask = float(fr["ask"][i])
                    fill = cost.fill(mid, bid, ask, float(np.sign(spec.qty)))
                    legs.append((cp, K, exp, mid, fill))
                    prem += abs(spec.qty * fill) * mult
                    nd += spec.qty * float(fr["delta"][i])
                exp_i = min(l[2] for l in legs)
                pos = dict(entry=d, under=under, legs=legs, premium=prem, exp_i=exp_i,
                           dte_held=_ddays(di, exp_i), net_delta=nd, prev_under=under,
                           hedge_pnl=0.0, hedge_slip_cost=0.0, last_hedge_i=di)
                last_entry_i = di
    return pd.DataFrame(trades)
