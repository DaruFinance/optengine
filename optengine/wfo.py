"""optengine.wfo — rolling walk-forward over a structure's IS-tunable knobs.

Each window: tune the knob grid on the IN-SAMPLE trading days (pick the combo
maximizing the objective), then evaluate ONLY that combo OUT-OF-SAMPLE. Aggregate
the OOS trades across all windows = the honest, un-tuned performance. The knobs are
IS-tunable (sampled per window); the structure shape is fixed. Effective-trials for
deflation = number of knob combos (NOT windows*combos).
"""
from __future__ import annotations
import itertools
import numpy as np, pandas as pd
from .position import run_structure

__all__ = ["wfo", "objective"]


def objective(tr, kind="expectancy"):
    if tr is None or not len(tr):
        return -np.inf
    n = tr["net"].values
    if kind == "total":
        return float(np.sum(n))
    if kind == "rrr":                     # tail-guarded reward / worst-loss
        return float(np.mean(n) / (abs(np.min(n)) + 1e-9))
    return float(np.mean(n))              # expectancy (default)


def wfo(ticker, dates, build_structure, knob_grid, exit_grid=(2,),
        is_len=126, oos_len=63, obj="expectancy", capture=1.0, delta_hedge=True,
        entry_gap=7, frames=None, entry_days=None, mult=100.0, commission=0.65, hedge_every=1):
    """build_structure(knobs)->list[LegSpec]; knob_grid: name->values (IS-tunable).
    Returns (oos_trades, window_summary, n_combos)."""
    dates = sorted(dates)
    N = len(dates)
    keys = list(knob_grid)
    combos = [dict(zip(keys, v)) for v in itertools.product(*[knob_grid[k] for k in keys])]
    combos = [dict(c, exit_dte=e) for c in combos for e in exit_grid]

    oos_all, rows = [], []
    t = 0
    win = 0
    while t + is_len + oos_len <= N:
        is_dates = dates[t:t + is_len]
        oos_dates = dates[t + is_len:t + is_len + oos_len]
        # ---- tune on IS ----
        best, best_s, best_is_tr = None, -np.inf, None
        for c in combos:
            st = build_structure(c)
            tr = run_structure(ticker, is_dates, st, entry_gap=entry_gap,
                               exit_dte=c["exit_dte"], capture=capture, delta_hedge=delta_hedge,
                               frames=frames, entry_days=entry_days, mult=mult, commission=commission,
                               hedge_every=hedge_every)
            s = objective(tr, obj)
            if s > best_s:
                best_s, best, best_is_tr = s, c, tr
        if best is None:                     # no tradeable data in this IS window -> skip it
            t += oos_len; win += 1
            continue
        # ---- evaluate best IS combo OOS (no peeking) ----
        st = build_structure(best)
        tr_oos = run_structure(ticker, oos_dates, st, entry_gap=entry_gap,
                               exit_dte=best["exit_dte"], capture=capture, delta_hedge=delta_hedge,
                               frames=frames, entry_days=entry_days, mult=mult, commission=commission,
                               hedge_every=hedge_every)
        oos_net = float(tr_oos["net"].sum()) if (tr_oos is not None and len(tr_oos)) else 0.0
        if tr_oos is not None and len(tr_oos):
            tr_oos = tr_oos.copy(); tr_oos["window"] = win
            oos_all.append(tr_oos)
        rows.append(dict(window=win, is_start=dates[t], oos_start=oos_dates[0],
                         best=str(best), is_score=round(best_s, 1),
                         is_net=float(best_is_tr["net"].sum()) if best_is_tr is not None and len(best_is_tr) else 0.0,
                         oos_net=oos_net))
        t += oos_len
        win += 1

    oos = pd.concat(oos_all, ignore_index=True) if oos_all else pd.DataFrame()
    return oos, pd.DataFrame(rows), len(combos)
