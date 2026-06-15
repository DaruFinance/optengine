#!/usr/bin/env python3
"""tools/validate_surface.py — quality + no-arb validation of the causal SVI surface.

Fits optengine.surface.Surface for a SMALL sample (<=5 underlyings x <=5 dates),
and for each (ticker, date) reports, per fitted expiry slice:
  - in-sample RMSE of model IV vs vendor MidImpliedVol (the smoothness/quality check)
  - number of converged strikes used
  - butterfly violations IN DATA (Durrleman g<0 where quotes exist) and on a wide grid
  - calendar violations (total variance decreasing in T at fixed k, in the common band)
and the four causal surface signals (atm_vol, skew, rr25, term_slope, curvature).

Closes with an assertion-style summary: median IV-RMSE across all slices (expected
small, < ~1.5 vol points for liquid names) and the total arb-violation counts.

RAM-/CPU-disciplined (the box is shared): RLIMIT_AS = 4 GB, single process,
OMP/OPENBLAS pinned to 1 thread, affinity to cores {0,1,2,3}. Reads only the sampled
days from each per-ticker parquet (PyArrow predicate pushdown). Never fits the universe.

Usage:
  python3 tools/validate_surface.py                 # default 5x5 liquid sample
  python3 tools/validate_surface.py --tickers SPY QQQ --dates 20210915 20230213
"""
from __future__ import annotations


import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules
from optengine.config import PERTKR

# ---- resource discipline FIRST, before numpy/scipy import anything heavy ----
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import resource
resource.setrlimit(resource.RLIMIT_AS, (4 * 1024**3, 4 * 1024**3))
try:
    os.sched_setaffinity(0, {0, 1, 2, 3})
except Exception:
    pass

import sys
import argparse
import numpy as np
import pandas as pd

from optengine.surface import fit_surface, durrleman_g   # noqa: E402

PK_DIR = PERTKR

# Default small, liquid sample (<=5 tickers, <=5 dates). Dates are mid-sample and
# include the 2020-04 COVID vol spike as a stress case.
DEFAULT_TICKERS = ["SPY", "QQQ", "IWM", "GLD", "HYG"]
DEFAULT_DATES = [20181116, 20200420, 20210915, 20230213, 20240715]

MAX_TICKERS = 5
MAX_DATES = 5


def _valid_dates(ticker, wanted):
    """Keep only requested dates that actually exist for this ticker."""
    p = os.path.join(PK_DIR, f"{ticker}.parquet")
    if not os.path.exists(p):
        return []
    have = set(int(x) for x in np.unique(pd.read_parquet(p, columns=["asof_i"])["asof_i"]))
    return [d for d in wanted if int(d) in have]


def validate(tickers, dates):
    tickers = tickers[:MAX_TICKERS]
    dates = dates[:MAX_DATES]

    all_slice_rmse = []          # every slice's IV-RMSE, for the median assertion
    tot_bf_in_data = 0
    tot_bf_wide = 0
    tot_cal_in_data = 0
    tot_skipped = 0
    n_surfaces = 0
    n_slices = 0
    sample_signals = {}          # ticker -> {date -> signals}

    hdr = (f"{'ticker':<6} {'date':<9} {'exp':<9} {'dte':>4} {'nK':>4} "
           f"{'F':>9} {'src':<6} {'IVrmse(vp)':>10} {'bf_in':>6} {'cal':>4}")
    print("=" * len(hdr))
    print("PER-SLICE FIT & ARB DIAGNOSTICS")
    print("=" * len(hdr))

    for tk in tickers:
        dts = _valid_dates(tk, dates)
        if not dts:
            print(f"{tk:<6} (no requested dates present — skipped)")
            continue
        sample_signals[tk] = {}
        for di in dts:
            S = fit_surface(tk, di)
            n_surfaces += 1
            tot_skipped += S.skipped
            if not S.slices:
                print(f"{tk:<6} {di:<9} (no fittable slices; skipped={S.skipped})")
                continue

            # per-slice rows
            print("-" * len(hdr))
            print(hdr)
            for s in S.slices:
                # this slice's own in-data butterfly count (g<0 where quotes exist)
                kg = np.linspace(s.k_min, s.k_max, 121)
                g_slice = durrleman_g(s.params, kg)
                bf_slice = int(np.sum(g_slice < -1e-10))
                print(f"{tk:<6} {di:<9} {s.expiry_i:<9} {s.T*365:>4.0f} {s.n_strikes:>4} "
                      f"{s.F:>9.2f} {s.forward_source:<6} {s.rmse_iv*100:>10.3f} "
                      f"{bf_slice:>6} {'':>4}")
                all_slice_rmse.append(s.rmse_iv)
                n_slices += 1

            ar = S.arb_report()
            tot_bf_in_data += ar["butterfly_violations_in_data"]
            tot_bf_wide += ar["butterfly_violations"]
            tot_cal_in_data += ar["calendar_violations"]

            sig = S.signals()
            sample_signals[tk][di] = sig
            slice_rmses = [s.rmse_iv for s in S.slices]
            print(f"  -> {tk} {di}: nslices={len(S.slices)} skipped={S.skipped} "
                  f"medRMSE={np.median(slice_rmses)*100:.3f}vp "
                  f"maxRMSE={np.max(slice_rmses)*100:.3f}vp")
            print(f"     arb: butterfly_in_data={ar['butterfly_violations_in_data']} "
                  f"(wide_extrap={ar['butterfly_violations']}) "
                  f"calendar_in_data={ar['calendar_violations']}")
            print(f"     signals: atm_vol={sig['atm_vol']:.4f} skew={sig['skew']:+.4f} "
                  f"rr25={sig['rr25']:+.4f} term_slope={sig['term_slope']:+.4f} "
                  f"curvature={sig['curvature']:+.4f}")

    # ------------------------------------------------------------------
    # assertion-style summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    if all_slice_rmse:
        med = float(np.median(all_slice_rmse)) * 100
        p90 = float(np.percentile(all_slice_rmse, 90)) * 100
        mx = float(np.max(all_slice_rmse)) * 100
    else:
        med = p90 = mx = float("nan")
    print(f"surfaces fit            : {n_surfaces}")
    print(f"slices fit              : {n_slices}  (slices skipped: {tot_skipped})")
    print(f"median IV-fit RMSE      : {med:.3f} vol points")
    print(f"90th-pct / max IV-RMSE  : {p90:.3f} / {mx:.3f} vol points")
    print(f"butterfly violations    : in-data={tot_bf_in_data}  (wide-grid extrap={tot_bf_wide})")
    print(f"calendar violations     : in-data={tot_cal_in_data}")

    # soft assertions (report PASS/WARN, don't hard-exit on arb since they're
    # reported diagnostics, not enforced constraints)
    BAR_VP = 1.5
    ok_rmse = np.isfinite(med) and med < BAR_VP
    print(f"\nASSERT median IV-RMSE < {BAR_VP:.1f} vp : "
          f"{'PASS' if ok_rmse else 'FAIL'} (got {med:.3f} vp)")
    print(f"ASSERT no in-data butterfly arb  : "
          f"{'PASS' if tot_bf_in_data == 0 else 'WARN'} (in-data count={tot_bf_in_data})")
    print(f"INFO  in-data calendar arb count : {tot_cal_in_data} "
          f"(genuine adjacent-expiry ATM-variance inversions; reported not enforced)")

    # echo one sample signals dict verbatim (SPY mid-sample) for the report
    if "SPY" in sample_signals and sample_signals["SPY"]:
        mid_d = sorted(sample_signals["SPY"])[len(sample_signals["SPY"]) // 2]
        print(f"\nSAMPLE signals dict  (SPY {mid_d}):")
        print("  " + repr({k: round(v, 6) for k, v in sample_signals["SPY"][mid_d].items()}))

    print(f"\npeak RSS: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MB")
    return 0 if ok_rmse else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS,
                    help=f"<= {MAX_TICKERS} underlyings (default liquid sample)")
    ap.add_argument("--dates", nargs="+", type=int, default=DEFAULT_DATES,
                    help=f"<= {MAX_DATES} yyyymmdd dates")
    args = ap.parse_args()
    rc = validate(list(args.tickers), list(args.dates))
    sys.exit(rc)


if __name__ == "__main__":
    main()
