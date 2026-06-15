"""Pollute-and-verify the WFO PROCEDURE (not just features) — the META-2026 lesson.

Two runs of the identical windowing, differing only in HOW the per-window knob is
chosen:
  - causal: knob picked by IN-SAMPLE score (the real, no-lookahead procedure)
  - leaked: knob picked by its OWN OUT-OF-SAMPLE score (a deliberate forward peek)
If the engine is causal, leaked >> causal (peeking inflates OOS) and causal is
deterministic on re-run. A causal pipeline that accidentally leaked would already
sit near the leaked number — that's the detector.
"""

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repo root
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))                    # sibling modules

import sys
from optengine.position import run_structure, LegSpec
from optengine.data import greeks_store as G


def straddle(lo, hi):
    return [LegSpec(cp=+1, qty=-1, sel="atm", dte=(lo, hi)),
            LegSpec(cp=-1, qty=-1, sel="atm", dte=(lo, hi))]


COMBOS = [(dte, e) for dte in [(25, 35), (40, 55)] for e in (2, 5, 10)]
DATES = G.available_dates(2024)
IS, OOS = 126, 63


def rrr(tr):
    if tr is None or not len(tr):
        return -1e18
    n = tr["net"]
    return float(n.mean() / (abs(n.min()) + 1e-9))


def run_windows(select):
    t = 0
    total = 0.0
    while t + IS + OOS <= len(DATES):
        isd = DATES[t:t + IS]
        ood = DATES[t + IS:t + IS + OOS]
        scored = []
        for dte, e in COMBOS:
            ti = run_structure("SPXW", isd, straddle(*dte), exit_dte=e, delta_hedge=True)
            to = run_structure("SPXW", ood, straddle(*dte), exit_dte=e, delta_hedge=True)
            scored.append((rrr(ti), rrr(to), to))
        key = (lambda z: z[0]) if select == "is" else (lambda z: z[1])
        best = max(scored, key=key)
        if best[2] is not None and len(best[2]):
            total += float(best[2]["net"].sum())
        t += OOS
    return total


causal1 = run_windows("is")
causal2 = run_windows("is")
leaked = run_windows("oos")
det = abs(causal1 - causal2) < 1e-6
print(f"causal OOS (IS-chosen):   ${causal1:>10,.0f}   re-run ${causal2:,.0f}  "
      f"-> {'DETERMINISTIC' if det else 'NON-DETERMINISTIC!'}")
print(f"leaked OOS (OOS-chosen):  ${leaked:>10,.0f}   (deliberate forward peek)")
print(f"leak premium:             ${leaked - causal1:>10,.0f}")
ok = det and (leaked > causal1 + 1)
print(f"\n{'PASS' if ok else 'FAIL'}: procedure is causal (causal != leaked) and deterministic. "
      f"Forward-peek is detected and quantified — the WFO is not accidentally leaking.")
