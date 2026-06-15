"""optengine.archetypes — structural option strategies as build_structure(knobs)->[LegSpec].

Module-level (picklable for multiprocessing). knobs carry the IS-tunable values the WFO
samples per window (dte window etc.); the structural SHAPE is fixed here. Strikes are
selected by delta or ATM. This is the directly-testable core of the 551-taxonomy; every
entry is run delta-hedged with real bid/ask costs under walk-forward.
"""
from .position import LegSpec


def _d(k):
    return k["dte"]


def _bk(k, off=30):                       # back-month window = front + off days (calendars)
    lo, hi = k["dte"]
    return (lo + off, hi + off)


# --- two-sided short premium (sell vol) ---
def short_straddle(k):
    return [LegSpec(+1, -1, "atm", dte=_d(k)), LegSpec(-1, -1, "atm", dte=_d(k))]


def short_strangle(k):
    d = k.get("delta", 0.16)
    return [LegSpec(+1, -1, "delta", d, dte=_d(k)), LegSpec(-1, -1, "delta", d, dte=_d(k))]


def wide_strangle(k):
    d = k.get("delta", 0.10)
    return [LegSpec(+1, -1, "delta", d, dte=_d(k)), LegSpec(-1, -1, "delta", d, dte=_d(k))]


# --- defined-risk short premium ---
def iron_condor(k):
    ds, dl = k.get("short_delta", 0.16), k.get("long_delta", 0.05)
    return [LegSpec(-1, -1, "delta", ds, dte=_d(k)), LegSpec(-1, +1, "delta", dl, dte=_d(k)),
            LegSpec(+1, -1, "delta", ds, dte=_d(k)), LegSpec(+1, +1, "delta", dl, dte=_d(k))]


def iron_butterfly(k):
    dl = k.get("long_delta", 0.10)
    return [LegSpec(+1, -1, "atm", dte=_d(k)), LegSpec(-1, -1, "atm", dte=_d(k)),
            LegSpec(-1, +1, "delta", dl, dte=_d(k)), LegSpec(+1, +1, "delta", dl, dte=_d(k))]


def put_credit_spread(k):
    ds, dl = k.get("short_delta", 0.30), k.get("long_delta", 0.15)
    return [LegSpec(-1, -1, "delta", ds, dte=_d(k)), LegSpec(-1, +1, "delta", dl, dte=_d(k))]


def call_credit_spread(k):
    ds, dl = k.get("short_delta", 0.30), k.get("long_delta", 0.15)
    return [LegSpec(+1, -1, "delta", ds, dte=_d(k)), LegSpec(+1, +1, "delta", dl, dte=_d(k))]


# --- one-sided premium (skew/single-leg vol) ---
def put_write(k):
    return [LegSpec(-1, -1, "delta", k.get("delta", 0.30), dte=_d(k))]


def call_overwrite(k):
    return [LegSpec(+1, -1, "delta", k.get("delta", 0.30), dte=_d(k))]


def short_put_atm(k):
    return [LegSpec(-1, -1, "atm", dte=_d(k))]


def short_call_atm(k):
    return [LegSpec(+1, -1, "atm", dte=_d(k))]


def put_ratio(k):                          # long 1 near, short 2 far (net short skew)
    dl, ds = k.get("long_delta", 0.30), k.get("short_delta", 0.15)
    return [LegSpec(-1, +1, "delta", dl, dte=_d(k)), LegSpec(-1, -2, "delta", ds, dte=_d(k))]


def risk_reversal(k):                      # short put / long call (skew + direction)
    d = k.get("delta", 0.25)
    return [LegSpec(-1, -1, "delta", d, dte=_d(k)), LegSpec(+1, +1, "delta", d, dte=_d(k))]


# --- term structure (calendars: short front / long back, same right) ---
def put_calendar(k):
    return [LegSpec(-1, -1, "atm", dte=_d(k)), LegSpec(-1, +1, "atm", dte=_bk(k))]


def call_calendar(k):
    return [LegSpec(+1, -1, "atm", dte=_d(k)), LegSpec(+1, +1, "atm", dte=_bk(k))]


# ============================================================================
# v2 expansion — structural SHAPES distinct from the 15 above (not delta re-spins).
# Adds the LONG-vol / convexity side the original corpus lacked, plus skew, ratio,
# diagonal, broken-wing, lizard, butterfly and directional-debit families, so the
# survival map covers BOTH signs of vega, not just premium selling. Deltas here are
# fixed structural choices; dte/exit remain the IS-tunable knobs (per the WFO grid).
# ============================================================================

# --- long premium / convexity (PAY the spread to buy vol — the other side) ---
def long_straddle(k):                       # buy ATM C+P — long gamma/vega
    return [LegSpec(+1, +1, "atm", dte=_d(k)), LegSpec(-1, +1, "atm", dte=_d(k))]


def long_strangle(k):
    d = k.get("delta", 0.16)
    return [LegSpec(+1, +1, "delta", d, dte=_d(k)), LegSpec(-1, +1, "delta", d, dte=_d(k))]


def long_put_tail(k):                       # buy OTM put — crash/tail hedge (Universa-style)
    return [LegSpec(-1, +1, "delta", k.get("delta", 0.10), dte=_d(k))]


def long_call(k):                           # buy OTM call — long upside convexity
    return [LegSpec(+1, +1, "delta", k.get("delta", 0.25), dte=_d(k))]


# --- backspreads (net long convexity via 1x2 ratio, opposite sign to ratios) ---
def put_backspread(k):                      # short 1 near, long 2 far OTM puts — long crash convexity
    dn, df = k.get("near_delta", 0.30), k.get("far_delta", 0.15)
    return [LegSpec(-1, -1, "delta", dn, dte=_d(k)), LegSpec(-1, +2, "delta", df, dte=_d(k))]


def call_backspread(k):                     # short 1 near, long 2 far OTM calls — long upside convexity
    dn, df = k.get("near_delta", 0.30), k.get("far_delta", 0.15)
    return [LegSpec(+1, -1, "delta", dn, dte=_d(k)), LegSpec(+1, +2, "delta", df, dte=_d(k))]


def call_ratio(k):                          # long 1 near, short 2 far calls — net short upside skew
    dl, ds = k.get("long_delta", 0.30), k.get("short_delta", 0.15)
    return [LegSpec(+1, +1, "delta", dl, dte=_d(k)), LegSpec(+1, -2, "delta", ds, dte=_d(k))]


# --- "no-upside-risk" premium (lizards) ---
def jade_lizard(k):                         # short put + short call-spread; credit covers call width
    dp = k.get("put_delta", 0.30)
    dcs, dcl = k.get("call_short_delta", 0.25), k.get("call_long_delta", 0.12)
    return [LegSpec(-1, -1, "delta", dp, dte=_d(k)),
            LegSpec(+1, -1, "delta", dcs, dte=_d(k)), LegSpec(+1, +1, "delta", dcl, dte=_d(k))]


def big_lizard(k):                          # short straddle + long OTM call (removes upside tail)
    return [LegSpec(+1, -1, "atm", dte=_d(k)), LegSpec(-1, -1, "atm", dte=_d(k)),
            LegSpec(+1, +1, "delta", k.get("long_delta", 0.15), dte=_d(k))]


# --- skewed defined-risk (broken wings) ---
def broken_wing_condor(k):                  # iron condor, put wing wider than call wing (skew tilt)
    ds = k.get("short_delta", 0.16)
    dpl, dcl = k.get("put_long_delta", 0.04), k.get("call_long_delta", 0.08)
    return [LegSpec(-1, -1, "delta", ds, dte=_d(k)), LegSpec(-1, +1, "delta", dpl, dte=_d(k)),
            LegSpec(+1, -1, "delta", ds, dte=_d(k)), LegSpec(+1, +1, "delta", dcl, dte=_d(k))]


def broken_wing_put_fly(k):                 # long high / short 2 mid / long far-OTM put (unbalanced)
    dh, dm, dl = k.get("high_delta", 0.40), k.get("mid_delta", 0.25), k.get("low_delta", 0.07)
    return [LegSpec(-1, +1, "delta", dh, dte=_d(k)), LegSpec(-1, -2, "delta", dm, dte=_d(k)),
            LegSpec(-1, +1, "delta", dl, dte=_d(k))]


# --- butterflies (long debit fly = short vol-of-vol / pin bet) ---
def call_butterfly(k):                      # long ITM, short 2 ATM, long OTM call
    w = k.get("wing", 0.05)
    return [LegSpec(+1, +1, "moneyness", 1 - w, dte=_d(k)), LegSpec(+1, -2, "atm", dte=_d(k)),
            LegSpec(+1, +1, "moneyness", 1 + w, dte=_d(k))]


def put_butterfly(k):
    w = k.get("wing", 0.05)
    return [LegSpec(-1, +1, "moneyness", 1 + w, dte=_d(k)), LegSpec(-1, -2, "atm", dte=_d(k)),
            LegSpec(-1, +1, "moneyness", 1 - w, dte=_d(k))]


# --- term structure + skew (diagonals; double calendar holds strike across months) ---
def double_calendar(k):                     # OTM put & call calendars (long vega term structure)
    wp, wc = 1 - k.get("wing", 0.05), 1 + k.get("wing", 0.05)
    return [LegSpec(-1, -1, "moneyness", wp, dte=_d(k)), LegSpec(-1, +1, "moneyness", wp, dte=_bk(k)),
            LegSpec(+1, -1, "moneyness", wc, dte=_d(k)), LegSpec(+1, +1, "moneyness", wc, dte=_bk(k))]


def put_diagonal(k):                        # short front OTM put, long back further-OTM put
    return [LegSpec(-1, -1, "moneyness", 1 - k.get("front_w", 0.03), dte=_d(k)),
            LegSpec(-1, +1, "moneyness", 1 - k.get("back_w", 0.07), dte=_bk(k))]


def call_diagonal(k):                       # short front OTM call, long back further-OTM call
    return [LegSpec(+1, -1, "moneyness", 1 + k.get("front_w", 0.03), dte=_d(k)),
            LegSpec(+1, +1, "moneyness", 1 + k.get("back_w", 0.07), dte=_bk(k))]


# --- directional defined-risk (debit spreads, collar, reverse skew) ---
def put_debit_spread(k):                    # long higher put, short lower put — bearish debit
    dl, ds = k.get("long_delta", 0.45), k.get("short_delta", 0.20)
    return [LegSpec(-1, +1, "delta", dl, dte=_d(k)), LegSpec(-1, -1, "delta", ds, dte=_d(k))]


def call_debit_spread(k):                   # long lower call, short higher call — bullish debit
    dl, ds = k.get("long_delta", 0.45), k.get("short_delta", 0.20)
    return [LegSpec(+1, +1, "delta", dl, dte=_d(k)), LegSpec(+1, -1, "delta", ds, dte=_d(k))]


def collar(k):                              # short OTM call + long OTM put (financed protection overlay)
    return [LegSpec(+1, -1, "delta", k.get("call_delta", 0.25), dte=_d(k)),
            LegSpec(-1, +1, "delta", k.get("put_delta", 0.20), dte=_d(k))]


def reverse_risk_reversal(k):               # long put / short call — bearish skew
    d = k.get("delta", 0.25)
    return [LegSpec(-1, +1, "delta", d, dte=_d(k)), LegSpec(+1, -1, "delta", d, dte=_d(k))]


_V1 = ["short_straddle", "short_strangle", "wide_strangle", "iron_condor", "iron_butterfly",
       "put_credit_spread", "call_credit_spread", "put_write", "call_overwrite",
       "short_put_atm", "short_call_atm", "put_ratio", "risk_reversal",
       "put_calendar", "call_calendar"]

_V2 = ["long_straddle", "long_strangle", "long_put_tail", "long_call",
       "put_backspread", "call_backspread", "call_ratio", "jade_lizard", "big_lizard",
       "broken_wing_condor", "broken_wing_put_fly", "call_butterfly", "put_butterfly",
       "double_calendar", "put_diagonal", "call_diagonal",
       "put_debit_spread", "call_debit_spread", "collar", "reverse_risk_reversal"]

ALL = _V1 + _V2                              # 35 structural archetypes (both vega signs)
