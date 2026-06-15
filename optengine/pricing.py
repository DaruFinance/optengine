"""optengine.pricing — self-contained option pricing & Greeks.

No QuantLib / py_vollib (neither installed). numpy + scipy only; fully vectorized.

Conventions
-----------
sigma : annualized vol, 0.20 == 20%.
T     : year fraction.
r, q  : continuous-compounded rate / carry (dividend yield for equity,
        foreign rate for FX, r for futures-style/Black-76).
cp    : +1 call, -1 put.

Greeks are returned in RAW units (per 1.00 move in the variable):
  delta  per 1.00 underlying
  gamma  per 1.00 underlying^2
  vega   per 1.00 vol  (==> per 1% == vega/100)
  theta  per 1.00 year (==> per calendar day == theta/365)
  rho    per 1.00 rate (==> per 1% == rho/100)
Callers scale to vendor conventions explicitly so the convention is never implicit.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import norm

__all__ = [
    "d1d2", "bs_price", "bs_greeks", "black76_price", "black76_greeks",
    "bachelier_price", "implied_vol", "forward_from_parity",
]

_CDF = norm.cdf
_PDF = norm.pdf


def d1d2(S, K, T, r, q, sigma):
    S = np.asarray(S, float); K = np.asarray(K, float); T = np.asarray(T, float)
    r = np.asarray(r, float); q = np.asarray(q, float); sigma = np.asarray(sigma, float)
    vsqrt = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / vsqrt
        d2 = d1 - vsqrt
    return d1, d2, vsqrt


def bs_price(S, K, T, r, q, sigma, cp):
    """Black-Scholes-Merton European price (carry q)."""
    d1, d2, _ = d1d2(S, K, T, r, q, sigma)
    cp = np.asarray(cp, float)
    dq = np.exp(-q * T); dr = np.exp(-r * T)
    call = S * dq * _CDF(d1) - K * dr * _CDF(d2)
    put = K * dr * _CDF(-d2) - S * dq * _CDF(-d1)
    px = np.where(cp > 0, call, put)
    # intrinsic at expiry / degenerate sigma*T -> 0
    intr = np.maximum(cp * (S - K), 0.0)
    return np.where(np.asarray(T) * np.asarray(sigma) <= 0, intr, px)


def bs_greeks(S, K, T, r, q, sigma, cp):
    """Raw-unit BSM Greeks. Returns dict(delta,gamma,vega,theta,rho)."""
    d1, d2, vsqrt = d1d2(S, K, T, r, q, sigma)
    cp = np.asarray(cp, float)
    pdf = _PDF(d1); dq = np.exp(-q * T); dr = np.exp(-r * T)
    S = np.asarray(S, float); K = np.asarray(K, float); T = np.asarray(T, float)
    sigma = np.asarray(sigma, float)
    delta = np.where(cp > 0, dq * _CDF(d1), dq * (_CDF(d1) - 1.0))
    gamma = dq * pdf / (S * sigma * np.sqrt(T))
    vega = S * dq * pdf * np.sqrt(T)
    theta = (-(S * dq * pdf * sigma) / (2.0 * np.sqrt(T))
             - cp * r * K * dr * _CDF(cp * d2)
             + cp * q * S * dq * _CDF(cp * d1))
    rho = cp * K * T * dr * _CDF(cp * d2)
    return dict(delta=delta, gamma=gamma, vega=vega, theta=theta, rho=rho)


def black76_price(F, K, T, r, sigma, cp):
    """Black-76 price for options on futures/forwards (premium-paid)."""
    return bs_price(F, K, T, r, r, sigma, cp)  # q==r collapses Se^{-qT}->Fe^{-rT}, drift->0


def black76_greeks(F, K, T, r, sigma, cp):
    g = bs_greeks(F, K, T, r, r, sigma, cp)
    return g


def bachelier_price(F, K, T, r, sigma_n, cp):
    """Normal/Bachelier model (rates: sigma_n in price/rate units, can be negative F-K)."""
    F = np.asarray(F, float); K = np.asarray(K, float); T = np.asarray(T, float)
    sigma_n = np.asarray(sigma_n, float); cp = np.asarray(cp, float)
    dr = np.exp(-r * T)
    s = sigma_n * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d = cp * (F - K) / s
    val = dr * (cp * (F - K) * _CDF(d) + s * _PDF(d))
    intr = dr * np.maximum(cp * (F - K), 0.0)
    return np.where(s <= 0, intr, val)


def implied_vol(price, S, K, T, r, q, cp, *, model="bs",
                lo=1e-4, hi=5.0, tol=1e-8, maxit=100):
    """Vectorized implied vol via bisection (robust; no Newton blow-ups at the wings).

    model: 'bs' (uses carry q) or 'b76' (futures, q==r ignored -> pass r as q).
    Returns nan where price is outside the no-arbitrage band.
    """
    price = np.asarray(price, float)
    pricer = (lambda sig: bs_price(S, K, T, r, q, sig, cp)) if model == "bs" \
        else (lambda sig: black76_price(S, K, T, r, sig, cp))
    lo = np.full(np.broadcast(price, S, K).shape, float(lo))
    hi = np.full_like(lo, float(hi))
    # no-arb band check (call: [max(0,Se^-qT-Ke^-rT), Se^-qT]); leave as nan if violated
    for _ in range(maxit):
        mid = 0.5 * (lo + hi)
        diff = pricer(mid) - price
        hi = np.where(diff > 0, mid, hi)
        lo = np.where(diff > 0, lo, mid)
        if np.all(hi - lo < tol):
            break
    iv = 0.5 * (lo + hi)
    # mark non-converged extremes as nan
    bad = (iv <= 1e-4 + tol) | (iv >= 5.0 - tol)
    return np.where(bad, np.nan, iv)


def forward_from_parity(strikes, call_mid, put_mid, index_price):
    """Forward F and discount D=e^{-rT} from put-call parity OLS: C-P = D*(F-K).

    The engine and the surface layer share this kernel so they agree.
    C - P = D*F - D*K  ->  slope=-D, intcpt=D*F. Returns (F, D).
    """
    K = np.asarray(strikes, float)
    y = np.asarray(call_mid, float) - np.asarray(put_mid, float)
    A = np.vstack([np.ones_like(K), -K]).T  # y = (D*F)*1 + D*(-K)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    DF, D = coef[0], coef[1]
    F = DF / D if D != 0 else float(index_price)
    return float(F), float(D)
