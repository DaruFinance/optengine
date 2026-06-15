"""optengine.american — American option pricing (early exercise).

Cox-Ross-Rubinstein binomial: correct and unambiguous, the trusted reference for
single-name equity options and options-on-futures (American). Vectorized across an
array of options (shared step count). A fast closed-form (Bjerksund-Stensland 2002)
is a later optimization, to be validated against THIS before use.

Early-exercise gates (standard): a call is exercised early only to capture carry
(q>0); a put only to capture interest on the strike (r>0). Elsewhere American==
European, so we short-circuit to the (exact, fast) BS closed form.
"""
from __future__ import annotations
import numpy as np
from . import pricing as _p

__all__ = ["binomial_american", "american_price"]


def _bc(args):
    arrs = [np.atleast_1d(np.asarray(x, float)) for x in args]
    n = max(a.size for a in arrs)
    return [np.broadcast_to(a, (n,)).astype(float) for a in arrs], n


def binomial_american(S, K, T, r, q, sigma, cp, steps=512):
    """CRR binomial American price, vectorized over an option array."""
    (S, K, T, r, q, sigma, cp), n = _bc((S, K, T, r, q, sigma, cp))
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt)); d = 1.0 / u
    p = np.clip((np.exp((r - q) * dt) - d) / (u - d), 0.0, 1.0)  # guard coarse-dt no-arb
    disc = np.exp(-r * dt)
    pu = (p * disc)[:, None]; pd = ((1.0 - p) * disc)[:, None]
    j = np.arange(steps + 1)
    ST = S[:, None] * u[:, None] ** j[None, :] * d[:, None] ** (steps - j[None, :])
    V = np.maximum(cp[:, None] * (ST - K[:, None]), 0.0)
    for i in range(steps - 1, -1, -1):
        jj = np.arange(i + 1)
        ST = S[:, None] * u[:, None] ** jj[None, :] * d[:, None] ** (i - jj[None, :])
        cont = pu * V[:, 1:i + 2] + pd * V[:, 0:i + 1]
        V = np.maximum(cont, cp[:, None] * (ST - K[:, None]))
    return V[:, 0]


def american_price(S, K, T, r, q, sigma, cp, steps=512):
    """American price; short-circuits to European BS where early exercise is
    provably suboptimal (calls with q<=0, puts with r<=0)."""
    (S, K, T, r, q, sigma, cp), n = _bc((S, K, T, r, q, sigma, cp))
    out = _p.bs_price(S, K, T, r, q, sigma, cp).astype(float).copy()
    need = np.where(cp > 0, q > 0.0, r > 0.0)   # early-ex possible only here
    if need.any():
        idx = np.where(need)[0]
        out[idx] = binomial_american(S[idx], K[idx], T[idx], r[idx], q[idx],
                                     sigma[idx], cp[idx], steps=steps)
    return out
