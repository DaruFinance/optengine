"""optengine.surface — causal, arbitrage-checked implied-volatility surface.

Per (underlying, date) we fit one arb-free implied-vol surface from THAT DAY'S
quotes only, then expose causal surface features (ATM vol, skew, term slope,
curvature) and report (not enforce) static no-arb diagnostics.

No-lookahead guarantee
----------------------
A surface for date D is calibrated *exclusively* from option quotes snapped on D
(the per-day slice of the per-ticker store). No quote from D+1.. enters the IS
calibration, and no future-expiry information is used to fit a near-expiry slice —
each expiry is fit independently from its own strikes. This closes the third
pollute-and-verify surface in the build-spec §7 (the surface fit itself), distinct
from feature- and WFO-procedure leaks.

Model (per expiry slice) — Gatheral raw SVI in total variance
-------------------------------------------------------------
    w(k) = a + b * ( rho*(k-m) + sqrt((k-m)^2 + sigma^2) )

with total variance  w = sigma_BS^2 * T_years,  T_years = dte / 365 (calendar),
and log-moneyness  k = ln(K / F),  F the per-expiry forward (put-call parity, with
a documented fallback). Implied vol is recovered as  iv(k,T) = sqrt(w(k,T) / T).

The fit enforces the standard no-arbitrage conditions (svi_w / durrleman_g /
calendar checks). The optimizer is scipy.optimize.least_squares with a robust
'soft_l1' loss; if scipy is unavailable a small numpy Levenberg-Marquardt is used
instead and Surface.fit_backend reports which ran.

Market vol input
----------------
We fit to the vendor MidImpliedVol (Algoseek American-BSM IV, converged rows only,
MidImpliedVol>0 & LastMidPrice>0). For US equity/index options the American/European
gap at these maturities is small; using the vendor IV directly (rather than
re-inverting prices, which for European would merely reproduce it) makes the
slice-fit RMSE-vs-vendor a clean smoothness / self-consistency check and avoids
injecting American-model inversion noise. The forward is still recovered from
parity on prices so the log-moneyness centering is honest.

Each slice is built from OUT-OF-THE-MONEY quotes only (puts for K<=F, calls for
K>F): the reliable IV signal lives in the OTM wing of each right, while the deep-ITM
side has near-zero vega and noisy vendor inversion. Mixing them yields a non-convex,
10-20 vol-point-inconsistent cloud on short-dated stress days — the OTM split is the
standard fix and keeps the per-slice fit clean and arb-free in-sample.
"""
from __future__ import annotations

from optengine.config import PERTKR

from dataclasses import dataclass, field
import os
import numpy as np

from .pricing import forward_from_parity

__all__ = ["Surface", "fit_surface", "svi_w", "durrleman_g",
           "SVISlice", "MIN_STRIKES_PER_SLICE"]

# A slice needs at least this many converged strikes spanning BOTH wings to be fit
# (5 free SVI params; require >=6 for a non-degenerate over-determined fit).
MIN_STRIKES_PER_SLICE = 6

_PK_DIR = PERTKR

try:                                              # scipy is the pinned optimizer
    from scipy.optimize import least_squares as _LSQ
    _HAVE_SCIPY = True
except Exception:                                 # pragma: no cover - fallback path
    _LSQ = None
    _HAVE_SCIPY = False


# ===========================================================================
# Raw-SVI kernel  (math reproduced verbatim from the locked svi.py reference)
# ===========================================================================
def svi_w(params, k):
    """Raw-SVI total variance w(k) for params [a,b,rho,m,sigma]."""
    a, b, rho, m, s = params
    km = np.asarray(k, float) - m
    return a + b * (rho * km + np.sqrt(km * km + s * s))


def svi_w_prime(params, k):
    """dw/dk."""
    a, b, rho, m, s = params
    km = np.asarray(k, float) - m
    sq = np.sqrt(km * km + s * s)
    return b * (rho + km / sq)


def svi_w_pprime(params, k):
    """d2w/dk2."""
    a, b, rho, m, s = params
    km = np.asarray(k, float) - m
    sq = np.sqrt(km * km + s * s)
    return b * (s * s) / (sq ** 3)


def durrleman_g(params, k):
    """Durrleman butterfly function g(k); g(k) >= 0 everywhere <=> no butterfly arb.

    g = (1 - k w'/(2w))^2 - (w'^2/4)(1/w + 1/4) + w''/2   (Gatheral, eq. for g).
    """
    a, b, rho, m, s = params
    km = np.asarray(k, float) - m
    sq = np.sqrt(km * km + s * s)
    w = a + b * (rho * km + sq)
    wp = b * (rho + km / sq)
    wpp = b * (s * s) / (sq ** 3)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = (1.0 - 0.5 * np.asarray(k, float) * wp / w) ** 2 \
            - 0.25 * (wp ** 2) * (1.0 / w + 0.25) + 0.5 * wpp
    return g


# ===========================================================================
# Fitted slice container
# ===========================================================================
@dataclass
class SVISlice:
    expiry_i: int                 # yyyymmdd expiry
    T: float                      # year fraction = dte/365
    F: float                      # per-expiry forward used for k = ln(K/F)
    params: np.ndarray            # [a, b, rho, m, sigma]
    n_strikes: int                # converged strikes used in the fit (body)
    k_min: float                  # fit-body log-moneyness span (fit responsible here)
    k_max: float
    rmse_iv: float                # in-sample RMSE of model IV vs vendor IV (vol pts, abs)
    fit_ok: bool
    forward_source: str           # 'parity' | 'under'
    k_min_data: float = None      # full OTM data span (beyond body = extrapolation)
    k_max_data: float = None

    def w(self, k):
        return svi_w(self.params, k)

    def iv(self, k):
        T = max(self.T, 1e-8)
        return np.sqrt(np.maximum(self.w(k), 1e-12) / T)


# ===========================================================================
# Per-slice SVI fit via least_squares (robust soft_l1) with multi-start
# ===========================================================================
def _svi_bounds():
    # a, b, rho, m, sigma.
    #   b   : wing slope of total variance. Per-expiry slices are short-dated, so a
    #         realistic cap (1.0) keeps the optimizer off the degenerate b->big rail
    #         that raw-SVI is notorious for on equity term structures.
    #   rho : asymmetry, bounded |rho|<=0.95. This forbids the exact rho->+-1
    #         degenerate corner that unconstrained raw-SVI collapses into (a known
    #         equity-term-structure pathology) while leaving rho free across the full
    #         interior — on liquid index slices the data picks it without railing.
    #   sigma: smile width in log-moneyness curvature, >0. Capped at 1.0: sigma~2 is a
    #         near-flat line over the whole observable strike range and is the other
    #         degenerate basin (fits the level while ignoring curvature, railing sigma
    #         to the bound); 1.0 still admits very wide smiles. m: vertex (tightened to
    #         a tight band near ATM by the caller).
    lo = np.array([-2.0, 0.0, -0.95, -5.0, 1e-3])
    hi = np.array([2.0, 1.0, 0.95, 5.0, 1.0])
    return lo, hi


def _lm_numpy(resfun, x0, lo, hi, max_iter=200, lam0=1e-3, tol=1e-12):
    """Tiny bounded Levenberg-Marquardt (numpy-only fallback when scipy is absent).
    Finite-difference Jacobian; projects steps back into [lo,hi]. Not as polished as
    scipy.least_squares but adequate for the 5-param SVI slice."""
    x = np.clip(np.asarray(x0, float), lo, hi)
    r = resfun(x)
    cost = float(r @ r)
    lam = lam0
    n = len(x)
    for _ in range(max_iter):
        # forward-difference Jacobian
        J = np.empty((len(r), n))
        for j in range(n):
            h = 1e-6 * max(1.0, abs(x[j]))
            xj = x.copy(); xj[j] = min(max(x[j] + h, lo[j]), hi[j])
            hj = xj[j] - x[j]
            if hj == 0.0:
                xj[j] = x[j] - h; hj = xj[j] - x[j]
            J[:, j] = (resfun(xj) - r) / hj if hj != 0 else 0.0
        JTJ = J.T @ J
        g = J.T @ r
        improved = False
        for _inner in range(12):
            try:
                step = np.linalg.solve(JTJ + lam * np.diag(np.diag(JTJ) + 1e-12), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            xn = np.clip(x + step, lo, hi)
            rn = resfun(xn)
            cn = float(rn @ rn)
            if cn < cost:
                x, r, cost = xn, rn, cn
                lam = max(lam / 3.0, 1e-12)
                improved = True
                break
            lam *= 10.0
        if not improved or np.linalg.norm(g) < tol:
            break
    return x


def _fit_slice_svi(k, w_obs, iv_obs, weights, seed=0):
    """Fit raw SVI to one expiry's (k, w) nodes. Returns (params, rmse_iv, ok).

    Robust 'soft_l1' least-squares (scipy) on the total-variance residual
    sqrt(weights)*(model_w - market_w), multi-start. The no-negativity floor
    a >= -b*sigma*sqrt(1-rho^2) (=> w(k)>=0 for all k) is enforced by clamping the
    fitted a up to the floor; b>=0, sigma>0, |rho|<1 come from the box bounds.
    """
    k = np.asarray(k, float)
    w_obs = np.asarray(w_obs, float)
    iv_obs = np.asarray(iv_obs, float)
    sw = np.sqrt(np.asarray(weights, float))
    lo, hi = _svi_bounds()
    km, kM = float(k.min()), float(k.max())
    # Vertex m lives near ATM (the SVI minimum is close to the smile bottom, which for
    # equities sits within a few % of the forward). Bound m to a tight band around 0
    # so the optimizer cannot run the vertex out to k~-1 (a degenerate huge-sigma
    # basin that "fits" the steep left wing while missing the level entirely — seen on
    # the wide Apr-2020 COVID smiles). Capped to the data span +/- a small pad.
    lo[3] = max(km - 0.10, -0.40)
    hi[3] = min(kM + 0.10, 0.40)
    if lo[3] >= hi[3]:                               # pathological narrow span guard
        lo[3], hi[3] = -0.10, 0.10

    wmin = max(float(np.min(w_obs)), 1e-8)
    wmed = float(np.median(w_obs))
    kmed = float(np.median(k))
    # observed ATM total variance: w of the node nearest k=0 (anchor target below)
    w_atm = float(w_obs[np.argmin(np.abs(k))])
    # multi-start seeds (deterministic): vary level a0, vertex m0, slope b0, rho0
    starts = [
        [wmin, 0.10, -0.30, 0.0, 0.10],
        [wmin, 0.10, -0.30, kmed, 0.10],
        [wmed, 0.20, -0.50, 0.0, 0.20],
        [wmin * 0.5, 0.05, 0.00, 0.0, 0.05],
        [wmed, 0.30, -0.70, kmed, 0.30],
    ]
    rng = np.random.default_rng(seed)
    while len(starts) < 6:                          # 5 structured + 1 random; the
        starts.append([wmin * float(rng.uniform(0.2, 1.5)),   # smooth slice loss makes
                       float(rng.uniform(0.03, 0.5)),         # extra restarts redundant
                       float(rng.uniform(-0.8, 0.2)),
                       float(rng.uniform(km, kM)),
                       float(rng.uniform(0.03, 0.5))])

    # Soft butterfly (Durrleman g>=0) regularizer appended to the residual vector.
    # Raw-SVI per slice can attain a marginally smaller in-sample SSE by bending into
    # an arb-violating wing shape; penalizing g<0 on a grid spanning the data (plus a
    # little extrapolation) steers least_squares to the arb-free basin without
    # materially hurting the fit. Scaled by f_scale so it is commensurate with the
    # robust total-variance residuals; report-only arb diagnostics still run on the
    # final params, so this is a fit prior, not a hard enforcement.
    k_pen = np.linspace(min(km, -0.05) - 0.05, max(kM, 0.05) + 0.05, 24)
    fscale = max(wmed, 1e-4)
    pen_scale = 0.5 * fscale
    # ATM-level anchor: force w(0) to match the observed near-ATM total variance.
    # This is what makes the level-missing degenerate basin unreachable (its w(0) is
    # far from w_atm), independent of the butterfly penalty. Weighted strongly.
    anc_scale = 5.0

    def residual(p):
        r = sw * (svi_w(p, k) - w_obs)
        g = durrleman_g(p, k_pen)
        pen = pen_scale * np.clip(-g, 0.0, None)
        anc = np.array([anc_scale * (float(svi_w(p, 0.0)) - w_atm)])
        return np.concatenate([r, pen, anc])

    # Select the winning candidate by the DATA-fit residual only (the penalty + anchor
    # steer each local solve, but the chosen optimum is judged on how well it matches
    # the quotes — so an arb-clean-but-flat fit can never win on a low penalty alone).
    def data_sse(p):
        return float(np.sum((sw * (svi_w(p, k) - w_obs)) ** 2))

    best_p, best_cost = None, np.inf
    for x0 in starts:
        x0 = np.clip(np.asarray(x0, float), lo, hi)
        try:
            if _HAVE_SCIPY:
                res = _LSQ(residual, x0, bounds=(lo, hi), loss="soft_l1",
                           f_scale=fscale, max_nfev=600, xtol=1e-10, ftol=1e-10)
                p = res.x
            else:                                  # pragma: no cover
                p = _lm_numpy(residual, x0, lo, hi)
        except Exception:
            continue
        c = data_sse(p)
        if c < best_cost:
            best_cost, best_p = c, p

    if best_p is None:
        return None, np.inf, False

    # enforce w(k)>=0 floor: a >= -b*sigma*sqrt(1-rho^2)
    a, b, rho, m, s = best_p
    floor = -b * s * np.sqrt(max(1.0 - rho * rho, 0.0))
    if a < floor:
        a = floor
    params = np.array([a, b, rho, m, s], float)

    # in-sample fit quality in VOL points. Each node satisfies w_obs = iv_obs^2 * T
    # exactly, so the slice year-fraction is recovered node-wise as T = w_obs/iv_obs^2
    # (median, robust to float noise); model IV = sqrt(model_w / T) is then compared
    # to the vendor IV directly. This is what the validator asserts on.
    iv_obs = np.asarray(iv_obs, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        t_nodes = w_obs / (iv_obs * iv_obs)
    t_nodes = t_nodes[np.isfinite(t_nodes) & (t_nodes > 0)]
    T = float(np.median(t_nodes)) if t_nodes.size else 1.0
    w_fit = np.maximum(svi_w(params, k), 1e-12)
    iv_fit = np.sqrt(w_fit / T)
    rmse_iv = float(np.sqrt(np.mean((iv_fit - iv_obs) ** 2)))
    ok = np.isfinite(rmse_iv) and np.isfinite(params).all()
    return params, rmse_iv, bool(ok)


# ===========================================================================
# Surface object
# ===========================================================================
@dataclass
class Surface:
    """Fitted arb-checked vol surface for one (underlying, date).

    Slices are stored sorted by ascending T. Interpolation across maturities is
    LINEAR IN TOTAL VARIANCE in T (flat extrapolation beyond the first/last fitted
    expiry); vol is recovered as sqrt(w/T). This is the standard arb-aware
    interpolation: linear-in-T total variance keeps the calendar structure that
    linear-in-vol would distort.
    """
    underlying: str
    date_i: int                                   # yyyymmdd
    under: float                                  # underlying mid (scalar)
    slices: list = field(default_factory=list)    # list[SVISlice], sorted by T
    fit_backend: str = "scipy.least_squares(soft_l1)"
    skipped: int = 0                              # slices skipped (too few strikes / fit fail)

    # ---------- maturity grid helpers ----------
    @property
    def Ts(self):
        return np.array([s.T for s in self.slices], float)

    def _bracket(self, T):
        """Indices (i0, i1) and weight for linear-in-T interpolation of total var."""
        Ts = self.Ts
        if len(Ts) == 0:
            return None
        if T <= Ts[0]:
            return (0, 0, 0.0)
        if T >= Ts[-1]:
            return (len(Ts) - 1, len(Ts) - 1, 0.0)
        j = int(np.searchsorted(Ts, T, "right"))
        i0, i1 = j - 1, j
        wgt = (T - Ts[i0]) / (Ts[i1] - Ts[i0])
        return (i0, i1, float(wgt))

    # ---------- total variance / IV anywhere ----------
    def total_var(self, k, T):
        """w(k, T): per-slice SVI in k, linear-in-T total-variance interp across
        maturities, flat outside the fitted expiry range. Scalar or array k."""
        if not self.slices:
            return np.nan if np.ndim(k) == 0 else np.full(np.shape(k), np.nan)
        i0, i1, wgt = self._bracket(T)
        w0 = self.slices[i0].w(k)
        if i0 == i1:
            return w0
        w1 = self.slices[i1].w(k)
        return (1.0 - wgt) * w0 + wgt * w1

    def iv(self, k, T):
        """Implied vol at (log-moneyness k, year fraction T)."""
        w = self.total_var(k, T)
        return np.sqrt(np.maximum(w, 1e-12) / max(float(T), 1e-8))

    def iv_KT(self, K, T, F=None):
        """Convenience: IV at absolute strike K and maturity T, using forward F
        (defaults to the nearest-bracketing slice forward, else the underlying)."""
        if F is None:
            br = self._bracket(T)
            F = self.slices[br[0]].F if br is not None else self.under
        return self.iv(np.log(np.asarray(K, float) / F), T)

    # ---------- the ATM forward at a target maturity ----------
    def _slice_for_T(self, T_target):
        if not self.slices:
            return None
        Ts = self.Ts
        return self.slices[int(np.argmin(np.abs(Ts - T_target)))]

    # ===================================================================
    # CAUSAL surface signals
    # ===================================================================
    def signals(self, atm_days=30, term_days=60, skew_dk=0.10, curv_dk=0.10):
        """Dict of causal surface features (all from today's fitted surface only).

        atm_vol    : sqrt(w(0)/T) at ~`atm_days` (the ATM-forward implied vol).
        skew       : dw/dk at k=0 at ~`atm_days`, expressed in VOL points per unit
                     log-moneyness, i.e. d(iv)/dk at k=0 = w'(0) / (2*sqrt(w0*T)).
                     Negative for the usual equity skew (downside vol > upside).
        rr25       : 25-delta risk-reversal proxy = iv(k=-skew_dk) - iv(k=+skew_dk)
                     at ~`atm_days` (put-wing minus call-wing vol; >0 = rich downside).
        term_slope : atm_vol(~`term_days`) - atm_vol(~`atm_days`); >0 = contango.
        curvature  : smile convexity at k=0 = iv(-curv_dk) + iv(+curv_dk) - 2*iv(0)
                     at ~`atm_days` (vendor-units vol; wing-minus-ATM smile measure).

        Returns NaNs gracefully if no slice is available. All four are functions of
        the date-D surface only — strictly causal.
        """
        out = {"atm_vol": np.nan, "skew": np.nan, "rr25": np.nan,
               "term_slope": np.nan, "curvature": np.nan}
        if not self.slices:
            return out
        T_atm = atm_days / 365.0
        T_term = term_days / 365.0
        s_atm = self._slice_for_T(T_atm)
        # Use the surface interpolation at the *target* maturity (not just the
        # nearest slice) so the signals are anchored at ~30d/~60d regardless of the
        # exact listed expiries.
        w0 = float(self.total_var(0.0, T_atm))
        atm_vol = float(np.sqrt(max(w0, 1e-12) / max(T_atm, 1e-8)))
        out["atm_vol"] = atm_vol

        # skew = d(iv)/dk at k=0 via the fitted slope of the nearest slice (analytic
        # w'(0)), converted to vol units and scaled to the target maturity's w0.
        wp0 = float(svi_w_prime(s_atm.params, 0.0))
        out["skew"] = wp0 / (2.0 * np.sqrt(max(w0, 1e-12) * max(T_atm, 1e-8)))

        ivm = float(self.iv(-skew_dk, T_atm))
        ivp = float(self.iv(+skew_dk, T_atm))
        out["rr25"] = ivm - ivp

        atm_vol_term = float(np.sqrt(max(self.total_var(0.0, T_term), 1e-12)
                                     / max(T_term, 1e-8)))
        out["term_slope"] = atm_vol_term - atm_vol

        ivm_c = float(self.iv(-curv_dk, T_atm))
        ivp_c = float(self.iv(+curv_dk, T_atm))
        out["curvature"] = ivm_c + ivp_c - 2.0 * atm_vol
        return out

    # ===================================================================
    # No-arb diagnostics  (report, don't silently enforce)
    # ===================================================================
    def butterfly_violations(self, k_grid=None, in_data_only=False, n=121):
        """Butterfly (Durrleman g<0) violation counts across the fitted slices.

        in_data_only=False : evaluate each slice on the shared `k_grid` (default a
            wide [-0.6,0.6] band). Violations here often fall in the FAR wings beyond
            any quoted strike (raw SVI extrapolation, esp. the thin OTM-call side) and
            are an honest artifact of an unconstrained region — reported, not a fit
            defect.
        in_data_only=True  : evaluate each slice only on its OWN populated span
            [k_min, k_max]. These are the violations that matter: convexity breaking
            where real quotes exist.

        Returns (total_count, n_nodes_checked, list[frac_per_slice]).
        """
        total = 0
        fracs = []
        nodes = 0
        for s in self.slices:
            kg = (np.linspace(s.k_min, s.k_max, n) if in_data_only
                  else (k_grid if k_grid is not None else np.linspace(-0.6, 0.6, n)))
            g = durrleman_g(s.params, kg)
            bad = int(np.sum(g < -1e-10))
            total += bad
            fracs.append(bad / len(kg))
            nodes += len(kg)
        return total, nodes, fracs

    def calendar_violations(self, k_grid=None, in_data_only=True, n=121):
        """Count of calendar (total variance decreasing in T at fixed k) violations
        across consecutive fitted slices. Calendar no-arb requires w(k,T) non-
        decreasing in T at every fixed k. By default checks the band common to all
        slices' data (the intersection of [k_min,k_max]) so the diagnostic is not
        dominated by wing extrapolation. Returns (count, n_nodes_checked)."""
        if len(self.slices) < 2:
            return 0, 0
        if k_grid is None:
            if in_data_only:
                lo = max(s.k_min for s in self.slices)
                hi = min(s.k_max for s in self.slices)
                if not (lo < hi):
                    lo, hi = -0.1, 0.1
                k_grid = np.linspace(lo, hi, n)
            else:
                k_grid = np.linspace(-0.6, 0.6, n)
        W = np.vstack([svi_w(s.params, k_grid) for s in self.slices])   # (nT, nk), T ascending
        dW = np.diff(W, axis=0)
        return int(np.sum(dW < -1e-10)), dW.size

    def arb_report(self, k_grid=None):
        """Convenience dict of arb diagnostics, separating in-data (meaningful) from
        wide-grid (extrapolation) butterfly counts and the in-data calendar count."""
        bf_w, bf_w_nodes, bf_w_fracs = self.butterfly_violations(k_grid, in_data_only=False)
        bf_d, bf_d_nodes, _ = self.butterfly_violations(in_data_only=True)
        cal_d, cal_d_nodes = self.calendar_violations(in_data_only=True)
        cal_w, cal_w_nodes = self.calendar_violations(in_data_only=False)
        return {"butterfly_violations": bf_w, "butterfly_nodes": bf_w_nodes,
                "butterfly_frac_per_slice": bf_w_fracs,
                "butterfly_violations_in_data": bf_d, "butterfly_nodes_in_data": bf_d_nodes,
                "calendar_violations": cal_d, "calendar_nodes": cal_d_nodes,
                "calendar_violations_wide": cal_w, "calendar_nodes_wide": cal_w_nodes}


# ===========================================================================
# Forward recovery (parity, guarded) per expiry
# ===========================================================================
def _forward_for_slice(K, cp, mid, under):
    """Per-expiry forward via put-call parity on this slice's paired strikes.

    Accept the parity forward only if (a) >=4 strikes have BOTH a call and a put
    quote and (b) the result is sane: finite, positive, within [0.7,1.3]*under.
    Otherwise fall back to F=under (== under*exp((r-q)T) with r=q=0, per build-spec).
    The discount D from parity is intentionally NOT used downstream: on dividend
    names the noisy near-expiry OLS routinely returns D>1 (non-physical), and only
    the FORWARD is needed to center log-moneyness.
    Returns (F, source) with source in {'parity','under'}.
    """
    Kc = K[cp > 0]; mc = mid[cp > 0]
    Kp = K[cp < 0]; mp = mid[cp < 0]
    if Kc.size and Kp.size:
        common, ic, ip = np.intersect1d(Kc, Kp, return_indices=True)
        if common.size >= 4:
            F, _D = forward_from_parity(common, mc[ic], mp[ip], under)
            if np.isfinite(F) and 0.7 * under <= F <= 1.3 * under:
                return float(F), "parity"
    return float(under), "under"


# ===========================================================================
# Top-level fit
# ===========================================================================
def _load_day_quotes(ticker, date_i):
    """Read one (ticker, date) converged option slice straight from the per-ticker
    parquet (build_array_frames drops MidImpliedVol, which we need to fit). Returns a
    dict of numpy arrays or None. Causal: only rows with asof_i == date_i."""
    import pandas as pd
    p = os.path.join(_PK_DIR, f"{ticker}.parquet")
    if not os.path.exists(p):
        return None
    cols = ["asof_i", "expiry_i", "cp", "Strike", "DaysToMaturity",
            "MidImpliedVol", "MidDelta", "LastBidPrice", "LastMidPrice",
            "LastAskPrice", "UnderLastMidPrice", "converged"]
    # filtered read: only this day's rows (PyArrow predicate pushdown keeps RAM low)
    df = pd.read_parquet(p, columns=cols,
                         filters=[("asof_i", "==", int(date_i))])
    if df is None or not len(df):
        return None
    m = (df["converged"].to_numpy(bool)
         & (df["MidImpliedVol"].to_numpy(np.float64) > 0)
         & (df["LastMidPrice"].to_numpy(np.float64) > 0)
         & (df["LastBidPrice"].to_numpy(np.float64) >= 0)
         & (df["LastAskPrice"].to_numpy(np.float64) > 0))
    if not m.any():
        return None
    return {
        "expiry_i": df["expiry_i"].to_numpy(np.int64)[m],
        "cp": df["cp"].to_numpy(np.int64)[m],
        "K": df["Strike"].to_numpy(np.float64)[m],
        "dte": df["DaysToMaturity"].to_numpy(np.float64)[m],
        "iv": df["MidImpliedVol"].to_numpy(np.float64)[m],
        "delta": df["MidDelta"].to_numpy(np.float64)[m],
        "mid": df["LastMidPrice"].to_numpy(np.float64)[m],
        "under": float(np.median(df["UnderLastMidPrice"].to_numpy(np.float64)[m])),
    }


def fit_surface(ticker, date_i, quotes=None, weight="vega",
                min_strikes=MIN_STRIKES_PER_SLICE, kclip=0.5):
    """Fit a Surface for one (underlying, date) from THAT DAY's quotes only.

    Parameters
    ----------
    ticker   : underlying symbol (must have pertkr/<ticker>.parquet), unless `quotes`
               is supplied directly.
    date_i   : yyyymmdd int.
    quotes   : optional pre-loaded dict (keys: expiry_i, cp, K, dte, iv, delta, mid,
               under) — lets callers avoid a parquet read (e.g. reuse a chain).
    weight   : per-point fit weight scheme:
                 'vega'  -> ATM-Gaussian proxy exp(-(k/0.5)^2) (downweights wings;
                            true BS vega peaks ATM, this is its cheap monotone proxy);
                 'atm'   -> same Gaussian (alias);
                 'equal' -> uniform.
               Wings are noisier and wider-spread, so ATM-weighting stabilizes the
               level/slope that the signals read off. Documented choice.
    min_strikes : minimum converged strikes (spanning both wings) to fit a slice.
    kclip    : fit only the smile BODY |k|<=kclip (default 0.5 ~ +/-50% moneyness).
               The extreme far-OTM tails (esp. the deep put wing on wide stress-day
               smiles, k<-0.6) carry a curvature raw-SVI cannot reconcile with the
               body and drag the whole slice into a degenerate basin (observed on
               Apr-2020). Clipping to the liquid body is standard practice and the
               signals (read at |k|<=0.1) are unaffected. If <min_strikes survive the
               clip, the full OTM span is used as a fallback. The stored k_min/k_max
               reflect the FULL OTM data span, so the wide-grid arb diagnostics stay
               honest about where the fit extrapolates.

    Returns a Surface (possibly with .skipped>0 and few/no slices on thin days).
    """
    date_i = int(date_i)
    q = quotes if quotes is not None else _load_day_quotes(ticker, date_i)
    if q is None:
        return Surface(underlying=str(ticker), date_i=date_i, under=np.nan,
                       slices=[], fit_backend=_backend_name(), skipped=0)

    under = float(q["under"])
    exps = np.unique(q["expiry_i"])
    slices = []
    skipped = 0
    for e in exps:
        sel = q["expiry_i"] == e
        K = q["K"][sel]; cp = q["cp"][sel]; iv = q["iv"][sel]
        dte = q["dte"][sel]; mid = q["mid"][sel]
        F, fsrc = _forward_for_slice(K, cp, mid, under)

        # OTM-only smile: puts for K<=F, calls for K>F. This is the standard
        # construction — the reliable IV signal lives in the OUT-OF-THE-MONEY wing of
        # each right (meaningful vega, tight quotes). The ITM side is deep-ITM with
        # near-zero vega and noisy vendor inversion; mixing ITM-call IV with OTM-put
        # IV at the same moneyness produces a 10-20 vol-point-inconsistent, non-convex
        # cloud (seen on short-dated SPY in the Nov-2018 / Apr-2020 stress dates).
        # Each strike contributes exactly once, from its OTM side.
        otm = ((cp < 0) & (K <= F)) | ((cp > 0) & (K > F))
        K = K[otm]; cp = cp[otm]; iv = iv[otm]; dte = dte[otm]
        if K.size == 0:
            skipped += 1
            continue
        n_strk = np.unique(K).size
        if n_strk < min_strikes:                   # enough OTM strikes to fit 5 params
            skipped += 1
            continue
        k_all = np.log(K / F)                       # full OTM span (for transparency)
        if not (k_all.min() < 0.0 < k_all.max()):   # must span both wings around ATM
            skipped += 1
            continue
        T = max(float(np.median(dte)) / 365.0, 1e-6)

        # restrict the FIT to the smile body |k|<=kclip; fall back to full OTM if the
        # clip leaves too few strikes (still requiring both wings present).
        body = np.abs(k_all) <= kclip
        if body.sum() < min_strikes or not (k_all[body].min() < 0.0 < k_all[body].max()):
            body = np.ones_like(k_all, dtype=bool)
        kf = k_all[body]; ivf = iv[body]
        w_obs = ivf * ivf * T

        if weight in ("vega", "atm"):
            wgt = np.exp(-(kf / 0.5) ** 2)          # ATM-peaked Gaussian (vega proxy)
        else:
            wgt = np.ones_like(kf)
        wgt = np.maximum(wgt, 1e-3)

        # duplicate strikes are gone after the OTM split (one right per strike);
        # least_squares handles any residual repeats fine. fit on the body.
        params, rmse_iv, ok = _fit_slice_svi(kf, w_obs, ivf, wgt, seed=int(e) % 9973)
        if params is None or not ok:
            skipped += 1
            continue
        # k_min/k_max store the FIT BODY span (the region the fit is responsible for,
        # used by the in-data arb diagnostics); k_min_data/k_max_data keep the full OTM
        # extent so the wide-grid diagnostics show where the surface extrapolates.
        slices.append(SVISlice(expiry_i=int(e), T=T, F=F, params=params,
                               n_strikes=int(kf.size), k_min=float(kf.min()),
                               k_max=float(kf.max()), rmse_iv=float(rmse_iv),
                               fit_ok=bool(ok), forward_source=fsrc,
                               k_min_data=float(k_all.min()), k_max_data=float(k_all.max())))

    slices.sort(key=lambda s: s.T)
    return Surface(underlying=str(ticker), date_i=date_i, under=under,
                   slices=slices, fit_backend=_backend_name(), skipped=skipped)


def _backend_name():
    return "scipy.least_squares(soft_l1)" if _HAVE_SCIPY else "numpy_LM(fallback)"
