"""Core state-space engine for the hashrate filter.

The model (paper, Section 4.3):

    observation:  y_t = h_t + eps_t,   Var(eps_t) = R1 = 1/144   (protocol-fixed)
    state:        h_t = h_{t-1} + eta_t,   Var(eta_t) = q_t

with y_t the raw daily log hashrate estimate and h_t the latent daily log
rate. A day at the prevailing rate yields a Poisson block count with
expectation 144; the delta method carries the count variance to the log
estimate, so the observation equation carries no estimated parameter. The
state variance blends a quiet-day variance and a jump variance through the
arrival-time weight: q_t = (1 - pi_t) q_quiet + pi_t q_jump.

One-sided: the Kalman recursion, which moves each day's estimate toward the
raw estimate by K_t = V_t / (V_t + 1/144) of the gap and uses no later
observations.

Two-sided: the fixed-interval Rauch-Tung-Striebel smoother over the same
model. It uses later days as well and is returned only as a benchmark.

Extended filter: on the days the protocol resets the difficulty, a second
observation anchors the level. Because the difficulty reflects the average
hashrate over the epoch that just ended, it lags the current rate when
hashrate trends; the observation is log D_t + offset + 7 s_t, where s_t is
the OLS slope of the last 14 filtered days (one epoch), projected half an
epoch (7 days), and the observation variance is 1/2016 + (7 se_t)^2, with
se_t the standard error of that slope. On reset days the measured growth
can therefore jump when the new difficulty enters as an observation; the
paper discloses this behaviour.

The observation variance is fixed by the protocol, q_jump is calibrated from
the dispersion on days with pi_t > 0.7, and q_quiet is the one parameter
estimated by maximum likelihood on the filter's one-step prediction errors.
"""

from dataclasses import dataclass, field
from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

# --- constant inventory -------------------------------------------------------
R1 = 1.0 / 144.0          # observation variance, daily count (protocol-fixed)
R2 = 1.0 / 2016.0         # anchor observation variance, epoch length (protocol-fixed)
Q_JUMP = 1.51e-2          # jump-regime state variance (calibrated: mean squared
                          # one-sided filtered daily growth on days with pi_t > 0.7)
P0 = 0.05                 # initial state variance (chosen)
EPOCH_DAYS = 2016.0 / 144.0   # expected length of a difficulty epoch in days
DELAG_LAG = 14            # slope window of the de-lag regression (days, one epoch)
DELAG_PROJ = EPOCH_DAYS / 2.0  # projection horizon, half an epoch (days)
PROTOCOL_OFFSET = float(np.log(2**32 / 600.0))  # log H - log D at ten-minute blocks
_MLE_BOUNDS = (-18.0, -6.0)  # bounds on log q_quiet


@dataclass
class FilterResult:
    """Output of the hashrate filter.

    Attributes
    ----------
    h_filt : np.ndarray
        One-sided estimate of the log daily hashrate, in log of the input
        units. Exponentiate for hashrate units.
    P_filt : np.ndarray
        Its variance in squared log units, so h_filt +/- 1.96 sqrt(P_filt)
        is the 95 percent band.
    h_smooth : np.ndarray or None
        Two-sided (RTS-smoothed) estimate, if two_sided=True, else None.
    P_smooth : np.ndarray or None
        Variance of the two-sided estimate, if two_sided=True.
    q_quiet : float
        Quiet-day state variance (the one fitted parameter).
    loglik : float
        Log-likelihood of the one-step prediction errors at q_quiet, up to
        the additive constant -n/2 log(2 pi). Day 0 contributes a zero
        innovation because the prior mean is y[0]. For the extended filter
        the difficulty observations are included.
    q_path : np.ndarray
        The daily state variance q_t used in the recursion.
    anchor_var : np.ndarray or None
        The variance of the difficulty observation on each day (extended
        filter): 1/2016 + (7 se_t)^2 on reset days, zero elsewhere.
    """

    h_filt: np.ndarray
    P_filt: np.ndarray
    h_smooth: Optional[np.ndarray]
    P_smooth: Optional[np.ndarray]
    q_quiet: float
    loglik: float
    q_path: np.ndarray = field(repr=False)
    anchor_var: Optional[np.ndarray] = None


def _check_pi(pi: np.ndarray, name: str) -> np.ndarray:
    pi = np.asarray(pi, dtype=float)
    if not np.all(np.isfinite(pi)) or np.any((pi < 0.0) | (pi > 1.0)):
        raise ValueError(f"{name} must be finite and inside [0, 1]")
    return pi


def q_pure(pi_d: np.ndarray, q_quiet: float) -> np.ndarray:
    """Two-regime blend: q_t = (1 - pi_t) q_quiet + pi_t q_jump."""
    pi_d = _check_pi(pi_d, "pi_d")
    return (1.0 - pi_d) * q_quiet + pi_d * Q_JUMP


def _forward(
    y: np.ndarray,
    q: np.ndarray,
    anchor_mask: Optional[np.ndarray] = None,
    anchor_target: Optional[np.ndarray] = None,
):
    """One forward pass of the recursion.

    Returns (h_filt, P_filt, h_pred, P_pred, loglik, anchor_var), where the
    pred arrays hold the pre-update (one-step-ahead) moments needed by the
    RTS backward pass. On reset days the anchor target is de-lagged by the
    OLS slope of the last DELAG_LAG filtered states (including the current
    post-update state) projected DELAG_PROJ days, and the anchor variance is
    R2 + (DELAG_PROJ x se)^2, with se the standard error of that slope; the
    per-day anchor variance is returned as the last element (zeros on
    non-anchor days).
    """
    y = np.asarray(y, dtype=float)
    q = np.asarray(q, dtype=float)
    n = len(y)
    h_filt = np.empty(n)
    P_filt = np.empty(n)
    h_pred = np.empty(n)
    P_pred = np.empty(n)
    anchor_var = np.zeros(n)
    loglik = 0.0
    hp = y[0]
    Pp = P0
    anchored = anchor_mask is not None
    hist = []
    for t in range(n):
        Pt = Pp + q[t]
        h_pred[t] = hp
        P_pred[t] = Pt
        S = Pt + R1
        iv = y[t] - hp
        loglik += -0.5 * (np.log(S) + iv * iv / S)
        K = Pt / S
        hp = hp + K * iv
        Pp = (1.0 - K) * Pt
        hist.append(hp)
        if anchored and anchor_mask[t]:
            if len(hist) > DELAG_LAG + 1:
                yy = np.asarray(hist[-DELAG_LAG:])
                x = np.arange(DELAG_LAG, dtype=float)
                sx = x - x.mean()
                sy = yy - yy.mean()
                slope = float(np.sum(sx * sy) / np.sum(sx * sx))
                resid = sy - slope * sx
                se = float(np.sqrt(np.sum(resid**2) / (DELAG_LAG - 2) / np.sum(sx * sx)))
            else:
                slope = 0.0
                se = 0.0
            avar = R2 + (DELAG_PROJ * se) ** 2
            tgt = anchor_target[t] + slope * DELAG_PROJ
            S2 = Pp + avar
            iv2 = tgt - hp
            loglik += -0.5 * (np.log(S2) + iv2 * iv2 / S2)
            K2 = Pp / S2
            hp = hp + K2 * iv2
            Pp = (1.0 - K2) * Pp
            hist[-1] = hp
            anchor_var[t] = avar
        h_filt[t] = hp
        P_filt[t] = Pp
    return h_filt, P_filt, h_pred, P_pred, loglik, anchor_var


def _rts_backward(
    h_filt: np.ndarray, P_filt: np.ndarray, h_pred: np.ndarray, P_pred: np.ndarray
):
    """Fixed-interval RTS smoother over the identical model (two-sided).

    h_s[t] = h_f[t] + J_t (h_s[t+1] - h_p[t+1]),   J_t = P_f[t] / P_p[t+1],
    P_s[t] = P_f[t] + J_t^2 (P_s[t+1] - P_p[t+1]).
    """
    n = len(h_filt)
    h_smooth = h_filt.copy()
    P_smooth = P_filt.copy()
    for t in range(n - 2, -1, -1):
        J = P_filt[t] / P_pred[t + 1]
        h_smooth[t] = h_filt[t] + J * (h_smooth[t + 1] - h_pred[t + 1])
        P_smooth[t] = P_filt[t] + J * J * (P_smooth[t + 1] - P_pred[t + 1])
    return h_smooth, P_smooth


def fit_q_quiet(
    y: np.ndarray,
    pi_d: np.ndarray,
    anchor_mask: Optional[np.ndarray] = None,
    anchor_target: Optional[np.ndarray] = None,
) -> float:
    """Maximum likelihood of the quiet-day state variance.

    Parameters
    ----------
    y : array-like
        Raw daily log hashrate estimates, length T.
    pi_d, anchor_mask, anchor_target
        As in run_filter.

    Returns
    -------
    float
        The quiet-day state variance (squared log units per day) maximising
        the one-step prediction-error log-likelihood over log q in
        [exp(-18), exp(-6)]. Warns if the optimum hits a bound.
    """
    y = np.asarray(y, dtype=float)
    pi_d = np.asarray(pi_d, dtype=float)

    def neg_ll(lq):
        q = q_pure(pi_d, np.exp(lq))
        return -_forward(y, q, anchor_mask, anchor_target)[4]

    res = minimize_scalar(neg_ll, bounds=_MLE_BOUNDS, method="bounded")
    lo, hi = _MLE_BOUNDS
    if not res.success or min(res.x - lo, hi - res.x) < 1e-3:
        warnings.warn(
            f"q_quiet MLE at or near a bound (log q = {res.x:.2f}); check the input series"
        )
    return float(np.exp(res.x))


def _check_dates(dates, n: int) -> None:
    """Require one row per consecutive calendar day when dates are given."""
    idx = pd.DatetimeIndex(dates).normalize()
    deltas = np.diff(idx.values).astype("timedelta64[D]").astype(int)
    bad = np.flatnonzero(deltas != 1)
    if len(bad):
        raise ValueError(
            "dates are not consecutive calendar days: first gap at position "
            f"{int(bad[0]) + 1}. The recursion steps one row per day; pass a "
            "daily series without gaps."
        )


def run_filter(
    y: np.ndarray,
    pi_d: Optional[np.ndarray] = None,
    q_quiet: Optional[float] = None,
    anchor_mask: Optional[np.ndarray] = None,
    anchor_target: Optional[np.ndarray] = None,
    two_sided: bool = False,
    dates=None,
) -> FilterResult:
    """Run the hashrate filter on a raw daily log-hashrate series.

    Parameters
    ----------
    y : array-like
        Raw daily log hashrate estimates, one row per consecutive calendar
        day.
    pi_d : array-like or None
        Daily arrival-time weights in [0, 1] (see evidence.daily_evidence),
        aligned to y. Zeros if None: the quiet-day variance on every day.
    q_quiet : float or None
        Quiet-day state variance, positive and finite. Fitted by MLE if None.
    anchor_mask, anchor_target : array-like or None
        Extended filter only: boolean mask of difficulty-reset days and the
        anchor base target log D_t + offset on those days. On reset days the
        target is de-lagged by the OLS slope of the last 14 filtered days
        projected 7 days, and the observation variance is
        1/2016 + (7 se)^2.
    two_sided : bool
        Also run the RTS backward pass and return the two-sided path.
    dates : array-like of datetime-like or None
        Calendar dates of the rows. If given, they must be consecutive days.

    Returns
    -------
    FilterResult
    """
    y = np.asarray(y, dtype=float)
    if not np.all(np.isfinite(y)):
        raise ValueError("y contains non-finite values (NaN or inf)")
    n = len(y)
    if n < 2:
        raise ValueError("y needs at least 2 days")
    if dates is not None:
        if len(dates) != n:
            raise ValueError(f"dates has length {len(dates)}, y has {n}")
        _check_dates(dates, n)
    for name, a in (("pi_d", pi_d), ("anchor_mask", anchor_mask), ("anchor_target", anchor_target)):
        if a is not None and len(a) != n:
            raise ValueError(f"{name} has length {len(a)}, y has {n}")
    if q_quiet is not None and not (np.isfinite(q_quiet) and q_quiet > 0):
        raise ValueError("q_quiet must be a positive finite number")

    pi_d = np.zeros(n) if pi_d is None else _check_pi(pi_d, "pi_d")
    if q_quiet is None:
        q_quiet = fit_q_quiet(y, pi_d, anchor_mask, anchor_target)
    q = q_pure(pi_d, q_quiet)

    h_filt, P_filt, h_pred, P_pred, loglik, anchor_var = _forward(
        y, q, anchor_mask, anchor_target
    )
    h_smooth = P_smooth = None
    if two_sided:
        h_smooth, P_smooth = _rts_backward(h_filt, P_filt, h_pred, P_pred)
    return FilterResult(
        h_filt=h_filt,
        P_filt=P_filt,
        h_smooth=h_smooth,
        P_smooth=P_smooth,
        q_quiet=float(q_quiet),
        loglik=float(loglik),
        q_path=q,
        anchor_var=anchor_var if anchor_mask is not None else None,
    )
