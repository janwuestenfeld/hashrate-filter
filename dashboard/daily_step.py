"""One-day update steps for the live hashrate dashboard.

The batch package (hashrate_filter) always runs from the first day of the
sample. The live dashboard must not re-run the whole history: each day it
adds one new day to the stored state. This module carries the three one-day
steps, written from the model description in the repository README and the
paper, with no private imports:

  * day_weight      the arrival-time weight pi_t of one day from the day's
                    block header timestamps, against the stored scan null
                    and the stored prior (the reference is fixed, so a new
                    day's weight is computed the same way every time);
  * filter_step     the one-sided Kalman update of the filtered log level
                    and its variance for one new day;
  * rts_tail        the two-sided (Rauch-Tung-Striebel) path over the tail
                    window. Only the tail is re-smoothed: the two-sided
                    line uses the following days by construction, so its
                    last days are revised as new days arrive; older days
                    keep their stored values.

Model (paper, Section 4.3; README, "The model"):

    observation:  y_t = h_t + eps_t,      Var(eps_t) = 1/144
    state:        h_t = h_{t-1} + eta_t,  Var(eta_t) = q_t = (1 - pi_t) q_quiet + pi_t q_jump

The one-day update moves the estimate toward the raw estimate by
K_t = V_t / (V_t + 1/144) of the gap.

The weight's construction (README, "The model"): the centroid statistic
z_t = (m_t - 1/2) sqrt(12 n_t), with m_t the mean arrival time as a
fraction of the day and n_t the block count; a scan statistic over 24
hourly bins (the maximum two-sample log-likelihood ratio over split
points of the cumulative intraday counts); the two p-values combined by
Fisher's method; the combined p-value mapped into pi_t with the stored
prior (threshold 0.001, likelihood ratio capped at 1000; README,
"Constants"). The scan p-value is read off the stored Monte Carlo null
(50,000 replications, seed 123, the settings of the paper's series).
"""

from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

from hashrate_filter.engine import Q_JUMP, R1

# --- constants of the weight construction (README, "Constants") ---------------
SCAN_BINS = 24          # intraday bins of the scan statistic
P_THRESHOLD = 1e-3      # p-value threshold of the likelihood ratio
LR_CAP = 1000.0         # cap of the likelihood ratio in the weight map

# Two-sided tail window (days). Smoother revisions decay by the gain
# J_t = P_filt[t] / P_pred[t+1] per day; 60 days is far beyond the point
# where a revision is visible (about 1e-4 in log units even after a jump).
TAIL_WINDOW = 60


def scan_statistic(counts: np.ndarray) -> float:
    """Max two-sample log-likelihood ratio over split points of the
    cumulative intraday bin counts (24 hourly bins)."""
    cum = np.cumsum(np.asarray(counts, dtype=float))
    total = cum[-1]
    if total <= 0.0:
        return 0.0
    nbins = len(cum)
    lr = 0.0
    for k in range(1, nbins):
        pk = k / nbins
        xx = cum[k - 1]
        term = xx * np.log(max(xx / (total * pk), 1e-12)) + (total - xx) * np.log(
            max((total - xx) / (total * (1 - pk)), 1e-12)
        )
        lr = max(lr, 2.0 * term)
    return float(lr)


def centroid_z(fracs: np.ndarray) -> float:
    """Centroid statistic z = (m - 1/2) sqrt(12 n); standard normal under a
    constant within-day rate."""
    fracs = np.asarray(fracs, dtype=float)
    n = len(fracs)
    if n == 0:
        return 0.0
    return float((fracs.mean() - 0.5) * np.sqrt(12.0 * n))


def day_weight(
    header_times: pd.Series, scan_null: np.ndarray, prior: float
) -> Tuple[float, int, float, float]:
    """Arrival-time weight of one day from the day's block header timestamps.

    Parameters
    ----------
    header_times : pd.Series
        Block header timestamps of the day (UTC). Blocks are assigned to
        days by header time here (the weight uses arrival times), not by
        median time.
    scan_null : np.ndarray
        Stored Monte Carlo null of the scan statistic (sorted).
    prior : float
        Stored prior probability of a real shift.

    Returns
    -------
    (pi, n, z, scan): the daily weight, the block count, the centroid
    statistic and the scan statistic.
    """
    t = pd.to_datetime(header_times)
    if t.dt.tz is not None:
        t = t.dt.tz_convert("UTC").dt.tz_localize(None)
    day = t.dt.floor("D")
    if day.nunique() != 1:
        raise ValueError("day_weight expects the header times of a single calendar day")
    fracs = (t - day).dt.total_seconds().values / 86400.0
    n = len(fracs)
    if n == 0:
        raise ValueError("day_weight got no header times")

    z = centroid_z(fracs)
    bins = np.clip((fracs * SCAN_BINS).astype(int), 0, SCAN_BINS - 1)
    counts = np.bincount(bins, minlength=SCAN_BINS)
    scan = scan_statistic(counts)

    m = len(scan_null)
    p_scan = float(np.clip(1.0 - np.searchsorted(scan_null, scan) / m, 1.0 / m, 1.0))
    p_centroid = float(2.0 * norm.sf(abs(z)))
    p_day = float(1.0 - chi2.cdf(-2.0 * (np.log(p_scan) + np.log(p_centroid)), 4))
    lr = float(np.clip(P_THRESHOLD / np.clip(p_day, 1e-6, 1.0), 1.0, LR_CAP))
    pi = prior * lr / (1.0 - prior + prior * lr)
    return float(pi), n, z, scan


def q_of(pi: float, q_quiet: float) -> float:
    """Two-regime blend: q = (1 - pi) q_quiet + pi q_jump."""
    return (1.0 - pi) * q_quiet + pi * Q_JUMP


def filter_step(
    h_prev: float, P_prev: float, y: float, q: float
) -> Tuple[float, float]:
    """One day of the one-sided recursion from a stored state.

    h_prev, P_prev : filtered log level and its variance after the previous day.
    y              : the new day's raw log hashrate estimate.
    q              : the new day's state variance (q_of of the day's weight).

    Returns the new day's filtered log level and its variance.
    """
    P_pred = P_prev + q
    S = P_pred + R1
    K = P_pred / S
    h = h_prev + K * (y - h_prev)
    P = (1.0 - K) * P_pred
    return float(h), float(P)


def rts_tail(
    h_filt: np.ndarray, P_filt: np.ndarray, q_path: np.ndarray
) -> np.ndarray:
    """Two-sided path over a tail window, from stored one-sided moments.

    h_filt, P_filt : filtered log levels and variances over the window.
    q_path         : daily state variances over the window; the one-step
                     prediction variance ahead of day t is
                     P_filt[t-1] + q_path[t].

    The recursion (paper, engine): h_s[t] = h_f[t] + J_t (h_s[t+1] - h_p[t+1])
    with J_t = P_f[t] / P_p[t+1], started from h_s[T] = h_f[T] at the window's
    last day — the identical terminal condition as the full-sample smoother,
    so the window values equal the full-sample two-sided values.
    """
    h_filt = np.asarray(h_filt, dtype=float)
    P_filt = np.asarray(P_filt, dtype=float)
    q_path = np.asarray(q_path, dtype=float)
    n = len(h_filt)
    if not (len(P_filt) == len(q_path) == n):
        raise ValueError("h_filt, P_filt and q_path must have the same length")
    h_smooth = h_filt.copy()
    for t in range(n - 2, -1, -1):
        P_pred_next = P_filt[t] + q_path[t + 1]
        J = P_filt[t] / P_pred_next
        h_pred_next = h_filt[t]
        h_smooth[t] = h_filt[t] + J * (h_smooth[t + 1] - h_pred_next)
    return h_smooth


def poisson_loglik_next_day(
    h_level: np.ndarray, n_next: np.ndarray, D_next: np.ndarray
) -> float:
    """Mean Poisson log-likelihood of the next day's block count.

    h_level : estimated hashrate levels in H/s (one per day, aligned so that
              element t predicts day t+1).
    n_next  : the next day's block count.
    D_next  : the next day's mean difficulty.

    A rate H against difficulty D expects H x 86400 / (D x 2^32) blocks a day.
    """
    lam = np.asarray(h_level, dtype=float) * 86400.0 / (np.asarray(D_next, dtype=float) * 2.0**32)
    n = np.asarray(n_next, dtype=float)
    from scipy.special import gammaln

    ll = n * np.log(lam) - lam - gammaln(n + 1.0)
    return float(ll.mean())
