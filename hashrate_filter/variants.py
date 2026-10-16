"""The two hashrate-filter estimators.

pure_filter      the pure filter: the day's own blocks, their number and
                 their arrival times, through the two-regime blend and the
                 one-sided Kalman recursion. The series used in the paper.

anchored_filter  the extended filter: the pure filter plus the difficulty
                 as an extra observation on reset days. Because the
                 difficulty reflects the average hashrate over the epoch
                 that just ended, it lags the current rate when hashrate
                 trends, so the observation is de-lagged by the OLS slope of
                 the last 14 filtered days projected half an epoch (7 days),
                 and its variance is 1/2016 plus the squared standard error
                 of that projection.

A change in hashrate reappears in later difficulty adjustments over up to
fourteen days, so the paper reports its results for the pure filter and
uses the extended filter to check that the two agree. Both take
two_sided=True to add the two-sided version (the RTS smoother over the
same model), which also uses later days; the paper uses it only as a
benchmark.
"""

from typing import Optional, Union
import warnings

import numpy as np

from .engine import (
    PROTOCOL_OFFSET,
    FilterResult,
    run_filter,
)


def _as_log(series, name: str) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains non-finite values (NaN or inf)")
    if np.any(x <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return np.log(x)


def pure_filter(
    hashrate,
    pi_d=None,
    q_quiet: Optional[float] = None,
    two_sided: bool = False,
    dates=None,
) -> FilterResult:
    """Pure filter: the paper's luck-corrected daily hashrate.

    Parameters
    ----------
    hashrate : array-like of float, length T
        Raw one-day network hashrate estimates,
        n_blocks x mean_difficulty x 2^32 / 86400 in H/s, one row per
        consecutive calendar day (any constant rescaling is allowed; the
        series is logged internally). Must be built from a single day's
        block count: the observation variance 1/144 does not hold for a
        series averaged over several days.
    pi_d : array-like or None
        Daily arrival-time weights from evidence.daily_evidence(times),
        aligned to the same dates. Zeros if None: the quiet-day variance on
        every day.
    q_quiet : float or None
        Quiet-day state variance; fitted by MLE if None (the one fitted
        parameter). The paper's value is 7.81e-4.
    two_sided : bool
        Also return the two-sided (RTS-smoothed) path.
    dates : array-like of datetime-like or None
        Calendar dates of the rows; if given, must be consecutive days.

    Returns
    -------
    FilterResult with h_filt (and h_smooth if two_sided) in log units;
    exponentiate for hashrate units.
    """
    y = _as_log(hashrate, "hashrate")
    if pi_d is not None:
        pi_d = np.asarray(pi_d, dtype=float)
    if len(y) > 20:
        g = np.diff(y)
        r1 = np.corrcoef(g[:-1], g[1:])[0, 1] if g.std() > 0 else -1.0
        if r1 > -0.3:
            warnings.warn(
                f"lag-1 autocorrelation of daily growth is {r1:.2f}; a raw "
                "count-based series has about -0.5. The input looks smoothed, "
                "so the observation variance 1/144 does not apply to it."
            )
    return run_filter(y, pi_d=pi_d, q_quiet=q_quiet, two_sided=two_sided, dates=dates)


def anchored_filter(
    hashrate,
    difficulty,
    pi_d=None,
    q_quiet: Optional[float] = None,
    offset: Union[float, str, None] = None,
    two_sided: bool = False,
    dates=None,
) -> FilterResult:
    """Extended filter: the pure filter plus the difficulty anchor.

    Parameters
    ----------
    hashrate : array-like of float, length T
        Raw one-day network hashrate estimates (see pure_filter).
    difficulty : array-like of float, length T
        Daily network difficulty, aligned to hashrate, taken at the start of
        each day (the day's first block). A reset then shows up once, on the
        first day that runs at the new difficulty. On reset days the level
        change carries the anchor observation log D_t + offset + 7 s_t,
        where s_t is the OLS slope of the last 14 filtered days, with
        variance 1/2016 + (7 se_t)^2.
    pi_d : array-like or None
        Daily arrival-time weights, as in pure_filter.
    q_quiet : float or None
        Quiet-day state variance; fitted by MLE if None.
    offset : float, 'protocol', or None
        Anchor offset log H - log D. None (default) uses the sample median.
        'protocol' uses log(2^32 / 600), the value at exactly ten minutes
        per block.
    two_sided : bool
        Also return the two-sided (RTS-smoothed) path.
    dates : array-like of datetime-like or None
        Calendar dates of the rows; if given, must be consecutive days.

    Returns
    -------
    FilterResult with h_filt (and h_smooth if two_sided) in log units.
    """
    y = _as_log(hashrate, "hashrate")
    log_d = _as_log(difficulty, "difficulty")
    n = len(y)
    if len(log_d) != n:
        raise ValueError("hashrate and difficulty must have the same length")
    if offset is None:
        offset = float(np.median(y - log_d))
    elif isinstance(offset, str):
        if offset != "protocol":
            raise ValueError("offset must be a number, 'protocol', or None")
        offset = PROTOCOL_OFFSET
    offset = float(offset)

    adj = np.zeros(n, dtype=bool)
    adj[1:] = np.abs(np.diff(log_d)) > 1e-9
    anchor_target = log_d + offset
    pi_d = np.zeros(n) if pi_d is None else np.asarray(pi_d, dtype=float)

    return run_filter(
        y, pi_d=pi_d, q_quiet=q_quiet,
        anchor_mask=adj, anchor_target=anchor_target,
        two_sided=two_sided, dates=dates,
    )


# The paper's name for the anchored filter.
extended_filter = anchored_filter
