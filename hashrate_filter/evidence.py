"""Arrival-time evidence: the daily weight pi_t of the hashrate filter.

Under a constant within-day rate, block arrival times are uniform on the day
(the conditional-uniformity property of the Poisson process), so departures
from uniformity are evidence of a real intraday move. Two statistics per day:

  * the centroid z_t = (m_t - 1/2) sqrt(12 n_t), with m_t the mean arrival
    time as a fraction of the day and n_t the block count; mean 0 and
    variance 1 exactly under the null, standard normal to numerical accuracy
    at daily block counts;
  * an hourly scan statistic (the maximum two-sample log-likelihood ratio
    over split points of the cumulative intraday counts), whose null
    distribution is simulated by Monte Carlo at the observed daily block
    counts.

The two p-values are combined by Fisher's method. The two statistics respond
to the same arrivals and are positively dependent, so the combined value is
used as a monotone score rather than an exact p-value; the prior in the
p-value-to-weight map is fitted to its empirical exceedance rate, which
absorbs the dependence.

The Monte Carlo null and the prior are drawn from the sample's own day-count
distribution, so appending data re-draws them and moves pi_t slightly on
every past day. 50,000 replications with seed 123 are the settings of the
series used in the paper. For real-time use, pass a fixed prior (the fitted
value is returned in the frame's attrs) so that past days move only through
the re-drawn null.
"""

from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm

_NCOLS = 24  # intraday bins for the scan statistic (chosen)
_MC_FLOOR = 1e-4  # prior floor in the p-value-to-weight map (chosen)
_P_THRESHOLD = 1e-3  # threshold of the exceedance-rate prior (chosen)
_LR_CAP = 1000.0  # cap of the likelihood ratio in the weight map (chosen)
_MIN_N = 20  # minimum block count for a day to enter the null sample (chosen)


def _as_utc_naive(times) -> pd.Series:
    """Block timestamps as timezone-naive UTC datetimes.

    Timezone-aware input is converted to UTC and made naive. Numeric input
    (for example Unix seconds, which pd.to_datetime would read as
    nanoseconds) is rejected, because silently misdated blocks switch the
    arrival-time weight off.
    """
    s = pd.Series(times)
    if pd.api.types.is_numeric_dtype(s):
        raise TypeError(
            "times must be datetimes, not numbers; for Unix seconds pass "
            "pd.to_datetime(times, unit='s') first"
        )
    s = pd.to_datetime(s)
    if s.dt.tz is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s


def _scan_max(cum: np.ndarray) -> np.ndarray:
    """Max two-sample log-LR over split points of cumulative bin counts."""
    N = cum[:, -1:]
    lr = np.zeros(len(cum))
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(1, cum.shape[1]):
            pk = k / cum.shape[1]
            xx = cum[:, k - 1]
            term = xx * np.log(np.clip(xx / (N[:, 0] * pk), 1e-12, None)) + (
                N[:, 0] - xx
            ) * np.log(np.clip((N[:, 0] - xx) / (N[:, 0] * (1 - pk)), 1e-12, None))
            lr = np.maximum(lr, 2 * term)
    return lr


def _period_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-day count n, centroid z, and scan statistic."""
    grp = df.groupby("date")["frac"]
    n = grp.size()
    z = (grp.mean() - 0.5) * np.sqrt(12 * n)
    hr = (df["frac"] * _NCOLS).astype(int).clip(0, _NCOLS - 1)
    tab = (
        df.groupby(["date", hr])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(_NCOLS), fill_value=0)
    )
    scan = _scan_max(tab.cumsum(axis=1).values.astype(float))
    return pd.DataFrame({"n": n, "z": z, "scan": pd.Series(scan, index=tab.index)})


def _mc_null(ns: np.ndarray, reps: int, rng: np.random.Generator) -> np.ndarray:
    """Monte Carlo null for the scan statistic: uniform arrivals, empirical n."""
    ns = rng.choice(ns, reps)
    counts = np.array(
        [rng.multinomial(k, [1 / _NCOLS] * _NCOLS) for k in ns]
    ).astype(float)
    return np.sort(_scan_max(counts.cumsum(axis=1)))


def _to_pi(pvals: np.ndarray, prior: Optional[float] = None) -> tuple:
    """Monotone p-value-to-weight map.

    The prior probability of a real shift is the share of days whose p-value
    falls below 0.001, net of the 0.1 percent expected by chance, floored at
    1e-4. Pass prior=float to freeze it (real-time use). Returns (pi, prior).
    """
    x = np.asarray(pvals, dtype=float)
    pri = max((x < _P_THRESHOLD).mean() - _P_THRESHOLD, _MC_FLOOR) if prior is None else float(prior)
    if not (0.0 < pri < 1.0):
        raise ValueError("prior must lie in (0, 1)")
    lr = np.clip(_P_THRESHOLD / np.clip(x, 1e-6, 1), 1, _LR_CAP)
    return pri * lr / (1 - pri + pri * lr), pri


def daily_evidence(
    times: pd.Series,
    reps: int = 50_000,
    seed: int = 123,
    prior: Optional[float] = None,
) -> pd.DataFrame:
    """Daily arrival-time weights from block header timestamps.

    Parameters
    ----------
    times : pd.Series
        Block header timestamps, UTC. Days are calendar days of these
        timestamps; timezone-aware input is converted to naive UTC. Numeric
        input (Unix seconds) is rejected: convert with pd.to_datetime(times,
        unit='s') first.
    reps : int
        Monte Carlo replications for the scan statistic's null. 50,000 with
        seed 123 are the settings of the series used in the paper. The null
        is drawn from the sample's own day counts, so a different input
        snapshot moves pi slightly on every day.
    seed : int
        RNG seed for the null.
    prior : float or None
        Prior probability of a real shift. If None, fitted to the sample's
        exceedance rate at p < 0.001. The fitted value is returned in
        attrs['prior']; pass it back to freeze the map for real-time use.

    Returns
    -------
    pd.DataFrame indexed by UTC calendar date with columns n, z, scan,
    p_scan, p_day, pi (the daily arrival-time weight). The fitted prior is
    in attrs['prior'].
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"time": _as_utc_naive(times)})
    df = df.sort_values("time").reset_index(drop=True)
    df["date"] = df["time"].dt.floor("D")
    df["frac"] = (df["time"] - df["date"]).dt.total_seconds() / 86400.0

    # A day seen only in part has its arrivals compressed into [0, k/24),
    # which the uniformity test reads as evidence. Warn rather than flag.
    span = df.groupby("date")["time"].agg(["min", "max"])
    first, last = span.index[0], span.index[-1]
    if span.loc[first, "min"] > first + pd.Timedelta(hours=2):
        warnings.warn(
            f"the first day ({first.date()}) starts {span.loc[first, 'min'].time()} UTC; "
            "a partial first day can be flagged spuriously"
        )
    if span.loc[last, "max"] < last + pd.Timedelta(hours=22):
        warnings.warn(
            f"the last day ({last.date()}) ends {span.loc[last, 'max'].time()} UTC; "
            "a partial last day can be flagged spuriously"
        )

    stats = _period_stats(df)
    null = _mc_null(stats["n"].values[stats["n"] > _MIN_N], reps, rng)
    stats = stats.copy()
    stats["p_scan"] = np.clip(
        1 - np.searchsorted(null, stats["scan"]) / len(null), 1 / len(null), 1
    )
    stats["p_day"] = 1 - chi2.cdf(
        -2 * (np.log(stats["p_scan"]) + np.log(2 * norm.sf(np.abs(stats["z"])))),
        4,
    )
    pi, pri = _to_pi(stats["p_day"].values, prior=prior)
    stats["pi"] = pi
    stats.attrs["prior"] = pri
    return stats
