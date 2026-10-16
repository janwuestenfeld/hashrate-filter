"""Evidence tests: the arrival-time weight, its null behaviour, its failure
modes, and its reproducibility. Runnable with pytest or as a plain script."""

import warnings

import numpy as np
import pandas as pd

from hashrate_filter import daily_evidence
from hashrate_filter.evidence import _to_pi


def _uniform_day_times(n_days=60, seed=0, start="2024-01-01"):
    """Blocks with uniform arrival times on each of n_days consecutive days."""
    rng = np.random.default_rng(seed)
    t = []
    for d in pd.date_range(start, periods=n_days):
        k = rng.poisson(144)
        t += list(d + pd.to_timedelta(np.sort(rng.random(k)) * 86400, "s"))
    return pd.Series(t)


def test_centroid_null_moments():
    """Under uniform arrivals the centroid is mean 0, variance 1."""
    ev = daily_evidence(_uniform_day_times(400, seed=3), reps=5_000, seed=1)
    assert abs(ev["z"].mean()) < 0.15
    assert abs(ev["z"].std() - 1) < 0.1


def test_weight_stays_small_on_quiet_days():
    """Uniform days should not receive a meaningful weight."""
    ev = daily_evidence(_uniform_day_times(120, seed=4), reps=5_000, seed=1)
    assert ev["pi"].max() < 0.5


def test_flags_intraday_shift():
    """A day whose block rate triples in the last eight hours is flagged."""
    quiet = _uniform_day_times(60, seed=5, start="2024-01-01")
    day = pd.Timestamp("2024-03-01")
    rng = np.random.default_rng(9)
    t1 = day + pd.to_timedelta(np.sort(rng.random(72)) * 16 * 3600, "s")
    t2 = day + pd.to_timedelta(16 * 3600 + np.sort(rng.random(144)) * 8 * 3600, "s")
    times = pd.concat([quiet, pd.Series(list(t1) + list(t2))])
    ev = daily_evidence(times, reps=5_000, seed=1)
    assert ev.loc[day, "pi"] > 0.5
    assert ev["pi"].drop(day).max() < 0.5


def test_deterministic_under_seed():
    t = _uniform_day_times(30, seed=6)
    a = daily_evidence(t, reps=2_000, seed=5)
    b = daily_evidence(t, reps=2_000, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_pi_map_monotone_and_bounded():
    pi, pri = _to_pi(np.logspace(-8, 0, 200))
    assert np.all(np.diff(pi) <= 1e-15)
    assert pi.min() >= 0.0 and pi.max() <= 1.0
    assert 0.0 < pri < 1.0


def test_tz_aware_times_match_naive_utc():
    """Timezone-aware UTC timestamps are converted, not silently dropped."""
    t = _uniform_day_times(40, seed=7)
    aware = t.dt.tz_localize("UTC")
    a = daily_evidence(t, reps=2_000, seed=11)
    b = daily_evidence(aware, reps=2_000, seed=11)
    assert b.index.tz is None
    pd.testing.assert_frame_equal(a, b)


def test_numeric_times_rejected():
    """Unix seconds must not be read as nanoseconds."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            daily_evidence(pd.Series([1_700_000_000, 1_700_000_600, 1_700_001_200]))
        except TypeError:
            return
    raise AssertionError("numeric timestamps were silently accepted")


def test_partial_last_day_warns():
    """A day exported at noon must not quietly pass as complete."""
    t = _uniform_day_times(30, seed=8)
    last_day = pd.to_datetime(t.max()).normalize()
    t = t[pd.to_datetime(t) < last_day + pd.Timedelta(hours=12)]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        daily_evidence(t, reps=1_000, seed=1)
    assert any("partial last day" in str(x.message) for x in w)


def test_prior_is_returned_and_fixed():
    t = _uniform_day_times(60, seed=10)
    fitted = daily_evidence(t, reps=2_000, seed=2)
    assert 0.0 < fitted.attrs["prior"] < 1.0
    fixed = daily_evidence(t, reps=2_000, seed=2, prior=0.3)
    assert fixed.attrs["prior"] == 0.3
    assert not np.allclose(fixed["pi"].values, fitted["pi"].values)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
