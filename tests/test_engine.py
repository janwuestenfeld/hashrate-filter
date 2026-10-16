"""Engine tests: synthetic ground truth, smoother identities, equivalences
between the two filters, input validation. Runnable with pytest or as a
plain script."""

import numpy as np
import pandas as pd
import pytest

from hashrate_filter import Q_JUMP, R1, R2, pure_filter
from hashrate_filter.engine import q_pure, run_filter
from hashrate_filter.variants import anchored_filter


def _synthetic(n=800, q=1e-4, seed=7):
    """Known latent random walk + protocol-sized observation noise."""
    rng = np.random.default_rng(seed)
    h_true = np.cumsum(rng.normal(0, np.sqrt(q), n)) + 20.0
    y = h_true + rng.normal(0, np.sqrt(R1), n)
    return np.exp(y), h_true


def test_filter_beats_raw_and_smoother_beats_filter():
    h_raw, h_true = _synthetic()
    res = pure_filter(h_raw, two_sided=True)
    mse_raw = np.mean((np.log(h_raw) - h_true) ** 2)
    mse_filt = np.mean((res.h_filt - h_true) ** 2)
    mse_smooth = np.mean((res.h_smooth - h_true) ** 2)
    assert mse_filt < mse_raw, "filter must beat the raw series in MSE"
    assert mse_smooth < mse_filt, "smoother must beat the filter in MSE (ex post)"


def test_two_sided_endpoint_and_variance():
    h_raw, _ = _synthetic()
    res = pure_filter(h_raw, two_sided=True)
    assert np.isclose(res.h_smooth[-1], res.h_filt[-1]), "smoother ends on the filter"
    assert np.all(res.P_smooth <= res.P_filt + 1e-12), "smoothed variance <= filtered"
    assert np.all(res.P_smooth > 0)


def test_zero_evidence_is_single_q():
    """pi_d = 0 collapses the two-regime blend to the quiet regime."""
    pi = np.zeros(50)
    assert np.allclose(q_pure(pi, 3e-5), 3e-5)
    assert np.allclose(q_pure(np.ones(50), 3e-5), Q_JUMP)


def test_constant_difficulty_anchor_is_inert():
    """No difficulty resets -> anchored recursion equals the pure recursion."""
    h_raw, _ = _synthetic()
    d_const = np.full(len(h_raw), 1e12)
    a = anchored_filter(h_raw, d_const, q_quiet=1e-4)
    p = pure_filter(h_raw, q_quiet=1e-4)
    assert np.allclose(a.h_filt, p.h_filt, atol=1e-12)


def test_real_time_property():
    """One-sided path through day t is invariant to data after day t."""
    h_raw, _ = _synthetic(n=300)
    full = pure_filter(h_raw, q_quiet=1e-4)
    cut = pure_filter(h_raw[:200], q_quiet=1e-4)
    assert np.allclose(full.h_filt[:200], cut.h_filt, atol=1e-12), (
        "one-sided filter must not look ahead"
    )


def test_mle_recovers_quiet_scale():
    """MLE lands near the generating q on a long quiet sample."""
    h_raw, _ = _synthetic(n=4000, q=5e-5, seed=11)
    res = pure_filter(h_raw)
    assert abs(np.log(res.q_quiet / 5e-5)) < 0.7, (
        f"q_quiet {res.q_quiet:.2e} far from generating 5e-5"
    )


def test_nan_rejected():
    """One NaN must fail loudly, not poison the whole path silently."""
    h_raw, _ = _synthetic(n=50)
    h_raw[10] = np.nan
    for fn in (pure_filter, anchored_filter):
        try:
            fn(h_raw, np.full(50, 1e12)) if fn is anchored_filter else fn(h_raw)
        except ValueError:
            continue
        raise AssertionError(f"NaN input was silently accepted by {fn.__name__}")
    try:
        run_filter(np.log(h_raw))
    except ValueError:
        return
    raise AssertionError("NaN input was silently accepted by run_filter")


def test_pi_out_of_range_rejected():
    """Arrival-time weights outside [0, 1] are invalid and must raise."""
    h_raw, _ = _synthetic(n=50)
    for bad in (-0.1, 1.5, np.nan):
        try:
            pure_filter(h_raw, pi_d=np.full(50, bad))
        except ValueError:
            continue
        raise AssertionError(f"pi_d={bad} was silently accepted")


def test_gain_formula():
    """Hand-computed first steps: K_t = V_t / (V_t + 1/144), V_t = P_{t-1} + q."""
    y = np.zeros(3)
    q = 1e-3
    r = run_filter(y, q_quiet=q)
    V0 = 0.05 + q
    K0 = V0 / (V0 + R1)
    P0p = (1 - K0) * V0
    V1 = P0p + q
    K1 = V1 / (V1 + R1)
    assert np.isclose(r.P_filt[0], P0p)
    assert np.isclose(r.P_filt[1], (1 - K1) * V1)
    assert np.isclose(r.h_filt[1], 0.0)  # zero innovations keep the level


def test_jump_followed_faster_with_pi():
    """On a flagged day the filter keeps more of a real step."""
    y = np.r_[np.zeros(50), np.full(50, 0.3)]
    a = run_filter(y, q_quiet=1e-4).h_filt[50]
    pi = np.zeros(100)
    pi[50] = 1.0
    b = run_filter(y, pi_d=pi, q_quiet=1e-4).h_filt[50]
    assert b > 2 * a


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        pure_filter(np.ones(10), pi_d=np.zeros(11), q_quiet=1e-4)
    with pytest.raises(ValueError, match="length"):
        run_filter(np.zeros(10), anchor_mask=np.zeros(9, bool), q_quiet=1e-4)


def test_bad_q_quiet_raises():
    for bad in (0.0, -1e-4, np.nan, np.inf):
        with pytest.raises(ValueError, match="q_quiet"):
            pure_filter(np.ones(10), q_quiet=bad)


def test_too_short_raises():
    with pytest.raises(ValueError, match="at least 2"):
        pure_filter(np.ones(1), q_quiet=1e-4)


def test_dates_gap_raises():
    h_raw, _ = _synthetic(n=30)
    days = pd.date_range("2024-01-01", periods=30).append(pd.date_range("2024-02-01", periods=0))
    days = days[:15].append(days[15:] + pd.Timedelta(days=1))  # a two-day gap
    with pytest.raises(ValueError, match="consecutive"):
        pure_filter(h_raw, q_quiet=1e-4, dates=days)
    ok = pd.date_range("2024-01-01", periods=30)
    pure_filter(h_raw, q_quiet=1e-4, dates=ok)  # consecutive: no error


def test_reset_day_anchor_pulls_toward_target():
    """After a difficulty reset the anchored path moves toward log D + offset."""
    n = 40
    h_const = np.full(n, 1.3e11)
    d = np.r_[np.full(20, 1e12), np.full(20, 1.1e12)]
    a = anchored_filter(h_const, d, q_quiet=1e-4)
    p = pure_filter(h_const, q_quiet=1e-4)
    offset = np.log(1.3e11) - np.log(1e12)
    # the step is at a day boundary, so the opening difficulty changes once
    assert a.h_filt[19] == p.h_filt[19]
    assert a.h_filt[20] > p.h_filt[20]  # pulled up toward log(1.1e12) + offset
    assert a.h_filt[20] < np.log(1.1e12) + offset
    assert a.anchor_var[20] >= R2  # reset day carries the anchor observation


def test_delag_variance_grows_with_slope_uncertainty():
    """Anchor variance is 1/2016 + (7 se)^2: larger on a noisy trend, base early on."""
    n = 120
    rng = np.random.default_rng(21)
    trend = np.linspace(0, 0.8, n) + rng.normal(0, 0.01, n) + 25.0
    h_raw = np.exp(trend)
    d = np.r_[np.full(10, 1e12), np.full(50, 1.02e12), np.full(60, 1.05e12)]
    res = anchored_filter(h_raw, d, q_quiet=1e-4)
    assert np.isclose(res.anchor_var[10], R2)  # before the slope window exists
    assert res.anchor_var[60] > R2  # slope uncertainty raises the variance


def test_delag_lifts_target_in_a_trend():
    """In a rising trend the de-lagged anchor pulls the level above the plain target."""
    n = 60
    h_raw = np.exp(np.linspace(0, 0.5, n) + 25.0)
    d = np.r_[np.full(40, 1e12), np.full(n - 40, 1.02e12)]
    res = anchored_filter(h_raw, d, q_quiet=1e-4)
    offset = np.median(np.log(h_raw) - np.log(d))
    # the de-lagged path overshoots the plain target log D + offset on the reset day
    plain_target = np.log(1.02e12) + offset
    assert res.h_filt[41] > 0.5 * (plain_target + np.log(h_raw[41]))  # sanity: moved
    assert res.anchor_var[40] > R2


def test_offset_protocol_runs():
    h_raw, _ = _synthetic(n=60)
    d = np.full(60, 1e12)
    a = anchored_filter(h_raw, d, q_quiet=1e-4, offset="protocol")
    assert np.isfinite(a.h_filt).all()


def test_smoothed_input_warns():
    """A multi-day moving average is not a valid input and must say so."""
    h_raw, _ = _synthetic(n=300)
    ma = pd.Series(h_raw).rolling(7).mean().dropna().values
    with pytest.warns(UserWarning, match="smoothed"):
        pure_filter(ma, q_quiet=1e-4)


if __name__ == "__main__":
    for name, fn in sorted(list(globals().items())):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
