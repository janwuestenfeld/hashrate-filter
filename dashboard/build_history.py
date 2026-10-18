"""One-time builder of the dashboard's stored history and evidence reference.

Runs once, locally, from the paper's block panel (one row per block with
time, median_time, id, difficulty, as in examples/reproduce_paper_series.py).
It writes three files into dashboard/data/, committed with the dashboard:

  * history.csv              one row per day, 2011-01-01 to the sample end:
                             date, n_blocks, difficulty, raw, pure,
                             pure_two_sided, ma3, weight, p_filt. The
                             published columns (raw, pure, pure_two_sided)
                             are the paper's series verbatim; the weight and
                             the filtered variance are rebuilt from the
                             paper's inputs and drive the daily update.
  * evidence_reference.npz   the stored scan null (50,000 replications,
                             seed 123, the settings of the paper's series),
                             so a new day's weight is computed against a
                             fixed reference.
  * meta.json                the stored prior, the fitted quiet-day
                             variance, provenance and the verification
                             numbers below.

Checks reported (and written to meta.json):

  * rebuilt one-sided series vs the published reference
    (examples/data/reference_pure_filter.csv): the README states a median
    absolute difference of about 5e-5 log units, the largest gap on the
    incomplete final day.
  * rebuilt two-sided series and raw estimate vs the published example
    output (examples/data/example_output.csv).
  * the one-day weight step of dashboard/daily_step.py against the
    package's batch evidence (must match to machine precision: same
    header times, same stored null, same prior).
  * the one-day filter step against the batch recursion over the tail
    window (machine precision).
  * the tail re-smoothing against the batch two-sided pass over the tail
    window (machine precision).

Usage:
    python dashboard/build_history.py BLOCKS_PANEL.parquet

The panel path stays local; it is an argument, never written to any file.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # the package, uninstalled

from hashrate_filter import daily_evidence, pure_filter  # noqa: E402
from hashrate_filter.evidence import _MIN_N, _mc_null  # noqa: E402

import daily_step  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "data"

START, END = "2011-01-01", "2026-07-15"  # the paper's sample window
REPS, SEED = 50_000, 123  # the settings of the paper's series
Q_QUIET_PAPER = 7.81e-4  # the paper's reported quiet-day variance

# Which published one-sided series the stored history carries. The two
# shipped files differ slightly (input snapshots; see examples/data/README):
# reference_pure_filter.csv is the series the paper's estimations ran on,
# example_output.csv is the series behind the repository's figure. Only the
# example output is fully self-consistent (raw, one-sided and two-sided from
# one build on the current panel), and the live update continues from the
# stored state, so the stored history carries the figure series; equality to
# the paper's estimation series is verified to the README's stated
# deviation (median about 5e-5 log units, largest gap on the incomplete
# final day). Set to "reference" to store the estimation series verbatim.
PURE_SOURCE = "example"


def daily_master(panel_path: str) -> pd.DataFrame:
    """Raw daily hashrate from a block panel (median-time days).

    Same construction as examples/reproduce_paper_series.py:
    H = n_blocks x mean_difficulty x 2^32 / 86400.
    """
    p = pd.read_parquet(panel_path, columns=["time", "median_time", "id", "difficulty"])
    p["Date"] = pd.to_datetime(p["median_time"]).dt.normalize()
    p = p.sort_values("id")
    d = p.groupby("Date").agg(
        n=("id", "size"), D=("difficulty", "mean"), D_open=("difficulty", "first")
    )
    d["H"] = d["n"] * d["D"] * 2**32 / 86400
    return d[["H", "D", "n"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("panel", help="block panel parquet (time, median_time, id, difficulty)")
    args = ap.parse_args()

    print("loading panel ...")
    times_all = pd.read_parquet(args.panel, columns=["time"])["time"]
    d0 = daily_master(args.panel)
    d = d0.loc[START:END]
    full = pd.date_range(d.index[0], d.index[-1], freq="D")
    missing = full.difference(d.index)
    if len(missing):
        raise ValueError(f"daily series has {len(missing)} missing days, e.g. {missing[:3]}")
    print(f"daily series: {len(d)} days, {d.index[0].date()} -> {d.index[-1].date()}")

    # --- batch evidence on all header times (the paper's construction) ---
    print(f"batch arrival-time evidence (reps={REPS}, seed={SEED}) ...")
    ev = daily_evidence(times_all, reps=REPS, seed=SEED)
    prior = float(ev.attrs["prior"])
    pi_d = pd.Series(d.index).map(ev["pi"]).fillna(0.0).values
    print(f"  stored prior: {prior:.6f}")

    # The identical scan null the batch call drew internally: same day-count
    # sample, same generator, same seed (the generator is consumed only by
    # the null draw inside daily_evidence). The count array must be ordered
    # by calendar day, exactly as the batch's per-day stats are.
    day_n = (
        pd.to_datetime(times_all)
        .dt.floor("D")
        .value_counts()
        .sort_index()
        .values.astype(float)
    )
    rng = np.random.default_rng(SEED)
    scan_null = _mc_null(day_n[day_n > _MIN_N], REPS, rng)
    # cross-check: the batch p_scan of every day against this null
    p_scan_check = np.clip(
        1.0 - np.searchsorted(scan_null, ev["scan"].values) / len(scan_null),
        1.0 / len(scan_null),
        1.0,
    )
    null_match = float(np.abs(p_scan_check - ev["p_scan"].values).max())
    print(f"  scan null re-drawn; max |p_scan diff| vs batch: {null_match:.2e}")

    # --- batch filter (the paper's series), one-sided and two-sided ---
    print("batch pure filter ...")
    res = pure_filter(d["H"].values, pi_d=pi_d, two_sided=True, dates=d.index)
    q_quiet = float(res.q_quiet)
    print(f"  fitted q_quiet {q_quiet:.6e} (paper: {Q_QUIET_PAPER:.2e})")

    # --- published series (verbatim) ---
    ref = pd.read_csv(REPO / "examples" / "data" / "reference_pure_filter.csv", parse_dates=["date"])
    ex = pd.read_csv(REPO / "examples" / "data" / "example_output.csv", parse_dates=["date"])
    ref = ref.set_index("date")
    ex = ex.set_index("date")
    if not ref.index.equals(d.index) or not ex.index.equals(d.index):
        raise ValueError("published series and panel days do not align")

    h_ref = ref["log_h"].values
    raw_pub = ex["raw"].values
    smooth_pub = np.log(ex["pure_two_sided"].values)
    pure_pub = np.log(ex["pure"].values)

    def stats(diff):
        diff = np.abs(np.asarray(diff, dtype=float))
        i = int(np.argmax(diff))
        return {
            "median": float(np.median(diff)),
            "mean": float(diff.mean()),
            "max": float(diff.max()),
            "max_date": str(d.index[i].date()),
        }

    chk_ref = stats(res.h_filt - h_ref)
    chk_pure_ex = stats(res.h_filt - pure_pub)
    chk_smooth = stats(res.h_smooth - smooth_pub)
    chk_raw = stats(np.log(d["H"].values) - np.log(raw_pub))
    print(f"  rebuilt one-sided vs paper reference:     {chk_ref}")
    print(f"  rebuilt one-sided vs published example:   {chk_pure_ex}")
    print(f"  rebuilt two-sided vs published example:   {chk_smooth}")
    print(f"  rebuilt raw vs published example:         {chk_raw}")

    # the stored one-sided levels must equal the paper's estimation series
    # to the README's stated deviation, with the largest gap on the
    # incomplete final day
    chk_stored = stats(pure_pub - h_ref) if PURE_SOURCE == "example" else stats(h_ref - h_ref)
    print(f"  stored one-sided vs paper reference:      {chk_stored} (source: {PURE_SOURCE})")
    if not (chk_stored["median"] < 1e-4 and chk_stored["max"] < 2e-2 and chk_stored["max_date"] == END):
        raise ValueError(
            "stored one-sided series deviates from the paper's reference beyond the "
            f"stated deviation: {chk_stored}"
        )
    h_store = pure_pub if PURE_SOURCE == "example" else h_ref

    # --- one-day steps against the batch run (machine precision) ---
    print("verifying one-day steps ...")
    # 1. weights: per-day step on each day's header times vs the batch frame
    tt = pd.to_datetime(times_all)
    if tt.dt.tz is not None:
        tt = tt.dt.tz_convert("UTC").dt.tz_localize(None)
    hdr_day = tt.dt.floor("D")
    w_diffs = []
    # group header times once, then one call per sample day
    grouped = pd.Series(tt.values, index=hdr_day.values).groupby(level=0)
    hdr_by_day = {day: pd.Series(g.values) for day, g in grouped}
    for day in d.index:
        g = hdr_by_day.get(day)
        if g is None or day not in ev.index:
            continue
        pi_t, _, _, _ = daily_step.day_weight(g, scan_null, prior)
        w_diffs.append(pi_t - ev.loc[day, "pi"])
    w_max = float(np.abs(w_diffs).max())
    print(f"  one-day weight vs batch, over {len(d)} days: max |diff| {w_max:.2e}")

    # 2. filter step: replay the tail window day by day from the state ahead
    #    of it, compare against the batch filtered levels and variances
    W = daily_step.TAIL_WINDOW
    h_b, P_b, q_b = res.h_filt, res.P_filt, res.q_path
    h_s, P_s = h_b[-W - 1], P_b[-W - 1]
    f_max = 0.0
    for t in range(len(d) - W, len(d)):
        h_s, P_s = daily_step.filter_step(h_s, P_s, np.log(d["H"].values[t]), q_b[t])
        f_max = max(f_max, abs(h_s - h_b[t]), abs(P_s - P_b[t]))
    print(f"  one-day filter step vs batch, tail {W} days: max |diff| {f_max:.2e}")

    # 3. tail re-smoothing vs the batch two-sided pass over the window
    h_tail = daily_step.rts_tail(h_b[-W:], P_b[-W:], q_b[-W:])
    s_max = float(np.abs(h_tail - res.h_smooth[-W:]).max())
    print(f"  tail re-smoothing vs batch, tail {W} days: max |diff| {s_max:.2e}")

    # --- write the stored history ---
    OUT.mkdir(exist_ok=True)
    ma3 = ex["raw"].rolling(3, min_periods=1).mean().values
    hist = pd.DataFrame(
        {
            "date": d.index.strftime("%Y-%m-%d"),
            "n_blocks": d["n"].values.astype(int),
            "difficulty": d["D"].values,
            "raw": raw_pub,
            "pure": np.exp(h_store),
            "pure_two_sided": np.exp(smooth_pub),
            "ma3": ma3,
            "weight": pi_d,
            "p_filt": res.P_filt,
        }
    )
    hist_path = OUT / "history.csv"
    hist.to_csv(hist_path, index=False, float_format="%.17g")
    print(f"wrote {hist_path} ({len(hist)} days)")

    npz_path = OUT / "evidence_reference.npz"
    np.savez_compressed(
        npz_path,
        scan_null=scan_null,
        day_counts=day_n[day_n > _MIN_N],
    )
    print(f"wrote {npz_path} (null of {len(scan_null)} draws)")

    meta = {
        "schema": 1,
        "sample_start": START,
        "sample_end": END,
        "days": int(len(d)),
        "prior": prior,
        "q_quiet": q_quiet,
        "q_quiet_paper": Q_QUIET_PAPER,
        "reps": REPS,
        "seed": SEED,
        "generated_by": "dashboard/build_history.py",
        "generated_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "pure_source": PURE_SOURCE,
        "verification": {
            "stored_pure_vs_reference": chk_stored,
            "rebuilt_pure_vs_reference": chk_ref,
            "rebuilt_pure_vs_example": chk_pure_ex,
            "rebuilt_two_sided_vs_example": chk_smooth,
            "rebuilt_raw_vs_example": chk_raw,
            "scan_null_p_scan_max_diff": null_match,
            "one_day_weight_max_diff": w_max,
            "one_day_filter_step_max_diff": f_max,
            "tail_resmooth_max_diff": s_max,
        },
    }
    meta_path = OUT / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {meta_path}")

    ok = (
        chk_ref["median"] < 1e-4
        and chk_pure_ex["max"] < 1e-9
        and w_max < 1e-9
        and f_max < 1e-9
        and s_max < 1e-9
        and null_match < 1e-9
    )
    print("VERDICT:", "OK" if ok else "CHECK THE NUMBERS ABOVE")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
