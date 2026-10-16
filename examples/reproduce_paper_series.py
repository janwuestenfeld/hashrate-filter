"""Reproduce the paper's filtered hashrate series from raw block data.

Pipeline: block panel -> raw daily hashrate
    H_day = n_blocks x mean_difficulty x 2^32 / 86400   (median-time dates)
-> daily arrival-time weight -> pure filter and extended filter (difficulty
anchor, de-lagged), each one-sided and two-sided -> comparison against the
reference series shipped in examples/data/.

Usage:
    python reproduce_paper_series.py BLOCKS_PANEL.parquet [--start 2011-01-01]
        [--end 2026-07-15] [--no-compare]
        [--reference-pure PATH] [--reference-extended PATH]

The panel must carry one row per block with columns: time (header
timestamp), median_time, id, difficulty. Dates are assigned on median_time
(the manipulation-robust timestamp); the arrival-time weight uses header
times within header-time days.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from hashrate_filter import daily_evidence
from hashrate_filter.variants import anchored_filter, pure_filter

START, END = "2011-01-01", "2026-07-15"  # paper sample window
EXT_END = "2026-07-11"  # the paper's extended-filter series ends here
DATA = Path(__file__).parent / "data"


def daily_master(panel_path: str) -> pd.DataFrame:
    """Raw daily hashrate and difficulty from a block panel.

    H uses the day's mean difficulty (the work actually done); the anchor
    uses the day's opening difficulty (the day's first block), so a reset
    shows up once, on the first day that runs at the new difficulty.
    """
    p = pd.read_parquet(
        panel_path, columns=["time", "median_time", "id", "difficulty"]
    )
    p["Date"] = pd.to_datetime(p["median_time"]).dt.normalize()
    p = p.sort_values("id")
    d = p.groupby("Date").agg(
        n=("id", "size"), D=("difficulty", "mean"), D_open=("difficulty", "first")
    )
    d["H"] = d["n"] * d["D"] * 2**32 / 86400
    last = d.index[-1]
    last_block = pd.to_datetime(p["time"]).max()
    if last_block < last + pd.Timedelta(days=1, hours=-2):
        warnings.warn(
            f"the final day ({last.date()}) ends at {last_block.time()} UTC; "
            "blocks with a median time on that day can still be missing, so the "
            "last day's estimate is the least reliable of the series"
        )
    return d[["H", "D", "D_open", "n"]]


def verdict(diff: np.ndarray) -> str:
    if diff.max() < 1e-6:
        return "REPRODUCES to machine precision"
    if diff.max() < 0.05:
        return (
            "CLOSE: the arrival-time null and prior are re-drawn on the input "
            "snapshot, so weights move slightly when the panel changes"
        )
    return "MISMATCH, investigate"


def compare(new: pd.Series, ref_path: str, ref_col: str, label: str) -> None:
    ref = pd.read_csv(ref_path)
    ref["date"] = pd.to_datetime(ref["date"])
    m = pd.DataFrame({"date": new.index, "new": new.values}).merge(ref, on="date")
    diff = (m["new"] - m[ref_col]).abs()
    print(
        f"  vs {Path(ref_path).name}: shared days {len(m)}/{len(ref)}  "
        f"median|diff| {diff.median():.3e}  mean|diff| {diff.mean():.3e}  "
        f"max|diff| {diff.max():.3e} (on {m.loc[diff.idxmax(), 'date'].date()})"
    )
    print(f"  {verdict(diff.values)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("panel", help="blocks panel parquet (time, median_time, id, difficulty)")
    ap.add_argument("--start", default=START)
    ap.add_argument("--end", default=END)
    ap.add_argument("--no-compare", action="store_true",
                    help="run without comparing to the reference series")
    ap.add_argument("--reference-pure", default=str(DATA / "reference_pure_filter.csv"),
                    help="reference pure-filter CSV (date, log_h); ships with the repo")
    ap.add_argument("--reference-extended", default=str(DATA / "reference_extended_filter.csv"),
                    help="reference extended-filter CSV (date, log_h); ships with the repo")
    args = ap.parse_args()

    d0 = daily_master(args.panel)
    times = pd.read_parquet(args.panel, columns=["time"])["time"]

    d = d0.loc[args.start:args.end]
    full = pd.date_range(d.index[0], d.index[-1], freq="D")
    missing = full.difference(d.index)
    if len(missing):
        raise ValueError(
            f"daily series has {len(missing)} missing days, e.g. "
            f"{[str(x.date()) for x in missing[:3]]}; the recursion steps one "
            "row per calendar day"
        )
    print(f"daily series: {len(d)} days, {d.index[0].date()} -> {d.index[-1].date()}")

    # --- pure filter (the paper's series), one-sided and two-sided ---
    ev_d = daily_evidence(times, reps=50_000, seed=123)
    pi_d = pd.Series(d.index).map(ev_d["pi"]).fillna(0.0).values
    res_pure = pure_filter(d["H"].values, pi_d=pi_d, two_sided=True, dates=d.index)
    print(
        f"pure filter: q_quiet fitted {res_pure.q_quiet:.3e} (paper: 7.81e-4)  "
        f"growth SD one-sided {100 * np.diff(res_pure.h_filt).std():.2f}%  "
        f"two-sided {100 * np.diff(res_pure.h_smooth).std():.2f}%"
    )
    if not args.no_compare:
        compare(pd.Series(res_pure.h_filt, index=d.index), args.reference_pure, "log_h", "pure")

    # --- extended filter: difficulty anchor, de-lagged ---
    # The paper's extended series was built with a 60,000-draw null (the pure
    # filter used 50,000) and ends 2026-07-11, so it is rebuilt on the sample
    # through that date.
    de = d.loc[:EXT_END] if d.index[-1] > pd.Timestamp(EXT_END) else d
    ev_e = daily_evidence(times, reps=60_000, seed=123)
    pi_e = pd.Series(de.index).map(ev_e["pi"]).fillna(0.0).values
    res_ext = anchored_filter(
        de["H"].values, de["D_open"].values, pi_d=pi_e, two_sided=True, dates=de.index
    )
    avar = res_ext.anchor_var[res_ext.anchor_var > 0]
    print(
        f"extended filter (anchor, de-lagged, to {de.index[-1].date()}): "
        f"q_quiet {res_ext.q_quiet:.3e}  median anchor variance {np.median(avar):.3e}  "
        f"growth SD one-sided {100 * np.diff(res_ext.h_filt).std():.2f}%  "
        f"two-sided {100 * np.diff(res_ext.h_smooth).std():.2f}%"
    )
    if not args.no_compare:
        compare(pd.Series(res_ext.h_filt, index=de.index), args.reference_extended, "log_h", "extended")
        print(
            "  context: the reference was built on the provider's raw column through\n"
            "  August 2025 with offset 15.8218; the de-lag projection amplifies that\n"
            "  input difference around the big early resets (2011, 2013). On the\n"
            "  paper's own inputs the filter reproduces the series to 1e-14."
        )

    # --- two-sided correction summary ---
    gap = res_pure.h_filt - res_pure.h_smooth
    print(
        f"two-sided revision (pure): RMS |two-sided - one-sided| "
        f"{100 * np.sqrt((gap**2).mean()):.2f} log-%"
    )


if __name__ == "__main__":
    main()
