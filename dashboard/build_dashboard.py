"""Daily builder of the live hashrate dashboard.

Reads the stored state (the CSV published by the previous run, or the
stored history in dashboard/data/ on the first run), adds every complete
day since, re-smooths the two-sided tail, runs the checks below, and
writes the deployable site: index.html, data/daily.csv, site/dashboard.css
and the link-preview image. Nothing is committed anywhere; the published
CSV is both the download and the next run's state.

    python dashboard/build_dashboard.py                  # full run
    python dashboard/build_dashboard.py --state-file X.csv   # start from a file
    python dashboard/build_dashboard.py --from-cache --blocks-cache CACHE.json   # offline

Checks (any failure exits nonzero, so the scheduled workflow never
deploys a bad page; the failed run is the alert):

  G1  stored published days equal dashboard/data/history.csv, and the
      stored one-sided series equals the paper's reference
      (examples/data/reference_pure_filter.csv) to the stated deviation;
  G2  the one-day weight step on the shipped sample blocks
      (examples/data/sample_blocks_2021-05-01_2021-07-31.csv) reproduces
      the stored weights of those days;
  G3  replaying the one-day filter step over every day after the stored
      history's end reproduces the stored filtered levels and variances;
  G4  per-day sanity of new rows (block count, difficulty, growth);
  G5  re-smoothing the two-sided tail leaves published days inside the
      window at their stored values;
  G6  schema: consecutive dates, no gaps, weights in [0, 1], positive
      levels and variances.

The past is never re-run and never republished: days up to the stored
history's end keep their published values exactly, and each new day is
added from the stored state of the day before. Only the two-sided line's
tail is re-smoothed, because it uses the following days by construction.
"""

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))  # the package, uninstalled

import daily_step  # noqa: E402
import mempool  # noqa: E402
import render  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SITE_DIR = HERE / "site"

STATE_URL = "https://janwuestenfeld.github.io/hashrate-filter/data/daily.csv"
SITE_URL = "https://janwuestenfeld.github.io/hashrate-filter/"
COLUMNS = [
    "date", "n_blocks", "difficulty", "raw", "pure",
    "pure_two_sided", "ma3", "weight", "p_filt",
]

# per-day sanity bounds (G4), set from the stored history's ranges
N_MIN, N_MAX = 40, 400            # historical range: 58 to 307 blocks/day
D_RATIO_MIN, D_RATIO_MAX = 0.2, 5.0  # day-over-day mean difficulty
GROWTH_MAX = 0.9                  # |raw daily growth| (historical max 0.59)


class GateFailure(RuntimeError):
    pass


def log(msg: str) -> None:
    print(f"[build] {msg}", flush=True)


# ---------------------------------------------------------------- state I/O
def load_meta() -> dict:
    return json.loads((DATA / "meta.json").read_text())


def load_scan_null() -> np.ndarray:
    return np.load(DATA / "evidence_reference.npz")["scan_null"]


def load_state(args) -> pd.DataFrame:
    """The stored daily series: --state-file, else the published CSV, else
    the stored history (first run)."""
    if args.state_file:
        log(f"state: file {args.state_file}")
        text = Path(args.state_file).read_text()
    else:
        try:
            log(f"state: {STATE_URL}")
            with urllib.request.urlopen(STATE_URL, timeout=60) as r:
                text = r.read().decode()
            if "date" not in text.splitlines()[0]:
                raise ValueError("unexpected content at the state URL")
        except Exception as e:
            log(f"state URL unavailable ({e}); bootstrapping from the stored history")
            text = (DATA / "history.csv").read_text()
    df = pd.read_csv(io.StringIO(text), parse_dates=["date"])
    if list(df.columns) != COLUMNS:
        raise GateFailure(f"state columns {list(df.columns)} != expected {COLUMNS}")
    return df.sort_values("date").reset_index(drop=True)


def write_state(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.17g", date_format="%Y-%m-%d")


# ---------------------------------------------------------------- gates
def gate_g1(df: pd.DataFrame, meta: dict) -> None:
    """Published days equal the stored history; the stored one-sided series
    equals the paper's reference to the stated deviation."""
    end = pd.Timestamp(meta["sample_end"])
    hist = pd.read_csv(DATA / "history.csv", parse_dates=["date"])
    pub = df[df["date"] <= end].reset_index(drop=True)
    if len(pub) != len(hist) or not pub["date"].equals(hist["date"]):
        raise GateFailure("G1: published days do not align with the stored history")
    for col in ["n_blocks", "difficulty", "raw", "pure", "pure_two_sided", "ma3", "weight", "p_filt"]:
        a, b = pub[col].values.astype(float), hist[col].values.astype(float)
        if not np.allclose(a, b, rtol=1e-12, atol=1e-12):
            raise GateFailure(f"G1: published column {col} differs from the stored history")

    ref = pd.read_csv(REPO / "examples" / "data" / "reference_pure_filter.csv", parse_dates=["date"])
    diff = np.abs(np.log(pub["pure"].values) - ref["log_h"].values)
    i = int(np.argmax(diff))
    stated = {"median": float(np.median(diff)), "max": float(diff.max()),
              "max_date": str(pub["date"].iloc[i].date())}
    if not (stated["median"] < 1e-4 and stated["max"] < 2e-2 and stated["max_date"] == meta["sample_end"]):
        raise GateFailure(f"G1: stored series vs the paper's reference beyond the stated deviation: {stated}")
    log(f"G1 ok: published days stored exactly; vs the paper's reference {stated}")


def gate_g2(scan_null: np.ndarray, prior: float) -> None:
    """The one-day weight step on the shipped sample blocks reproduces the
    stored weights of those days."""
    blocks = pd.read_csv(
        REPO / "examples" / "data" / "sample_blocks_2021-05-01_2021-07-31.csv",
        parse_dates=["time", "median_time"],
    )
    hist = pd.read_csv(DATA / "history.csv", parse_dates=["date"]).set_index("date")
    t = blocks["time"]
    days = t.dt.floor("D")
    span = t.groupby(days).agg(["min", "max"])
    max_diff = 0.0
    checked = 0
    for day in sorted(days.unique()):
        # skip partial edge days, as the batch evidence does
        if day == days.min() and span.loc[day, "min"] > day + pd.Timedelta(hours=2):
            continue
        if day == days.max() and span.loc[day, "max"] < day + pd.Timedelta(hours=22):
            continue
        pi, _, _, _ = daily_step.day_weight(t[days == day], scan_null, prior)
        max_diff = max(max_diff, abs(pi - hist.loc[day, "weight"]))
        checked += 1
    if checked < 80:
        raise GateFailure(f"G2: only {checked} sample days checked")
    if max_diff > 1e-9:
        raise GateFailure(f"G2: one-day weight step deviates from the stored weights by {max_diff:.2e}")
    log(f"G2 ok: one-day weight step reproduces the stored weights on {checked} sample days")


def gate_g3(df: pd.DataFrame, meta: dict) -> None:
    """Replay the one-day filter step over every post-history day."""
    end = pd.Timestamp(meta["sample_end"])
    i0 = int(np.flatnonzero(df["date"].values == np.datetime64(end))[0])
    if i0 == len(df) - 1:
        log("G3 ok: no appended days to replay")
        return
    h = float(np.log(df["pure"].iloc[i0]))
    P = float(df["p_filt"].iloc[i0])
    q_quiet = float(meta["q_quiet"])
    worst = 0.0
    for t in range(i0 + 1, len(df)):
        y = float(np.log(df["raw"].iloc[t]))
        q = daily_step.q_of(float(df["weight"].iloc[t]), q_quiet)
        h, P = daily_step.filter_step(h, P, y, q)
        worst = max(
            worst,
            abs(h - np.log(df["pure"].iloc[t])),
            abs(P - df["p_filt"].iloc[t]),
        )
    if worst > 1e-9:
        raise GateFailure(f"G3: replay of the daily step deviates by {worst:.2e}")
    log(f"G3 ok: daily step replays the stored series exactly over {len(df) - i0 - 1} appended days")


def gate_g6(df: pd.DataFrame) -> None:
    deltas = np.diff(df["date"].values).astype("timedelta64[D]").astype(int)
    if (deltas != 1).any():
        raise GateFailure("G6: dates are not consecutive calendar days")
    if df[COLUMNS].isna().any().any():
        raise GateFailure("G6: missing values in the stored series")
    if ((df["weight"] < 0) | (df["weight"] > 1)).any():
        raise GateFailure("G6: weights outside [0, 1]")
    if (df[["raw", "pure", "pure_two_sided", "ma3", "p_filt", "difficulty"]] <= 0).any().any():
        raise GateFailure("G6: non-positive levels or variances")
    log("G6 ok: schema")


# ---------------------------------------------------------------- update
def append_new_days(df: pd.DataFrame, blocks: pd.DataFrame, meta: dict, scan_null: np.ndarray) -> tuple:
    """Append every complete day after the stored end. Returns
    (df, n_added, two-sided tail shift)."""
    prior = float(meta["prior"])
    q_quiet = float(meta["q_quiet"])
    last = df["date"].iloc[-1]
    days = [day for day in mempool.complete_days(blocks) if day > last]
    if not days:
        log("no new complete days")
        return df, 0, 0.0
    expected = pd.date_range(last + pd.Timedelta(days=1), periods=len(days), freq="D")
    if list(days) != list(expected):
        raise GateFailure(f"new days not consecutive after {last.date()}: {days[:3]}...")

    med_day = blocks["median_time"].dt.floor("D")
    hdr_day = blocks["time"].dt.floor("D")
    h = float(np.log(df["pure"].iloc[-1]))
    P = float(df["p_filt"].iloc[-1])
    rows = []
    for day in days:
        b_med = blocks[med_day == day]
        n = len(b_med)
        D_mean = float(b_med["difficulty"].mean())
        raw = n * D_mean * 2.0**32 / 86400.0
        y = float(np.log(raw))
        pi, _, _, _ = daily_step.day_weight(blocks.loc[hdr_day == day, "time"], scan_null, prior)
        q = daily_step.q_of(pi, q_quiet)
        h, P = daily_step.filter_step(h, P, y, q)

        # G4: per-day sanity
        prev_raw = df["raw"].iloc[-1] if not rows else rows[-1]["raw"]
        prev_D = df["difficulty"].iloc[-1] if not rows else rows[-1]["difficulty"]
        g = abs(y - float(np.log(prev_raw)))
        if not (N_MIN <= n <= N_MAX):
            raise GateFailure(f"G4: {day.date()} block count {n} outside [{N_MIN}, {N_MAX}]")
        if not (D_RATIO_MIN <= D_mean / prev_D <= D_RATIO_MAX):
            raise GateFailure(f"G4: {day.date()} difficulty ratio {D_mean / prev_D:.3f}")
        if g > GROWTH_MAX:
            raise GateFailure(f"G4: {day.date()} raw growth {g:.2f} beyond {GROWTH_MAX}")
        rows.append(
            {
                "date": day, "n_blocks": n, "difficulty": D_mean, "raw": raw,
                "pure": np.exp(h), "pure_two_sided": np.nan, "ma3": np.nan,
                "weight": pi, "p_filt": P,
            }
        )

    n_added = len(rows)
    df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    log(f"appended {n_added} days: {rows[0]['date'].date()} -> {rows[-1]['date'].date()}")

    # trailing three-day moving average over the raw levels (the paper's
    # comparison series; rolling mean, min_periods=1)
    df["ma3"] = df["raw"].rolling(3, min_periods=1).mean()

    # Re-smooth the two-sided line. The window covers every newly appended
    # day (they have no value yet) plus TAIL_WINDOW days before the first
    # new day: the two-sided line uses the following days by construction,
    # so its tail revises as new days arrive. Revisions decay by the
    # smoother gain per day and are far below visibility beyond the window;
    # older days keep their stored values.
    first_new = len(df) - n_added
    a = max(0, first_new - daily_step.TAIL_WINDOW)
    revised = np.exp(
        daily_step.rts_tail(
            np.log(df["pure"].values[a:]),
            df["p_filt"].values[a:],
            daily_step.q_of(df["weight"].values[a:], q_quiet),
        )
    )
    stored = df["pure_two_sided"].values[a:]
    had_value = ~np.isnan(stored)
    shift = 0.0
    if had_value.any():
        # G5: revisions of previously stored days stay within the plausible
        # envelope of two-sided tail revision; a larger shift means a bug
        shift = float(np.abs(revised[had_value] / stored[had_value] - 1.0).max())
        if shift > 5e-2:
            raise GateFailure(f"G5: two-sided revision of stored days reached {shift:.2%}")
        log(f"G5 ok: two-sided tail revised stored days by at most {shift:.2%}")
    new_two_sided = df["pure_two_sided"].values.copy()
    new_two_sided[a:] = revised
    df["pure_two_sided"] = new_two_sided
    return df, n_added, shift


# ---------------------------------------------------------------- main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-file", help="start from this CSV instead of the published one")
    ap.add_argument("--out", default=str(HERE / "build"), help="output directory for the site")
    ap.add_argument("--blocks-cache", help="cache file for the fetched blocks")
    ap.add_argument("--from-cache", action="store_true", help="read blocks from the cache, do not fetch")
    ap.add_argument("--pace", type=float, default=0.3, help="seconds between API calls")
    args = ap.parse_args()

    meta = load_meta()
    scan_null = load_scan_null()
    df = load_state(args)

    gate_g1(df, meta)
    gate_g6(df)

    if args.from_cache:
        if not args.blocks_cache:
            raise ValueError("--from-cache needs --blocks-cache")
        blocks = mempool.fetch_blocks_back_to(
            df["date"].iloc[-1] + pd.Timedelta(days=1), cache_path=args.blocks_cache, from_cache=True
        )
    else:
        blocks = fetch_with_retry(df["date"].iloc[-1] + pd.Timedelta(days=1), args)

    df, n_added, tail_shift = append_new_days(df, blocks, meta, scan_null)

    gate_g2(scan_null, float(meta["prior"]))
    gate_g3(df, meta)
    gate_g6(df)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_state(df, out / "data" / "daily.csv")
    render.write_site(
        df, meta, out,
        site_url=SITE_URL, repo_root=REPO, css_src=SITE_DIR / "dashboard.css",
        tail_shift=tail_shift,
        fetched_utc=blocks.attrs.get("fetched_utc", ""),
    )
    log(f"wrote site to {out}")
    log(f"as of {df['date'].iloc[-1].date()}: filtered {df['pure'].iloc[-1] / 1e18:.1f} EH/s, "
        f"raw {df['raw'].iloc[-1] / 1e18:.1f} EH/s, weight {df['weight'].iloc[-1]:.3f}")


def fetch_with_retry(first_day: pd.Timestamp, args) -> pd.DataFrame:
    """Fetch blocks back to the day after the stored end; on any fetch
    failure, abort (the deploy must not run on partial data)."""
    log(f"fetching blocks back to {first_day.date()} (pace {args.pace}s)")
    blocks = mempool.fetch_blocks_back_to(first_day, pace=args.pace, cache_path=args.blocks_cache)
    log(
        f"fetched {len(blocks)} blocks, tip {blocks.attrs['tip_height']}, "
        f"complete through {blocks.attrs['tip_median_day'] - pd.Timedelta(days=1):%Y-%m-%d}"
    )
    return blocks


if __name__ == "__main__":
    try:
        main()
    except (GateFailure, mempool.FetchError) as e:
        print(f"[build] FAILED: {e}", file=sys.stderr)
        sys.exit(1)
