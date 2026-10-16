"""Build the repository's overview figure: the raw estimate and the filters.

Reads the shipped example output (examples/data/example_output.csv) and
writes examples/figures/filter_overview.png. Three panels: daily growth over
the full sample, the 2021 Chinese mining ban (the one-sided filter follows
the decline as it happens; the two-sided version moves before the shock
because it uses the following days), and daily growth in the first half of
2024 with the three-day moving average for comparison.

    python make_figure.py                        # plot the shipped series
    python make_figure.py --panel BLOCKS.parquet # rebuild the series first

With --panel the series is rebuilt from a block panel (columns time,
median_time, id, difficulty) and written to
examples/data/example_output_rebuilt.csv; the shipped file is never
overwritten. Requires matplotlib (the examples extra).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
DATA = HERE / "data" / "example_output.csv"
REBUILT = HERE / "data" / "example_output_rebuilt.csv"
OUT = HERE / "figures" / "filter_overview.png"

RAW, MA3, PURE, PURE2S, EXT = "#c4c4c4", "#4878a8", "#c0392b", "#2e8b57", "#2c3e50"

START, END = "2011-01-01", "2026-07-15"


def rebuild(panel_path: str) -> None:
    """Rebuild the example output from a block panel."""
    from hashrate_filter import daily_evidence
    from hashrate_filter.variants import anchored_filter, pure_filter

    p = pd.read_parquet(
        panel_path, columns=["time", "median_time", "id", "difficulty"]
    )
    p["Date"] = pd.to_datetime(p["median_time"]).dt.normalize()
    p = p.sort_values("id")
    d = p.groupby("Date").agg(
        n=("id", "size"), D=("difficulty", "mean"), D_open=("difficulty", "first")
    )
    d["H"] = d["n"] * d["D"] * 2**32 / 86400
    d = d.loc[START:END]

    ev = daily_evidence(p["time"], reps=50_000, seed=123)
    pi_d = pd.Series(d.index).map(ev["pi"]).fillna(0.0).values
    rp = pure_filter(d["H"].values, pi_d=pi_d, two_sided=True, dates=d.index)

    ev_e = daily_evidence(p["time"], reps=60_000, seed=123)
    pi_e = pd.Series(d.index).map(ev_e["pi"]).fillna(0.0).values
    ra = anchored_filter(d["H"].values, d["D_open"].values, pi_d=pi_e, two_sided=True, dates=d.index)

    pd.DataFrame(
        {
            "date": d.index,
            "raw": d["H"].values,
            "pure": np.exp(rp.h_filt),
            "pure_two_sided": np.exp(rp.h_smooth),
            "extended": np.exp(ra.h_filt),
            "extended_two_sided": np.exp(ra.h_smooth),
        }
    ).to_csv(REBUILT, index=False)
    print(f"rebuilt {REBUILT} ({len(d)} days)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", help="optional block panel parquet to rebuild the series first")
    args = ap.parse_args()
    source = DATA
    if args.panel:
        rebuild(args.panel)
        source = REBUILT

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = pd.read_csv(source, parse_dates=["date"])
    eh = lambda c: d[c] / 1e18  # H/s -> EH/s
    lg = lambda c: np.log(d[c])
    ma3 = np.log(d["raw"].rolling(3, min_periods=1).mean())

    fig, axes = plt.subplots(3, 1, figsize=(11, 12))

    ax = axes[0]
    ax.plot(d["date"], 100 * lg("raw").diff(), color=RAW, lw=0.3,
            label=f"raw estimate (SD {100 * lg('raw').diff().std():.1f}%/day)")
    ax.plot(d["date"], 100 * lg("pure").diff(), color=PURE, lw=0.7,
            label=f"pure filter, one-sided (SD {100 * lg('pure').diff().std():.1f}%/day)")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel("%/day")
    ax.set_title("Daily hashrate growth, 2011 to 2026")
    ax.legend(loc="upper right")

    ax = axes[1]
    m = (d["date"] >= "2021-03-01") & (d["date"] <= "2021-11-30")
    ax.plot(d["date"][m], eh("raw")[m], color=RAW, lw=0.6, label="raw estimate")
    ax.plot(d["date"][m], eh("pure")[m], color=PURE, lw=1.7, label="pure filter, one-sided")
    ax.plot(d["date"][m], eh("pure_two_sided")[m], color=PURE2S, lw=1.2, ls="--",
            label="pure filter, two-sided")
    ax.plot(d["date"][m], eh("extended")[m], color=EXT, lw=1.1, label="extended filter")
    ax.axvspan(pd.Timestamp("2021-05-15"), pd.Timestamp("2021-07-05"), color="orange", alpha=0.12)
    ax.set_ylabel("hashrate (EH/s)")
    ax.set_title("Hashrate during the 2021 Chinese mining ban")
    ax.legend(loc="lower left")

    ax = axes[2]
    m = (d["date"] >= "2024-01-01") & (d["date"] <= "2024-06-30")
    w = lambda c: 100 * lg(c).diff()[m].std()
    ax.plot(d["date"][m], 100 * lg("raw").diff()[m], color=RAW, lw=0.5,
            label=f"raw estimate (SD {w('raw'):.1f}%/day in window)")
    ax.plot(d["date"][m], 100 * ma3.diff()[m], color=MA3, lw=0.9,
            label="three-day moving average")
    ax.plot(d["date"][m], 100 * lg("pure").diff()[m], color=PURE, lw=1.5,
            label=f"pure filter, one-sided (SD {w('pure'):.1f}%/day in window)")
    ax.plot(d["date"][m], 100 * lg("pure_two_sided").diff()[m], color=PURE2S, lw=1.1, ls="--",
            label=f"pure filter, two-sided (SD {w('pure_two_sided'):.1f}%/day in window)")
    ax.axhline(0, color="k", lw=0.4)
    ax.set_ylabel("%/day")
    ax.set_title("Daily hashrate growth, first half of 2024")
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
