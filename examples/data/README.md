# Example data

All dates are UTC calendar days.

| file | rows | span | columns | units |
|---|---|---|---|---|
| `reference_pure_filter.csv` | 5,675 | 2011-01-01 to 2026-07-15 | `date`, `log_h` | natural log of H/s |
| `reference_extended_filter.csv` | 5,671 | 2011-01-01 to 2026-07-11 | `date`, `log_h` | natural log of H/s |
| `example_output.csv` | 5,675 | 2011-01-01 to 2026-07-15 | `date`, `raw`, `pure`, `pure_two_sided`, `extended`, `extended_two_sided` | H/s |
| `sample_blocks_2021-05-01_2021-07-31.csv` | 12,332 | 2021-05-01 to 2021-07-31 | `id`, `time`, `median_time`, `difficulty` | one row per block |

## Reference series

The two reference files are the series the paper reports. The pure-filter
file is the series the paper's estimations ran on. The extended-filter file
is the de-lagged difficulty-anchor construction the paper uses as its
agreement check; it ends on 2026-07-11, four days before the pure-filter
file. On the paper's own inputs the package reproduces the extended series
exactly (to 1e-14, verified 16 October 2026).

`examples/reproduce_paper_series.py` rebuilds both from a block panel and
compares them with these files. The rebuilt pure filter matches to a median
absolute difference of about 5e-5 in log units, with the largest gap on the
incomplete final day. The extended rebuild differs by a few 1e-3 in level,
because the paper's raw estimate came from the provider's column through
August 2025 and from the block panel after; the largest gaps fall around
the big 2011 and 2013 difficulty resets, where the de-lag projection is
large.

In the pure-filter reference, the value for 2020-05-05 was restored after
the block panel was repaired (the day's 158 blocks were missing when the
series was first built), and the recursion was not re-run on the days
before it.

## Example output

`example_output.csv` is the series behind `examples/figures/filter_overview.png`,
built from the block panel with `examples/make_figure.py --panel`. The `raw`
column is the one-day estimate `n_blocks x mean_difficulty x 2^32 / 86400`.

## Sample blocks

The sample file carries every block of May to July 2021, the window of the
Chinese mining ban, so the arrival-time weight has flagged days to work on.
It is enough to run every example in the README, including the extended
filter. The columns follow the Blockchair dump schema: `id` is the block
height, `time` the header timestamp, `median_time` the median of the header
timestamps of the last eleven blocks, `difficulty` the block's difficulty.
