# The live dashboard

A public page that shows, every day, the raw daily Bitcoin hashrate
estimate against the paper's filter, the three-day moving average, and the
two-sided benchmark, so the filtered series can be seen and downloaded
without running code. Served by GitHub Pages at
`https://janwuestenfeld.github.io/hashrate-filter/`, rebuilt daily by
`.github/workflows/dashboard.yml`.

## How it works

The past is never re-run and never republished. `data/history.csv` holds
the stored history (2011-01-01 to 2026-07-15, the paper's sample); each
daily run reads the CSV published by the previous run, adds the new
complete UTC day(s) from mempool.space block data, re-smooths the
two-sided tail, runs its checks, and deploys the build directory as the
Pages artifact. Nothing is committed by the workflow.

- `data/history.csv` — the stored history. One row per day: `date`,
  `n_blocks`, `difficulty` (day mean), `raw`, `pure`, `pure_two_sided`,
  `ma3` (H/s), `weight` (the arrival-time weight), `p_filt` (variance of
  the filtered log level). The published columns are the paper's series
  (the figure series in `examples/data/example_output.csv`, verified
  against `examples/data/reference_pure_filter.csv` to the README's stated
  deviation); the weight and variance are rebuilt from the paper's inputs
  and drive the daily update.
- `data/evidence_reference.npz` — the stored scan null (50,000 Monte Carlo
  replications, seed 123, the settings of the paper's series). A new day's
  weight is computed against this fixed reference, so it is computed the
  same way every time.
- `data/meta.json` — the stored prior, the fitted quiet-day variance, and
  the verification numbers of the one-time history build.
- `build_history.py` — the one-time builder of the above (local use, from
  the paper's block panel; the panel is an argument, never stored).
- `daily_step.py` — the one-day steps (weight, filter update, two-sided
  tail), written from the model in the repository README with no private
  imports.
- `blockchair.py` — the fetcher for Blockchair's daily dumps
  (`gz.blockchair.com/bitcoin/blocks/`), the paper's source. New days come
  from the same dumps the paper's history was built from.
- `mempool.py` — the mempool.space live API fetcher, used in two
  documented cases only: a day whose dump is not available (yet), and the
  after-midnight stragglers whose median time still belongs to the day
  before (they close the latest day; the following day's own dump is
  published only when that day ends). Both read the same blockchain; the
  build log records which source served each day.
- `build_dashboard.py` — the daily build: state, append, gates, render.
- `render.py`, `site/dashboard.css` — the page. Design language adapted
  from the Bitcoin Market Structure Dashboard
  (`janwuestenfeld/btc-dashboard`).

## Local runs

```bash
python dashboard/build_dashboard.py --out dashboard/build
# offline, reusing fetched blocks:
python dashboard/build_dashboard.py --from-cache --blocks-cache dashboard/build/blocks_cache.json
```

The build refuses to write a bad page: any failed check (G1–G6, listed in
`build_dashboard.py`) exits nonzero, so the scheduled workflow fails
before the deploy steps and nothing wrong is published.

## The two-sided line

By construction the two-sided version uses the following days, so it runs
a week behind: days with fewer than seven following days are left empty in
the published CSV (they fill in as new days arrive), and its last days
keep being revised. It is the only line on the page whose past moves: the
raw estimate, the one-sided filter and the moving average are never
revised once a day is published. Published days of the paper's sample are
exempt from the empty-tail rule — they carry the paper's published values.
The page says all of this, next to the line itself.
