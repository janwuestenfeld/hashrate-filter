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
- `mempool.py` — the paced block fetcher. A day is complete once the chain
  tip's median time has passed into the next day; median time never
  decreases along the chain.
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

By construction the two-sided version uses the following days, so its last
week is revised as new days arrive. It is the only line on the page whose
past moves: the raw estimate, the one-sided filter and the moving average
are never revised once a day is published. The page says so, next to the
line itself.
