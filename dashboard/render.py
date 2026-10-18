"""Render the dashboard page from the stored daily series.

Computes the live readings (in the paper's own conventions, documented
below), embeds the series as JSON, and writes index.html in the design
language of Jan's Bitcoin Market Structure Dashboard (Plotly 2.32,
Geist/Inter/JetBrains Mono, one muted palette, instrument grammar:
eyebrow, question, answer, claim-number, prose, figure, note).

Live readings and their conventions (paper, Section 4; README):

  * luck removed      100 x (var(w(g_raw)) - var(w(g_filter))) /
                      mean(1/n_t + 1/n_{t-1}), with w() winsorizing daily
                      growth at 0.5 percent per tail, identically across
                      series; over the full sample and from 2016;
  * forecast scores   mean Poisson log-likelihood of the next day's block
                      count under each construction's implied rate,
                      lambda(t+1) = H_cand(t) x 86400 / (D(t+1) x 2^32),
                      from 2016; plus the mean absolute count error;
  * smear and lag     around the largest absolute raw daily move of the
                      last 45 complete days: how much of it the filter and
                      the three-day moving average record on the day and
                      on the two following days;
  * two-sided gap     RMS and tail maxima of two-sided minus one-sided
                      over the days the two-sided estimate exists (it runs
                      a week behind; the trailing days are empty), and how
                      much the latest update revised the tail.

Numbers quoted as the paper's (95.6 percent, 93.6 percent, 11.8 against
3.10 percent per day, 31/half/95 percent response, the forecast scores
-4.095/-4.167/-4.484) are literals from the publication; numbers without
that label are computed from the live series at build time.
"""

import json
import shutil
import string
from pathlib import Path

import numpy as np
import pandas as pd

from daily_step import poisson_loglik_next_day

PAPER_URL = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5786322"
REPO_URL = "https://github.com/janwuestenfeld/hashrate-filter"
JAN_URL = "https://janwuestenfeld.github.io/wuestenfeld.eu/"
TEO_URL = "https://www.linkedin.com/in/teo-geldner-96802bb7/"

EH = 1e18  # H/s per EH/s

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fmt_date(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def fmt_pct(x, dp=1, sign=True) -> str:
    s = "+" if (sign and x > 0) else ("−" if x < 0 else "")
    return f"{s}{abs(x):.{dp}f}%"


def _winsor(x):
    lo, hi = np.nanquantile(x, [0.005, 0.995])
    return np.clip(x, lo, hi)


def compute_stats(df: pd.DataFrame, meta: dict, tail_shift: float) -> dict:
    d = df
    n = len(d)
    days = d["date"]
    log_raw = np.log(d["raw"].values)
    log_pure = np.log(d["pure"].values)
    two_vals = d["pure_two_sided"].values.astype(float)
    valid_two = ~np.isnan(two_vals)
    log_two = np.full(n, np.nan)
    log_two[valid_two] = np.log(two_vals[valid_two])
    log_ma3 = np.log(d["ma3"].values)
    n_blocks = d["n_blocks"].values.astype(float)

    g_raw = 100.0 * np.diff(log_raw)
    g_pure = 100.0 * np.diff(log_pure)
    g_ma3 = 100.0 * np.diff(log_ma3)
    gdays = days.values[1:]

    st = {}
    st["as_of"] = days.iloc[-1]
    st["two_last"] = days[valid_two].iloc[-1]
    st["pure_eh"] = d["pure"].iloc[-1] / EH
    st["raw_eh"] = d["raw"].iloc[-1] / EH
    st["d1"] = g_pure[-1]
    st["d7"] = 100.0 * (log_pure[-1] - log_pure[-8]) if n > 7 else np.nan
    st["luck_last"] = 100.0 * (log_raw[-1] - log_pure[-1])
    st["band_pct"] = 100.0 * 1.96 * np.sqrt(d["p_filt"].iloc[-1])
    st["weight_last"] = float(d["weight"].iloc[-1])
    st["n_last"] = int(d["n_blocks"].iloc[-1])

    # --- luck removed (the paper's convention; growth in log units, so the
    # variance and the per-day Poisson variance 1/n_t + 1/n_{t-1} match) ---
    ln_raw = np.diff(log_raw)
    ln_pure = np.diff(log_pure)

    def luck_share(ln_f, mask, lo_date):
        m = mask & (gdays >= np.datetime64(lo_date))
        luck_var = (1.0 / n_blocks[1:] + 1.0 / n_blocks[:-1])[m]
        return 100.0 * (np.var(_winsor(ln_raw[m])) - np.var(_winsor(ln_f[m]))) / np.mean(luck_var)

    full = np.ones(n - 1, dtype=bool)
    st["luck_pure_full"] = luck_share(ln_pure, full, "2011-01-01")
    st["luck_pure_16"] = luck_share(ln_pure, full, "2016-01-01")
    st["luck_two_full"] = luck_share(np.diff(log_two), valid_two[1:], "2011-01-01")
    st["sd_raw_full"] = float(g_raw.std())
    st["sd_pure_full"] = float(g_pure.std())
    st["sd_raw_16"] = float(g_raw[gdays >= np.datetime64("2016-01-01")].std())
    st["sd_pure_16"] = float(g_pure[gdays >= np.datetime64("2016-01-01")].std())

    # --- forecast scores, from 2016 (the paper's construction) ---
    idx = np.flatnonzero(days.values >= np.datetime64("2016-01-01"))
    idx = idx[idx < n - 1]
    n_next = n_blocks[idx + 1]
    D_next = d["difficulty"].values[idx + 1]
    st["score_years"] = int(pd.Timestamp(days.values[idx[0]]).year), int(days.iloc[-1].year)
    st["score_n"] = len(idx)
    st["score_pure"] = poisson_loglik_next_day(d["pure"].values[idx], n_next, D_next)
    st["score_ma3"] = poisson_loglik_next_day(d["ma3"].values[idx], n_next, D_next)
    st["score_raw"] = poisson_loglik_next_day(d["raw"].values[idx], n_next, D_next)
    lam = lambda col: d[col].values[idx] * 86400.0 / (D_next * 2.0**32)
    st["mae_pure"] = float(np.abs(n_next - lam("pure")).mean())
    st["mae_raw"] = float(np.abs(n_next - lam("raw")).mean())

    # --- smear and lag around the latest large raw move ---
    # the event day needs a follow-up week (the filter's convergence) and
    # two following days (the moving average's echo), so it is picked from
    # the last 45 complete days with room to spare
    lo = max(1, n - 46)
    hi = n - 7
    e = lo + int(np.argmax(np.abs(g_raw[lo - 1: hi - 1])))
    st["ev_i"] = e
    st["ev_date"] = days.iloc[e]
    st["ev_graw"] = g_raw[e - 1]
    st["ev_gpure"] = g_pure[e - 1]
    st["ev_gma3"] = [g_ma3[e - 1], g_ma3[e], g_ma3[e + 1]]
    st["ev_win"] = [max(0, e - 6), min(n - 1, e + 6)]
    # what the move turned out to be: the filtered level five days after
    # against the day before the event, as a share of the raw move
    ev_move = log_raw[e] - log_raw[e - 1]
    st["ev_held5"] = 100.0 * (log_pure[e + 5] - log_pure[e - 1]) / ev_move
    st["ev_next_raw"] = 100.0 * (log_raw[e + 1] - log_raw[e]) / ev_move
    st["ev_kind"] = "luck" if st["ev_held5"] < 40.0 else "real"

    # --- the two-sided gap (over the days the estimate exists) ---
    gap = log_two[valid_two] - log_pure[valid_two]
    st["two_rms_full"] = 100.0 * float(np.sqrt((gap ** 2).mean()))
    st["two_rms_90"] = 100.0 * float(np.sqrt((gap[-90:] ** 2).mean()))
    st["two_tail7"] = 100.0 * float(np.abs(gap[-7:]).max())
    st["tail_shift"] = 100.0 * tail_shift

    st["q_quiet"] = float(meta["q_quiet"])
    st["prior"] = float(meta["prior"])
    return st


# ------------------------------------------------------------------ payload
def build_payload(df: pd.DataFrame, st: dict) -> str:
    d = df

    def eh(col):
        return [None if not np.isfinite(v) else round(v / EH, 4) for v in col]

    payload = {
        "as_of": str(st["as_of"].date()),
        "dates": d["date"].dt.strftime("%Y-%m-%d").tolist(),
        "raw": eh(d["raw"].values),
        "pure": eh(d["pure"].values),
        "two": eh(d["pure_two_sided"].values.astype(float)),
        "ma3": eh(d["ma3"].values),
        "pf": [float(f"{v:.4e}") for v in d["p_filt"].values],
        "ev": {"i": int(st["ev_i"]), "win": st["ev_win"]},
    }
    return json.dumps(payload, separators=(",", ":"))


# ------------------------------------------------------------------ html
TEMPLATE = string.Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bitcoin Hashrate Dashboard</title>
<meta name="description" content="$DESCRIPTION">
<link rel="canonical" href="${SITE_URL}">
<meta property="og:type" content="website">
<meta property="og:url" content="${SITE_URL}">
<meta property="og:title" content="Bitcoin Hashrate Dashboard">
<meta property="og:description" content="$DESCRIPTION">
<meta property="og:image" content="${SITE_URL}og-card.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Bitcoin Hashrate Dashboard">
<meta name="twitter:description" content="$DESCRIPTION">
<meta name="twitter:image" content="${SITE_URL}og-card.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" integrity="sha384-7TVmlZWH60iKX5Uk7lSvQhjtcgw2tkFjuwLcXoRSR4zXTyWFJRm9aPAguMh7CIra" crossorigin="anonymous" onerror="__plotlyFallback(0)"></script>
<script>
(function(){
  var MIRRORS = [
    "https://cdn.jsdelivr.net/npm/plotly.js@2.32.0/dist/plotly.min.js",
    "https://unpkg.com/plotly.js@2.32.0/dist/plotly.min.js"
  ];
  window.__plotlyFallback = function(nextIdx){
    if (typeof Plotly !== 'undefined') return;
    if (nextIdx >= MIRRORS.length) { console.error('Plotly failed to load from all CDN mirrors.'); return; }
    var s = document.createElement('script');
    s.src = MIRRORS[nextIdx];
    s.crossOrigin = 'anonymous';
    s.onerror = function(){ __plotlyFallback(nextIdx + 1); };
    document.head.appendChild(s);
  };
})();
</script>
<link rel="stylesheet" href="site/dashboard.css">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Ccircle cx='8' cy='8' r='7' fill='%23F26B16'/%3E%3C/svg%3E">
<script data-goatcounter="https://hashrate-filter.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
</head>
<body>
<main class="wrap">
  <a class="sr-only skip-link" href="#sec-series">Skip to content</a>

  <h1>Bitcoin Hashrate <span class="ac">Dashboard</span></h1>
  <div class="byline">by <a href="$JAN_URL" target="_blank" rel="noopener">Dr. Jan Wüstenfeld<span class="ext"> (homepage&nbsp;↗)</span></a> and <a href="$TEO_URL" target="_blank" rel="noopener">Teo Geldner<span class="ext"> (LinkedIn&nbsp;↗)</span></a> · <a href="$PAPER_URL" target="_blank" rel="noopener">the paper<span class="ext"> (SSRN&nbsp;↗)</span></a> · <a href="$REPO_URL" target="_blank" rel="noopener">the code<span class="ext"> (GitHub&nbsp;↗)</span></a></div>
  <div class="freshness">Updated daily from the public Bitcoin blockchain · data as of <span id="data-as-of">$AS_OF</span>, the last complete UTC day</div>
  <div class="tagline">Bitcoin's hashrate, without the luck.</div>
  <div class="lede prose">A day's raw hashrate estimate is mostly block-discovery luck: chance alone moves it by about 11.8% a day. This page shows the raw estimate against the filtered series of Wüstenfeld and Geldner (2025), which removes that luck using its known distribution — no window length, no bandwidth — updated daily, free to download.</div>

  <div class="bignums">
    <div class="bn"><span class="k">Filtered · $AS_OF</span><span class="v actual">$PURE_EH EH/s</span></div>
    <div class="bn"><span class="k">Day</span><span class="v">$D1</span></div>
    <div class="bn"><span class="k">Week</span><span class="v">$D7</span></div>
    <div class="bn"><span class="k">Raw today</span><span class="v">$LUCK_LAST</span><span class="sub">$LUCK_SIDE the filter — mostly luck</span></div>
  </div>

  <nav class="secnav" aria-label="Sections">
    <a href="#sec-series">The series</a>
    <a href="#sec-luck">Luck &amp; the average</a>
    <a href="#sec-twosided">Two-sided</a>
    <a href="#sec-download">Download</a>
  </nav>

  <!-- =================== 1 · THE SERIES =================== -->
  <div class="secthead first" id="sec-series">The series<span class="desc">the raw daily estimate against the filter, 2011 to today</span></div>

  <section class="panel" id="ind1">
    <div class="ptop"><div class="eyebrow">The main chart</div><span class="tier desc">daily · log scale</span></div>
    <h2>What is the network hashrate right now?</h2>
    <div class="answer"><span><b>$PURE_EH EH/s</b> on $AS_OF — $D1 on the day, $D7 on the week, on the filtered series.</span></div>
    <div class="claimnum">the raw estimate read $RAW_EH EH/s on the same day — $LUCK_LAST_ABS $LUCK_SIDE the filter, mostly block luck · $N_LAST blocks found</div>
    <p class="pnote">Bitcoin's hashrate has to be inferred from how many blocks the network finds, and block discovery is a lottery: given the difficulty, a day's block count is Poisson with mean 144, so luck alone moves a raw daily estimate by about 11.8% a day. The filter uses the known distribution of that luck. In the paper it removes 95.6% of the luck variance in daily growth from 2016 onward, and it records 31% of a lasting hashrate change on the day itself, half by the next day, 95% within a week.</p>
    <div class="tgl" role="group" aria-label="Chart overlays">
      <span class="tgllabel">Overlays</span>
      <button id="btn-raw" class="on" type="button">Raw estimate</button>
      <button id="btn-two" class="on" type="button">Two-sided</button>
      <button id="btn-band" type="button">95% band</button>
    </div>
    <figure>
      <div class="cap">Daily Bitcoin network hashrate, 2011 to today (log scale, EH/s)</div>
      <div id="ch-main" class="chart" style="height:460px"></div>
      <p class="chartnote">Drag the slider to zoom; hover for the day's values. The two-sided line runs a week behind: with fewer than seven following days there is no two-sided estimate yet.</p>
      <p class="chartsrc">Source: Blockchair daily block dumps, as in the paper · only complete UTC days</p>
    </figure>
    <details class="method">
      <summary>Methodology</summary>
      <div class="why prose">
        <p>Let y be the raw daily log hashrate estimate and h the log of the day's average hashrate. The model is a local-level state-space system: observation y = h + eps with Var(eps) = 1/144 fixed by the protocol (a day at the prevailing rate finds a Poisson number of blocks with mean 144); state h_t = h_{t-1} + eta_t with Var(eta_t) = q_t = (1 − pi_t) q_quiet + pi_t q_jump. Each day the filter moves its estimate toward the raw estimate by K_t = V_t / (V_t + 1/144) of the gap. The weight pi_t comes from the arrival times of the day's blocks: under a constant hashrate, arrivals are uniform over the day, so clustering is evidence of a real intraday shift (a centroid statistic and a 24-bin scan statistic, combined by Fisher's method, mapped into pi_t with a fixed prior).</p>
        <p>Constants: R = 1/144 (protocol) · q_quiet = 7.81e-4 (the one fitted parameter; this page runs $Q_QUIET) · q_jump = 1.51e-2 (calibrated) · scan null 50,000 Monte Carlo draws, seed 123 · prior $PRIOR. The reference for the weights is stored once and reused, so a new day's weight is computed the same way every time.</p>
        <p><b>What changes and what never changes.</b> Once a day is published here, its raw estimate, filtered value and moving-average value never change — the filter works from past and present data only and the past is never re-run. The two-sided line is the exception: it uses the following days by construction, so it runs a week behind and its last days are revised as new days arrive. Only complete UTC days are shown; days are assigned by median time (a day closes once the chain tip's median time passes midnight, which never decreases along the chain), and the arrival-time weight uses header times.</p>
        <p><b>Sources.</b> The history (2011 to 2026-07-15) is the paper's series, built from Blockchair's daily block dumps. New days come from the same dumps; if a day's dump is not yet published when the page is built, the same three fields (header time, median time, difficulty) are read from mempool.space's live API, which also supplies the post-midnight blocks that close the latest day. The page is built daily at about 04:30 UTC; the full construction is in the <a href="$REPO_URL" target="_blank" rel="noopener">repository</a>.</p>
      </div>
    </details>
  </section>

  <!-- =================== 2 · LUCK AND THE MOVING AVERAGE =================== -->
  <div class="secthead" id="sec-luck">Luck and the moving average<span class="desc">what the filter removes, and what the three-day average does to a move</span></div>

  <section class="panel" id="ind2">
    <div class="ptop"><div class="eyebrow">Block luck</div><span class="tier desc">daily · live</span></div>
    <h2>How much of the raw daily move is just luck?</h2>
    <div class="answer"><span><b>$LUCK_16%</b> since 2016, live — in the paper, 95.6%.</span></div>
    <div class="claimnum">raw $SD_RAW_16%/day against filter $SD_PURE_16%/day since 2016 · $SD_RAW_FULL% against $SD_PURE_FULL% over the full sample · the two-sided version removes $LUCK_TWO_FULL%</div>
    <p class="pnote">The raw estimate's growth is about four times as dispersed as the filtered series, and the difference is block luck. The bars show each day's luck directly: how far the raw estimate sat above or below the filtered rate at that day's close.</p>
    <figure>
      <div class="cap">The day's luck: raw minus filtered (%), last 120 days</div>
      <div id="ch-luck" class="chart" style="height:280px"></div>
      <p class="chartnote">Positive bars: block luck flattered the raw estimate that day. The orange marker is the latest day.</p>
    </figure>
    <p class="lawline"><b>The filter also predicts.</b> Whatever measures the hashrate best must predict the next day's block count best — no forecast model fitted. Mean Poisson log-likelihood of the next day's count, live, $SCORE_YEARS (n=$SCORE_N): <b>filter $SCORE_PURE</b>, three-day moving average $SCORE_MA3, raw $SCORE_RAW (paper: −4.095, −4.167, −4.484). In blocks, the mean next-day count error is $MAE_PURE for the filter against $MAE_RAW for the raw estimate.</p>
  </section>

  <section class="panel" id="ind3">
    <div class="ptop"><div class="eyebrow">The moving average's smear</div><span class="tier desc">daily · live</span></div>
    <h2>And what the three-day average does with it</h2>
    <div class="answer"><span>$EV_ANSWER</span></div>
    <div class="claimnum">on the day: raw $EV_GRAW · filter $EV_GPURE · three-day average $EV_GMA3_0 · then the average echoed $EV_GMA3_1 and $EV_GMA3_2 · five days later the filtered level held $EV_HELD5% of the raw move</div>
    <p class="pnote">Most published series smooth the raw estimate with a moving average. A moving average damps luck and real changes alike, and each day's noise echoes through it for the length of its window. The filter uses the known distribution of the luck instead, so there is no window length to choose. This panel recomputes the comparison around the largest raw daily move of the last 45 complete days, every day.</p>
    <figure>
      <div class="cap">Anatomy of the latest large move: the days around $EV_DATE</div>
      <div id="ch-event" class="chart" style="height:280px"></div>
      <p class="chartnote">The shaded day is the event. The moving average spends three days digesting what the filter records on the day — and keeps echoing it after the raw series has moved on.</p>
    </figure>
  </section>

  <!-- =================== 3 · THE TWO-SIDED BENCHMARK =================== -->
  <div class="secthead" id="sec-twosided">The two-sided benchmark<span class="desc">the version that also uses the following days</span></div>

  <section class="panel" id="ind4">
    <div class="ptop"><div class="eyebrow">The two-sided line</div><span class="tier desc">benchmark</span></div>
    <h2>What would the future add?</h2>
    <div class="answer"><span>$TWO_ANSWER</span></div>
    <div class="claimnum">RMS difference against the filter: $TWO_RMS_FULL over the full sample · $TWO_RMS_90 over the last 90 days · the latest update revised the tail by at most $TAIL_SHIFT</div>
    <p class="pnote revnote"><b>This is the only line on the page whose past moves.</b> It uses the following days by construction, so it runs a week behind — days with fewer than seven following days are left empty — and its last days keep being revised as new days arrive. The raw estimate, the filter and the moving average are never revised: once a day is published here, those values for it never change. The paper uses the two-sided version only as a benchmark: it removes more luck ($LUCK_TWO_FULL% on the live sample, 99.1% in the paper) but brings future information into each day's estimate.</p>
    <figure>
      <div class="cap">Filter and two-sided benchmark, last 180 days (EH/s)</div>
      <div id="ch-two" class="chart" style="height:320px"></div>
      <p class="chartnote">The shaded band marks the two-sided line's last week, the part still being revised as new days arrive. The line stops a week before today by construction.</p>
    </figure>
  </section>

  <!-- =================== DOWNLOAD =================== -->
  <div class="secthead" id="sec-download">Download &amp; citation<span class="desc">the full daily series, and how to cite it</span></div>

  <section class="panel">
    <div class="ptop"><div class="eyebrow">The series</div><span class="tier desc">CSV · daily</span></div>
    <h2>The daily series, as a CSV</h2>
    <p class="pnote">Every figure on this page is computed from one file: <a href="data/daily.csv" download><b>data/daily.csv</b></a> (one row per day, 2011 to $AS_OF, updated daily). Columns: <span class="num">date</span> (UTC calendar day) · <span class="num">n_blocks</span> · <span class="num">difficulty</span> (day mean) · <span class="num">raw</span>, <span class="num">pure</span>, <span class="num">ma3</span> (H/s: the raw one-day estimate, the filter, the trailing three-day moving average) · <span class="num">pure_two_sided</span> (the two-sided benchmark; the last seven days are empty — with fewer than seven following days there is no two-sided estimate yet, and the values fill in as new days arrive) · <span class="num">weight</span> (the arrival-time weight pi) · <span class="num">p_filt</span> (variance of the filtered log level; ±1.96√p_filt is the 95% band in log units). To run the filter on your own block data instead, the package and the paper's reference series are in the <a href="$REPO_URL" target="_blank" rel="noopener">repository</a>.</p>
    <p class="pnote"><b>Cite as:</b> Wüstenfeld, J. and Geldner, T. (2025). Bitcoin hashrate dynamics: filtering Poisson-luck and identifying responses to price and fees. <i>SSRN Working Paper 5786322</i>. <a href="$PAPER_URL" target="_blank" rel="noopener">papers.ssrn.com/sol3/papers.cfm?abstract_id=5786322</a></p>
  </section>

  <footer>
    <p class="fby">Bitcoin Hashrate Dashboard by <a href="$JAN_URL" target="_blank" rel="noopener">Dr. Jan Wüstenfeld</a> and <a href="$TEO_URL" target="_blank" rel="noopener">Teo Geldner</a></p>
    <p><a href="$PAPER_URL" target="_blank" rel="noopener">The paper (SSRN 5786322)</a> · <a href="$REPO_URL" target="_blank" rel="noopener">The code (GitHub)</a> · <a href="data/daily.csv" download>The series (CSV)</a></p>
    <p class="fts">Built $BUILT · data as of $AS_OF, the last complete UTC day · block data: Blockchair daily dumps, as in the paper · the two-sided line runs a week behind and its last days revise as new days arrive; everything else never changes.</p>
    <p class="fbuild">This page republishes no history: days stay exactly as first published, and each new day is added from the stored state of the day before.</p>
  </footer>
</main>

<script>
window.HF = $PAYLOAD;
</script>
<script>
"use strict";
var CO = {ink:'#0B0B0D', dim:'#5C616B', faint:'#9DA1A8', rule:'#ECEDEF', btc:'#F26B16', green:'#2C7A5B', amber:'#B0832E', red:'#C24A3A', rawgrey:'#C7CAD0', greyfill:'rgba(11,11,13,.06)', btcfill:'rgba(242,107,22,.10)'};
var LF = {family:"'JetBrains Mono', monospace", size:11, color:'#9DA1A8'};
var baseLayout = function(){ return {
  paper_bgcolor:'#fff', plot_bgcolor:'#fff', font:LF,
  margin:{l:60,r:20,t:12,b:44}, hovermode:'x unified', showlegend:false,
  xaxis:{gridcolor:'#F3F4F6', zerolinecolor:'#E2E3E6', linecolor:'#ECEDEF', hoverformat:'%d %b %Y', tickfont:{size:10}},
  yaxis:{gridcolor:'#F3F4F6', zerolinecolor:'#E2E3E6', linecolor:'#ECEDEF', tickfont:{size:10}}
};};
var CONFIG = {responsive:true, displayModeBar:'hover', displaylogo:false,
  modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d'],
  toImageButtonOptions:{format:'png', filename:'bitcoin-hashrate-dashboard', scale:2}};
var chartTitle = function(L, text, minT){
  L.title={text:text, font:{size:18, family:LF.family, color:CO.ink}, x:0.5, xanchor:'center', yanchor:'top', y:0.945};
  var legUp = !!(L.showlegend && L.legend && L.legend.orientation==='h');
  if(legUp) L.legend=Object.assign({}, L.legend, {y:1.02, yanchor:'bottom'});
  L.margin=Object.assign({}, L.margin, {t:Math.max((L.margin&&L.margin.t)||12, minT||(legUp?92:62))});
};
var watermark = function(L, inside){
  var a = {text:'hashrate-filter · Wüstenfeld &amp; Geldner (2025)', showarrow:false,
    font:{size:9, family:LF.family, color:CO.faint}, xref:'paper', yref:'paper'};
  if(inside){ a.x=0.995; a.xanchor='right'; a.y=0.97; a.yanchor='top'; }
  else { a.x=1; a.xanchor='right'; a.y=-0.14; a.yanchor='top'; a.yshift=-2; }
  L.annotations = (L.annotations||[]).concat([a]);
};
var H = window.HF, D = H.dates, N = D.length;
var legend = function(x){ return {orientation:'h', x:0, xanchor:'left', y:1.04, yanchor:'bottom', font:{size:10, family:LF.family, color:CO.dim}}; };
var DAYMS = 24*3600*1000;
var iso = function(ms){ return new Date(ms).toISOString().slice(0,16).replace('T',' '); };
var dms = function(i){ return new Date(D[i]+'T00:00:00Z').getTime(); };
// the two-sided line runs a week behind: index of its last value
var T_LAST = (function(){ for(var i=N-1;i>=0;i--) if(H.two[i]!=null) return i; return N-1; })();

/* ---------- 1 · main chart ---------- */
(function(){
  var bandHi=[], bandLo=[];
  for(var i=0;i<N;i++){ var b=1.96*Math.sqrt(H.pf[i]); bandHi.push(H.pure[i]*Math.exp(b)); bandLo.push(H.pure[i]*Math.exp(-b)); }
  var trRaw = {x:D, y:H.raw, type:'scatter', mode:'lines', name:'raw estimate', line:{color:CO.rawgrey, width:0.8}, visible:true};
  var trBand = {x:D.concat(D.slice().reverse()), y:bandHi.concat(bandLo.slice().reverse()), type:'scatter', mode:'lines', fill:'toself', fillcolor:CO.greyfill, line:{width:0}, name:'95% band', hoverinfo:'skip', visible:false};
  var trTwo = {x:D, y:H.two, type:'scatter', mode:'lines', name:'two-sided (runs a week behind)', line:{color:CO.green, width:1.1, dash:'dash'}, visible:true};
  var trPure = {x:D, y:H.pure, type:'scatter', mode:'lines', name:'filtered hashrate (the paper)', line:{color:CO.ink, width:1.7}, visible:true};
  var trToday = {x:[D[N-1]], y:[H.pure[N-1]], type:'scatter', mode:'markers', name:'today', showlegend:false,
    marker:{color:CO.btc, size:7, line:{color:'#fff', width:1.5}}, hoverinfo:'skip'};
  var L = baseLayout();
  L.showlegend = true; L.legend = legend();
  L.yaxis.type='log'; L.yaxis.title={text:'EH/s (log)', font:{size:10}};
  L.xaxis.rangeslider = {visible:true, thickness:0.12, bgcolor:'#FAFAFB'};
  var i0 = Math.max(0,N-1100);
  L.xaxis.range = [D[i0], D[N-1]];
  var lo=Infinity, hi=-Infinity;
  for(var i=i0;i<N;i++){ lo=Math.min(lo,H.raw[i]); hi=Math.max(hi,H.raw[i]); }
  L.yaxis.range = [Math.log10(lo*0.85), Math.log10(hi*1.18)];
  L.margin.b = 70;
  chartTitle(L, 'Bitcoin network hashrate — raw estimate, filter, two-sided', 92);
  watermark(L, true);
  Plotly.newPlot('ch-main', [trRaw, trBand, trTwo, trPure, trToday], L, CONFIG);
  var bind = function(idx, btn){
    document.getElementById(btn).addEventListener('click', function(){
      var vis = this.classList.toggle('on');
      Plotly.restyle('ch-main', {visible: vis}, [idx]);
    });
  };
  bind(2, 'btn-two'); bind(1, 'btn-band'); bind(0, 'btn-raw');
})();

/* ---------- 2 · luck bars ---------- */
(function(){
  var W=120, i0=N-W, x=D.slice(i0), y=[];
  for(var i=i0;i<N;i++) y.push(100*Math.log(H.raw[i]/H.pure[i]));
  var tr = {x:x, y:y, type:'bar', name:'raw minus filtered', marker:{color:CO.faint, opacity:0.55}};
  var trToday = {x:[x[x.length-1]], y:[y[y.length-1]], type:'scatter', mode:'markers', showlegend:false,
    marker:{color:CO.btc, size:7, line:{color:'#fff', width:1.5}}, hoverinfo:'skip'};
  var L = baseLayout();
  L.yaxis.title={text:'%', font:{size:10}}; L.yaxis.zeroline=true; L.yaxis.zerolinecolor='#D8DADE';
  L.xaxis.range=[iso(dms(i0)-DAYMS), iso(dms(N-1)+DAYMS)];
  chartTitle(L, 'The day\u2019s luck: raw minus filtered', 62);
  watermark(L);
  Plotly.newPlot('ch-luck', [tr, trToday], L, CONFIG);
})();

/* ---------- 2b · anatomy of the latest large move ---------- */
(function(){
  var gRaw=[null], gPure=[null], gMa3=[null];
  for(var i=1;i<N;i++){ gRaw.push(100*Math.log(H.raw[i]/H.raw[i-1])); gPure.push(100*Math.log(H.pure[i]/H.pure[i-1])); gMa3.push(100*Math.log(H.ma3[i]/H.ma3[i-1])); }
  var i0=H.ev.win[0], i1=H.ev.win[1], x=D.slice(i0,i1+1);
  var bar = function(g, name, color){ return {x:x, y:g.slice(i0,i1+1), type:'bar', name:name, marker:{color:color}}; };
  var L = baseLayout();
  L.barmode='group'; L.showlegend=true; L.legend=legend();
  L.yaxis.title={text:'% / day', font:{size:10}}; L.yaxis.zeroline=true; L.yaxis.zerolinecolor='#D8DADE';
  L.xaxis.range=[iso(dms(i0)-DAYMS), iso(dms(i1)+DAYMS)];
  L.shapes=[{type:'rect', xref:'x', yref:'paper', x0:iso(dms(H.ev.i)-DAYMS/2), x1:iso(dms(H.ev.i)+DAYMS/2), y0:0, y1:1,
    fillcolor:CO.btcfill, line:{width:0}, layer:'below'}];
  L.annotations=[{x:D[H.ev.i], y:1, xref:'x', yref:'paper', yanchor:'bottom', text:'the move',
    showarrow:false, font:{size:9, family:LF.family, color:CO.btc}}];
  chartTitle(L, 'Anatomy of the latest large move', 92);
  watermark(L);
  Plotly.newPlot('ch-event', [bar(gRaw,'raw estimate',CO.faint), bar(gMa3,'three-day moving average',CO.amber), bar(gPure,'filtered',CO.ink)], L, CONFIG);
})();

/* ---------- 3 · two-sided, last 180 days ---------- */
(function(){
  var W=180, i0=N-W, x=D.slice(i0);
  var trTwo = {x:x, y:H.two.slice(i0), type:'scatter', mode:'lines', name:'two-sided benchmark', line:{color:CO.green, width:1.4, dash:'dash'}};
  var trPure = {x:x, y:H.pure.slice(i0), type:'scatter', mode:'lines', name:'filtered', line:{color:CO.ink, width:1.6}};
  var trToday = {x:[D[N-1]], y:[H.pure[N-1]], type:'scatter', mode:'markers', showlegend:false,
    marker:{color:CO.btc, size:7, line:{color:'#fff', width:1.5}}, hoverinfo:'skip'};
  var L = baseLayout();
  L.showlegend=true; L.legend=legend();
  L.yaxis.title={text:'EH/s', font:{size:10}};
  var j0=Math.max(i0, T_LAST-6);
  L.shapes=[{type:'rect', xref:'x', yref:'paper', x0:D[j0], x1:D[T_LAST], y0:0, y1:1,
    fillcolor:CO.btcfill, line:{width:0}, layer:'below'}];
  L.annotations=[{x:D[T_LAST], y:0.03, xref:'x', yref:'paper', text:'still revising as new days arrive',
    showarrow:false, font:{size:9, family:LF.family, color:CO.btc}, xanchor:'right'}];
  chartTitle(L, 'The two-sided benchmark, last 180 days', 92);
  watermark(L);
  Plotly.newPlot('ch-two', [trTwo, trPure, trToday], L, CONFIG);
})();
</script>
</body>
</html>
""")


def write_site(
    df: pd.DataFrame,
    meta: dict,
    out: Path,
    *,
    site_url: str,
    repo_root: Path,
    css_src: Path,
    tail_shift: float,
    fetched_utc: str,
) -> None:
    st = compute_stats(df, meta, tail_shift)
    payload = build_payload(df, st)
    out = Path(out)

    ev_date = st["ev_date"]
    # the event answer line, worded by what the move turned out to be
    move = st["ev_graw"]
    direction = "jumped" if move > 0 else "dropped"
    rev = st["ev_next_raw"]  # share of the move undone the next day (negative: reversal)
    rev_txt = ""
    if rev < -25:
        back = "recouped" if move < 0 else "gave back"
        rev_txt = f" and {back} {abs(rev):.0f}% of it the next day"
    ma = [fmt_pct(v) for v in st["ev_gma3"]]
    graw, gpure = fmt_pct(move), fmt_pct(st["ev_gpure"])
    held = f"{st['ev_held5']:.0f}"
    if st["ev_kind"] == "luck":
        ev_answer = (
            f"On <b>{fmt_date(ev_date)}</b>, the raw estimate {direction} <b>{graw}</b>{rev_txt} — mostly block luck: "
            f"five days later the filtered level held {held}% of the move. The filter read <b>{gpure}</b> on the day; "
            f"the three-day average spread it over three (<b>{ma[0]}</b> on the day, then <b>{ma[1]}</b> and <b>{ma[2]}</b>)."
        )
    else:
        ev_answer = (
            f"On <b>{fmt_date(ev_date)}</b>, the raw estimate {direction} <b>{graw}</b> and held it: "
            f"five days later the filtered level kept {held}% of the move. The filter recorded <b>{gpure}</b> on the day "
            f"itself — the three-day average needed three days (<b>{ma[0]}</b> on the day, then <b>{ma[1]}</b> and <b>{ma[2]}</b>)."
        )
    # the two-sided answer line: honest in quiet and in fast-moving weeks
    two_tail7 = st["two_tail7"]
    if two_tail7 < 1.0:
        two_answer = (
            f"Not much — but you wait a week for it. The two-sided version also reads the following days, "
            f"so it runs a week behind; across its last week it sits within <b>{two_tail7:.2f}%</b> of the daily filter."
        )
    else:
        two_answer = (
            f"In a fast move like the current one, up to <b>{two_tail7:.2f}%</b> — and you wait a week for it. "
            f"The two-sided version also reads the following days, so it runs a week behind; the gap is where "
            f"the daily filter still lags the move."
        )
    description = (
        "Bitcoin's network hashrate without the luck: the raw daily estimate against the filtered series of "
        "Wüstenfeld and Geldner (2025), updated daily. See how much of the raw move is block luck, how the "
        "three-day moving average smears it, and the two-sided benchmark."
    )

    values = {
        "DESCRIPTION": description,
        "SITE_URL": site_url,
        "PAPER_URL": PAPER_URL,
        "REPO_URL": REPO_URL,
        "JAN_URL": JAN_URL,
        "TEO_URL": TEO_URL,
        "AS_OF": fmt_date(st["as_of"]),
        "TWO_LAST": fmt_date(st["two_last"]),
        "BUILT": pd.Timestamp.utcnow().strftime("%d %b %Y %H:%M UTC") + (
            f" · block data fetched {fetched_utc}" if fetched_utc else ""
        ),
        "PURE_EH": f"{st['pure_eh']:,.1f}",
        "RAW_EH": f"{st['raw_eh']:,.1f}",
        "D1": fmt_pct(st["d1"]),
        "D7": fmt_pct(st["d7"]),
        "N_LAST": f"{st['n_last']}",
        "LUCK_LAST": fmt_pct(st["luck_last"]),
        "LUCK_LAST_ABS": f"{abs(st['luck_last']):.1f}%",
        "LUCK_SIDE": "above" if st["luck_last"] >= 0 else "below",
        "LUCK_16": f"{st['luck_pure_16']:.1f}",
        "LUCK_TWO_FULL": f"{st['luck_two_full']:.1f}",
        "SD_RAW_16": f"{st['sd_raw_16']:.1f}",
        "SD_PURE_16": f"{st['sd_pure_16']:.1f}",
        "SD_RAW_FULL": f"{st['sd_raw_full']:.1f}",
        "SD_PURE_FULL": f"{st['sd_pure_full']:.1f}",
        "SCORE_PURE": f"{st['score_pure']:.3f}",
        "SCORE_MA3": f"{st['score_ma3']:.3f}",
        "SCORE_RAW": f"{st['score_raw']:.3f}",
        "SCORE_YEARS": f"{st['score_years'][0]}–{st['score_years'][1]}",
        "SCORE_N": f"{st['score_n']:,}",
        "MAE_PURE": f"{st['mae_pure']:.1f}",
        "MAE_RAW": f"{st['mae_raw']:.1f}",
        "EV_DATE": fmt_date(ev_date),
        "EV_GRAW": fmt_pct(st["ev_graw"]),
        "EV_GPURE": fmt_pct(st["ev_gpure"]),
        "EV_GMA3_0": fmt_pct(st["ev_gma3"][0]),
        "EV_GMA3_1": fmt_pct(st["ev_gma3"][1]),
        "EV_GMA3_2": fmt_pct(st["ev_gma3"][2]),
        "EV_HELD5": held,
        "EV_ANSWER": ev_answer,
        "TWO_TAIL7": f"{st['two_tail7']:.2f}%",
        "TWO_ANSWER": two_answer,
        "TWO_RMS_FULL": f"{st['two_rms_full']:.2f}%",
        "TWO_RMS_90": f"{st['two_rms_90']:.2f}%",
        "TAIL_SHIFT": f"{st['tail_shift']:.2f}%",
        "Q_QUIET": f"{st['q_quiet']:.3e}",
        "PRIOR": f"{st['prior']:.4f}",
        "PAYLOAD": payload,
    }

    html = TEMPLATE.substitute(values)
    (out / "site").mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html)
    shutil.copy(css_src, out / "site" / "dashboard.css")
    og_src = repo_root / "examples" / "figures" / "filter_overview.png"
    if og_src.exists():
        shutil.copy(og_src, out / "og-card.png")
