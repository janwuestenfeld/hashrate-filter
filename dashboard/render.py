"""Render the dashboard page from the stored daily series.

Live part: the latest filtered hashrate and raw estimate, and one zoomable
chart of the raw estimate, our filter, the two-sided version and the
moving averages. Evidence part (fixed): the typical daily change year by
year, the 2021 mining ban, and how our filter works. Then the download.

Numbers labelled as the paper's are literals from numbers.tex (one place,
PAPER below). Every other number is computed from the series at build time.
"""

import json
import shutil
import string
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import poisson

from daily_step import Q_JUMP

PAPER_URL = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5786322"
REPO_URL = "https://github.com/janwuestenfeld/hashrate-filter"
JAN_URL = "https://janwuestenfeld.github.io/wuestenfeld.eu/"
TEO_URL = "https://www.linkedin.com/in/teo-geldner-96802bb7/"

# citation (as in CITATION.cff)
CITE_YEAR = 2025
CITE_TITLE = "Bitcoin hashrate dynamics: filtering Poisson-luck and identifying responses to price and fees"
CITE_TEXT = (f"Wüstenfeld, J. and Geldner, T. ({CITE_YEAR}). {CITE_TITLE}. "
             f"SSRN Working Paper 5786322.")
CITE_BIB = (
    "@unpublished{wuestenfeld" + str(CITE_YEAR) + "hashrate,\n"
    "  author = {W{\\\"u}stenfeld, Jan and Geldner, Teo},\n"
    "  title  = {Bitcoin hashrate dynamics: filtering {P}oisson-luck and identifying responses to price and fees},\n"
    f"  year   = {{{CITE_YEAR}}},\n"
    "  note   = {SSRN Working Paper 5786322},\n"
    f"  url    = {{{PAPER_URL}}}\n"
    "}"
)
MARK = "Wüstenfeld & Geldner · SSRN 5786322"

# the paper's numbers (numbers.tex)
PAPER = {
    "sd_raw": "11.8%",          # \sdrawh, daily growth of the raw estimate
    "sd_filter": "3.1%",        # \sdvsixe (3.10%)
    "noise_share": "96%",       # \noisesharefull, luck share of the variance of raw daily growth
    "growth_unc": 3.7,          # \filtgrowthsd, sd of the filter's daily-growth error, %
    "luck_full": "93.6%",       # \luckremvsixe, full sample
    "luck_2016": "95.6%",       # \luckremvsixemature, from 2016
    "gain_day": "31%",          # \gainquiet
    "gain_week": "95%",         # \stepweek
    "ban": "97%",               # \banvsixe
    "ma_acf3": "−0.48",         # \maacfthree
    "cov_quiet": "96%", "cov_jump1": "9%", "cov_jump2": "44%",   # \bandcov*
    "luck_two": "99.1%",        # \luckremtwosided
}
TWO_LAG = 7

EH = 1e18
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def fmt_date(d) -> str:
    d = pd.Timestamp(d)
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def fmt_pct(x, dp=1, sign=True) -> str:
    s = "+" if (sign and x > 0) else ("−" if x < 0 else "")
    return f"{s}{abs(x):.{dp}f}%"


def fmt_eh(x) -> str:
    return f"{x:,.0f}"


def neg(x, dp=2) -> str:
    return f"{x:.{dp}f}".replace("-", "−")


def expected_blocks(h, d):
    """A rate h (H/s) against difficulty d expects h x 86400 / (d x 2^32) blocks a day."""
    return h * 86400.0 / (d * 2.0 ** 32)


# ------------------------------------------------------------------ stats
def ordinary_gain(q_quiet: float, prior: float) -> float:
    """Steady-state gain on a day whose arrival-time weight sits at the prior."""
    q = (1.0 - prior) * q_quiet + prior * Q_JUMP
    p = q
    for _ in range(5000):
        pt = p + q
        k = pt / (pt + 1.0 / 144.0)
        p = (1.0 - k) * pt
    return k


def acf(x: np.ndarray, k: int) -> float:
    x = x[np.isfinite(x)]
    a, b = x[:-k] - x[:-k].mean(), x[k:] - x[k:].mean()
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def hourly_counts(blocks, day):
    """Blocks of the day, assigned to the day and to each hour by median time past."""
    if blocks is None or len(blocks) == 0:
        return None
    b = blocks if isinstance(blocks, pd.DataFrame) else pd.DataFrame(blocks)
    mt = pd.to_datetime(b["median_time"])
    sel = mt.dt.normalize() == pd.Timestamp(day)
    if not sel.any():
        return None
    hrs = ((mt[sel] - pd.Timestamp(day)) / pd.Timedelta(hours=1)).astype(float).clip(0, 23.999).astype(int)
    return np.bincount(hrs.values, minlength=24).tolist()


def compute_stats(df: pd.DataFrame, meta: dict) -> tuple:
    d = df.copy()
    d["ma7"] = d["raw"].rolling(7, min_periods=1).mean()
    n = len(d)
    st = {}
    last = d.iloc[-1]
    st["as_of"] = last["date"]
    st["pure"] = last["pure"] / EH
    b = 1.96 * np.sqrt(last["p_filt"])
    st["band_lo"], st["band_hi"] = st["pure"] * np.exp(-b), st["pure"] * np.exp(b)
    st["raw"] = last["raw"] / EH
    st["n"] = int(last["n_blocks"])
    lam = expected_blocks(last["pure"], last["difficulty"])
    st["lam"] = lam
    st["n_lo"], st["n_hi"] = int(poisson.ppf(0.025, lam)), int(poisson.ppf(0.975, lam))
    st["h_lo"], st["h_hi"] = int(poisson.ppf(0.025, lam / 24)), int(poisson.ppf(0.975, lam / 24))
    lp = np.log(d["pure"].values)
    st["d1"] = 100 * (lp[-1] - lp[-2])
    st["d7"] = 100 * (lp[-1] - lp[-8])
    st["w_last"] = float(last["weight"])
    st["w_typ"] = float(d["weight"].median())
    st["w_n"] = int((d["weight"] > 0.7).sum())

    g_all = np.log(d[["raw", "ma3", "ma7", "pure", "pure_two_sided"]]).diff()
    st["sd"] = {c: 100 * float(g_all[c].std()) for c in ["raw", "ma3", "ma7", "pure", "pure_two_sided"]}

    # correlation of daily growth at lags 1-10, from 2016
    g = g_all[d["date"] >= pd.Timestamp("2016-01-01")]
    st["acf"] = {c: [round(acf(g[c].values, k), 3) for k in range(1, 11)] for c in ["raw", "ma3", "ma7", "pure"]}
    st["acf_raw1"] = st["acf"]["raw"][0]
    st["acf_ma3_3"] = st["acf"]["ma3"][2]
    st["acf_ma7_7"] = st["acf"]["ma7"][6]
    st["acf_pure_max"] = max(st["acf"]["pure"], key=abs)
    k = ordinary_gain(float(meta["q_quiet"]), float(meta["prior"]))
    st["gain"] = k
    st["past_share"] = k / 2.0   # share of the luck in a filter move that comes from earlier days

    # one lucky day at the top of today's luck range, true rate unchanged
    eps = float(np.log(168 / 144))
    st["lk"] = {"raw": 100 * eps, "m3": 100 * np.log((2 + np.exp(eps)) / 3),
                "m7": 100 * np.log((6 + np.exp(eps)) / 7), "f0": 100 * k * eps, "f1": -100 * k * k * eps}

    # the dated event on the main chart: the largest raw move of the last 30 days
    # whose exit from the three-day average three days later is visible
    gr = 100 * g_all["raw"].values
    g3 = 100 * g_all["ma3"].values
    ev, best = -1, 0.0
    for i in range(max(1, n - 30), n - 3):
        if np.isfinite(gr[i]) and np.isfinite(g3[i + 3]) and gr[i] * g3[i + 3] < 0 \
                and abs(g3[i + 3]) >= abs(gr[i]) / 6 and abs(gr[i]) > best:
            ev, best = i, abs(gr[i])
    st["ev"] = ev
    if ev > 0:
        st["ev_d"], st["ev_g"], st["ev_g3"] = d["date"].iloc[ev], gr[ev], g3[ev + 3]

    # the 2021 ban
    w = d[(d["date"] >= "2021-05-01") & (d["date"] <= "2021-07-31")].set_index("date")
    st["ban_hi"], st["ban_hi_d"] = w["pure"].loc[:"2021-06-15"].max() / EH, w["pure"].loc[:"2021-06-15"].idxmax()
    st["ban_lo"], st["ban_lo_d"] = w["pure"].loc["2021-06-15":].min() / EH, w["pure"].loc["2021-06-15":].idxmin()
    st["ban_lo7_d"] = w["ma7"].loc["2021-06-15":].idxmin()
    st["ban_lag7"] = int((st["ban_lo7_d"] - st["ban_lo_d"]).days)
    half = (st["ban_hi"] + st["ban_lo"]) / 2 * EH
    after = w.loc[st["ban_hi_d"]:]
    st["ban_half_f"] = after.index[after["pure"] <= half][0]
    st["ban_half_7"] = after.index[after["ma7"] <= half][0]
    st["ban_half_lag"] = int((st["ban_half_7"] - st["ban_half_f"]).days)
    gap = (after["ma7"] - after["pure"]).loc[:st["ban_lo_d"]]
    st["ban_gap_d"] = gap.idxmax()
    st["ban_gap_7"], st["ban_gap_f"] = after["ma7"].loc[st["ban_gap_d"]] / EH, after["pure"].loc[st["ban_gap_d"]] / EH
    t9 = int(np.flatnonzero(d["date"].values == np.datetime64("2021-06-09"))[0])
    w9 = float(d["weight"].iloc[t9])
    q9 = (1.0 - w9) * float(meta["q_quiet"]) + w9 * Q_JUMP
    st["ban_flag_k"] = 1.0 - float(d["p_filt"].iloc[t9]) / (float(d["p_filt"].iloc[t9 - 1]) + q9)
    return st, d


def payload(d: pd.DataFrame, st: dict, hours) -> str:
    def col(c):
        return [None if not np.isfinite(v) else float(f"{float(v) / EH:.6g}") for v in d[c].values]
    b = 1.96 * np.sqrt(d["p_filt"].values)
    p = {
        "dates": d["date"].dt.strftime("%Y-%m-%d").tolist(),
        "raw": col("raw"), "pure": col("pure"), "ma3": col("ma3"), "ma7": col("ma7"), "two": col("pure_two_sided"),
        "acf": st["acf"], "lag": TWO_LAG,
        "hours": hours, "lam": round(st["lam"], 2), "hlo": st["h_lo"], "hhi": st["h_hi"],
        "ev": int(st["ev"]), "gain": round(st["gain"], 4), "nhi": st["n_hi"],
        "asof": str(st["as_of"].date()), "asof_txt": fmt_date(st["as_of"]),
        "mark": MARK,
        "ban": {"halfF": st["ban_half_f"].strftime("%-d %b"), "half7": st["ban_half_7"].strftime("%-d %b"),
                "half": round((st["ban_hi"] + st["ban_lo"]) / 2, 1), "gapD": st["ban_gap_d"].strftime("%Y-%m-%d"), "gap7": round(st["ban_gap_7"], 1), "gapF": round(st["ban_gap_f"], 1)},
    }
    return json.dumps(p, separators=(",", ":"))


# ------------------------------------------------------------------ share image
def share_images(d: pd.DataFrame, st: dict, out: Path, site_url: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
    except Exception:
        return False
    have = {f.name for f in font_manager.fontManager.ttflist}
    fam = next((f for f in ("Geist", "Inter") if f in have), "DejaVu Sans")
    w = d.iloc[-180:]
    short_url = site_url.replace("https://", "").rstrip("/")
    bw = 1.96 * np.sqrt(w["p_filt"].values)
    with plt.rc_context({"font.family": fam}):
        for name, size in [("share.png", (12, 6.75)), ("og-card.png", (12, 6.3))]:
            fig = plt.figure(figsize=size, dpi=100)
            fig.patch.set_facecolor("white")
            fig.text(0.05, 0.93, "Bitcoin's daily hashrate moves are mostly luck.", fontsize=20,
                     fontweight="bold", color="#0B0B0D", va="top")
            fig.text(0.05, 0.86, "We built a filter that removes it, using the protocol's own rules.",
                     fontsize=20, fontweight="bold", color="#F26B16", va="top")
            ax = fig.add_axes([0.07, 0.15, 0.80, 0.55])
            ax.scatter(w["date"], w["raw"] / EH, s=10, color="#AEB2BA", label="raw estimate", zorder=2)
            ax.plot(w["date"], w["pure"] / EH, color="#F26B16", lw=3, label="our filter", zorder=3)
            ax.annotate(f"{st['pure']:,.0f} EH/s\n{fmt_date(st['as_of'])}", xy=(w["date"].iloc[-1], st["pure"]),
                        xytext=(10, 0), textcoords="offset points", va="center", fontsize=14,
                        color="#C2622A", fontweight="bold", annotation_clip=False)
            ax.set_ylabel("EH/s", fontsize=13)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.tick_params(labelsize=12)
            for s in ["top", "right"]:
                ax.spines[s].set_visible(False)
            ax.legend(frameon=False, fontsize=12, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
            fig.text(0.05, 0.035, f"{MARK}  ·  {short_url}", fontsize=12, color="#5C616B")
            fig.savefig(out / name, dpi=100)
            plt.close(fig)
    return True


# ------------------------------------------------------------------ html
TEMPLATE = string.Template(r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bitcoin Hashrate Filter: the daily hashrate without the block luck</title>
<meta name="description" content="$DESC">
<link rel="canonical" href="$SITE_URL">
<meta property="og:type" content="website">
<meta property="og:url" content="$SITE_URL">
<meta property="og:title" content="Bitcoin Hashrate Filter">
<meta property="og:description" content="$DESC">
<meta property="og:image" content="${SITE_URL}og-card.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="Bitcoin Hashrate Filter">
<meta name="twitter:description" content="$DESC">
<meta name="twitter:image" content="${SITE_URL}og-card.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js" integrity="sha384-7TVmlZWH60iKX5Uk7lSvQhjtcgw2tkFjuwLcXoRSR4zXTyWFJRm9aPAguMh7CIra" crossorigin="anonymous" onerror="__plotlyFallback(0)"></script>
<script>
(function(){
  var MIRRORS = ["https://cdn.jsdelivr.net/npm/plotly.js@2.32.0/dist/plotly.min.js","https://unpkg.com/plotly.js@2.32.0/dist/plotly.min.js"];
  window.__plotlyFallback = function(i){
    if (typeof Plotly !== 'undefined') return;
    if (i >= MIRRORS.length) return;
    var s = document.createElement('script'); s.src = MIRRORS[i]; s.crossOrigin = 'anonymous';
    s.onerror = function(){ __plotlyFallback(i + 1); }; s.onload = function(){ if (window.__draw) window.__draw(); };
    document.head.appendChild(s);
  };
})();
</script>
<style>$CSS</style>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'%3E%3Ccircle cx='8' cy='8' r='7' fill='%23F26B16'/%3E%3C/svg%3E">
<script data-goatcounter="https://hashrate-filter.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
</head>
<body>

<main>
<section class="hero">
  <div class="wrap">
    <h1>Bitcoin Hashrate <span class="ac">Filter</span></h1>
    <div class="byline">by <a href="$JAN_URL" target="_blank" rel="noopener">Dr. Jan Wüstenfeld<span class="ext"> (homepage&nbsp;↗)</span></a> and <a href="$TEO_URL" target="_blank" rel="noopener">Teo Geldner<span class="ext"> (LinkedIn&nbsp;↗)</span></a> · <a href="$PAPER_URL" target="_blank" rel="noopener" data-goatcounter-click="byline-paper">Paper<span class="ext"> (SSRN&nbsp;↗)</span></a> · <a href="$REPO_URL" target="_blank" rel="noopener" data-goatcounter-click="byline-code">Code<span class="ext"> (GitHub&nbsp;↗)</span></a></div>
    <div class="freshness">Updated daily from public block data · last updated <span>$UPDATED</span></div>
    <div class="tagline">Bitcoin's daily hashrate moves are mostly luck. We built a filter that removes it, using the protocol's own rules.</div>
    <p class="lede">Every daily hashrate figure comes from the number of blocks found that day and the difficulty. Blocks arrive at random, as a Poisson process, so the raw estimate moves by about $SD_RAW from one day to the next even when the hashrate stays the same. Moving averages damp these swings, but when a lucky day leaves their window the average falls although the hashrate did not change. We are the first to remove the luck without a smoothing window, using the size of Poisson luck at each day's block count and the times at which the day's blocks arrive. Since 2016 our filter removes $LUCK_2016 of this luck.</p>
    <div class="tiles two" id="live">
      <div class="tile main">
        <div class="k">Our filter, $AS_OF</div>
        <div class="v">$PURE <span class="u">EH/s</span></div>
      </div>
      <div class="tile">
        <div class="k">Raw estimate, same day</div>
        <div class="v">$RAW <span class="u">EH/s</span></div>
        <div class="s">from $N blocks</div>
      </div>
    </div>
  </div>
</section>

<section class="sec">
  <div class="wrap">
    <h2>Our filter follows the hashrate without the daily luck of the raw estimate</h2>
    <p class="lede">Grey dots are the raw estimate, the orange line is our filter and the dotted line the 3-day average. The 7-day average is hidden, and you can show it by clicking it in the legend. Drag across the chart to zoom, and use the bar below it to move through the history since 2011.</p>
    <figure class="fig"><div id="ch-main" class="chart tall"></div></figure>
    <p class="note">The green line is the two-sided version of our filter. It also uses the following seven days, so its last seven days (shaded, dotted) are not final and change as new days come in. Use the orange line for real-time work and the green line to look back at history. All other lines never change once published.</p>
  </div>
</section>

<section class="sec alt" id="evidence">
  <div class="wrap">
    <h2>Most of the raw estimate's daily movement is luck</h2>
    <div class="nums"><div><span class="v grey">$NOISE_SHARE</span><span class="k">of the variation in the raw estimate's daily changes is luck, 2011 to July 2026</span></div><div><span class="v ac">$LUCK_FULL</span><span class="k">of that luck our filter removes, 2011 to July 2026 ($LUCK_2016 since 2016)</span></div></div>
    <p class="lede">In every year the raw estimate moves close to what Poisson luck alone produces (dashed line), about $SD_RAW a day. Our filter (orange bars) moves about $SD_FILTER a day, 2011 to July 2026.</p>
    <figure class="fig"><div id="ch-years" class="chart mid"></div></figure>

    <h2 class="gap">In the 2021 China mining ban our filter showed half of the fall $BAN_HALF_LAG days before the 7-day average</h2>
    <p class="lede">Our filter showed half of the fall on $BAN_HALF_F and the 7-day average on $BAN_HALF_7, $BAN_HALF_LAG days later. In May and June 2021 China banned bitcoin mining, and our filter fell by more than half, from $BAN_HI EH/s on $BAN_HI_D to $BAN_LO EH/s on $BAN_LO_D. On $BAN_GAP_D the 7-day average was still at $BAN_GAP_7 EH/s, against $BAN_GAP_F EH/s in our filter, because the average still included the six days before, when the hashrate was higher. On 9 Jun 2021 the day's block times pointed to a real change, so our filter moved $BAN_FLAG_K of the way to that day's raw estimate instead of the usual $GAIN.</p>
    <figure class="fig"><div id="ch-ban" class="chart mid"></div></figure>

    <h2 class="gap">How our filter works</h2>
    <p class="lede">Bitcoin's protocol adjusts the difficulty every 2,016 blocks, about every two weeks, so that the network finds on average 144 blocks a day. Because blocks arrive as a Poisson process, the chance variation of a day's count follows from the expected count itself. On a 144-block day its standard deviation is the square root of 144, which is 12 blocks or 8.3%. A change from one day to the next combines the luck of two days, which gives the $SD_RAW of the raw estimate. Each day our filter moves part of the way from its previous estimate toward the day's raw estimate. When the day's blocks come faster in one part of the day than in the other, which points to a real change in hashrate, our filter moves further.</p>
    <p class="lede">The two-sided version of our filter also uses the following seven days and removes $P_TWO of the luck, 2011 to July 2026. It is for looking back at history. It cannot be used in real time, and it should not feed a model of how miners respond to price and fees, because each of its values already contains the following week.</p>
  </div>
</section>

<section class="sec" id="data">
  <div class="wrap">
    <h2>Download the series</h2>
    <div class="cols">
      <div>
        <p class="lede">We add one row per UTC day, from 1 January 2011, once the day is complete. Published values never change, except the last seven days of the two-sided version of our filter.</p>
        <ul class="dl">
          <li><a href="data/daily.csv" download data-goatcounter-click="data-csv">data/daily.csv</a> <span>the full daily series</span></li>
          <li><a href="data/latest.json" data-goatcounter-click="data-json">data/latest.json</a> <span>values for the latest complete day</span></li>
          <li><a href="data/README.txt" data-goatcounter-click="data-readme">data/README.txt</a> <span>what each column means</span></li>
          <li><a href="share.png" download="hashrate-filter_$AS_OF_ISO.png" data-goatcounter-click="data-png">share.png</a> <span>the latest chart as an image</span></li>
        </ul>
      </div>
      <div>
        <p class="lede">If you use the series, please cite our paper.</p>
        <div class="cite"><p id="cite-text">$CITE_TEXT</p><button class="copy" data-copy="cite-text" data-goatcounter-click="copy-cite">Copy</button></div>
        <div class="cite"><pre id="cite-bib">$CITE_BIB</pre><button class="copy" data-copy="cite-bib" data-goatcounter-click="copy-bib">Copy BibTeX</button></div>
      </div>
    </div>
  </div>
</section>
</main>

<footer>
  <div class="wrap">
    <p><b>Bitcoin Hashrate Filter</b> by <a href="$JAN_URL" target="_blank" rel="noopener">Dr. Jan Wüstenfeld</a> and <a href="$TEO_URL" target="_blank" rel="noopener">Teo Geldner</a> · <a href="$PAPER_URL" target="_blank" rel="noopener" data-goatcounter-click="foot-paper">Paper (SSRN 5786322)</a> · <a href="$REPO_URL" target="_blank" rel="noopener" data-goatcounter-click="foot-code">Code (GitHub)</a> · <a href="data/daily.csv" download data-goatcounter-click="foot-csv">Data (CSV)</a></p>
    <p class="fine">Last updated $UPDATED</p>
  </div>
</footer>

<script>window.HF = $PAYLOAD;</script>
<script>
"use strict";
window.__draw = function(){
if (typeof Plotly === 'undefined' || window.__drawn) return; window.__drawn = true;
var H = window.HF, D = H.dates, N = D.length;
var C = {ink:'#0B0B0D', dim:'#5C616B', grid:'#F3F4F6', raw:'#AEB2BA', ma3:'#3E4350', ma7:'#8A8F99', two:'#2C7A5B', f:'#F26B16', band:'rgba(242,107,22,.14)', pre:'rgba(44,122,91,.08)'};
var F = {family:"'JetBrains Mono', monospace", size:11, color:'#5C616B'};
var narrow = window.innerWidth < 640;
var MARKTXT = narrow ? H.mark : H.mark + ' · janwuestenfeld.github.io/hashrate-filter';
var mark = function(y){ return {text:MARKTXT, showarrow:false, xref:'paper', yref:'paper', x:1, y:(y||0), xanchor:'right', yanchor:'bottom',
  xshift:-4, yshift:4, bgcolor:'rgba(255,255,255,0.88)', borderpad:3, font:{size:narrow?11:12, color:'#3E4350', family:F.family}}; };
var base = function(){ return {paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)', font:F, hovermode:'x unified', showlegend:true,
  hoverlabel:{font:{family:F.family, size:12}},
  legend:{orientation:'h', x:1, xanchor:'right', y:1.0, yanchor:'bottom', font:{size:11}, itemclick:false, itemdoubleclick:false},
  margin:{l:56, r:12, t:40, b:40},
  xaxis:{gridcolor:C.grid, linecolor:'#DADCE0', tickfont:{size:11}}, yaxis:{gridcolor:C.grid, zerolinecolor:'#D5D7DB', tickfont:{size:11}}}; };
var head = function(L, title, short){
  if (narrow){ L.title = {text:''}; L.legend = Object.assign(L.legend, {x:0, xanchor:'left', y:1.0, yanchor:'bottom', entrywidth:0.5, entrywidthmode:'fraction'}); L.margin.t = 74; }
  else { L.title = {text:title, x:0, xanchor:'left', y:0.995, yanchor:'top', font:{size:14, color:C.ink, family:F.family}}; }
};
var cfg = function(name){ return {responsive:true, displaylogo:false, displayModeBar:narrow?false:'hover', modeBarButtonsToRemove:['lasso2d','select2d','autoScale2d','toggleSpikelines'],
  toImageButtonOptions:{format:'png', filename:'hashrate-filter_'+name+'_'+H.asof, scale:2, width:1200, height:675}}; };
var g = function(a){ var o=[null]; for(var i=1;i<a.length;i++) o.push(a[i]&&a[i-1] ? 100*Math.log(a[i]/a[i-1]) : null); return o; };
var GR = g(H.raw), GP = g(H.pure), G3 = g(H.ma3), G7 = g(H.ma7);
var DAY = 864e5, t = function(i){ return new Date(D[i]+'T00:00:00Z').getTime(); }, iso = function(ms){ return new Date(ms).toISOString().slice(0,10); };
var P0 = N - H.lag;   // first preliminary day of the two-sided line

/* ---------- main chart: zoomable, legend toggles ---------- */
(function(){
  var twoFull = H.two.map(function(v,i){ return i<P0 ? v : null; });
  var twoPre = H.two.map(function(v,i){ return i>=P0-1 ? v : null; });
  var T = [
    {x:D, y:H.raw, type:'scatter', mode:'markers', marker:{size:5, color:C.raw}, name:'raw estimate', hovertemplate:'%{y:,.0f} EH/s'},
    {x:D, y:H.ma7, type:'scatter', mode:'lines', line:{color:C.ma7, width:1.6, dash:'dash'}, name:'7-day average', visible:'legendonly', hovertemplate:'%{y:,.0f} EH/s'},
    {x:D, y:H.ma3, type:'scatter', mode:'lines', line:{color:C.ma3, width:1.6, dash:'dot'}, name:'3-day average', hovertemplate:'%{y:,.0f} EH/s'},
    {x:D, y:twoFull, type:'scatter', mode:'lines', line:{color:C.two, width:1.8}, name:'two-sided version of our filter', legendgroup:'two', hovertemplate:'%{y:,.0f} EH/s'},
    {x:D, y:twoPre, type:'scatter', mode:'lines', line:{color:C.two, width:1.8, dash:'dot'}, name:'two-sided version of our filter, not final', legendgroup:'two', showlegend:false, hoverinfo:'skip'},
    {x:D, y:H.two.map(function(v,i){ return i>=P0 ? v : null; }), type:'scatter', mode:'markers', marker:{size:4, color:C.two}, name:'two-sided version of our filter, not final', legendgroup:'two', showlegend:false, hovertemplate:'%{y:,.0f} EH/s, not final yet'},
    {x:D, y:H.pure, type:'scatter', mode:'lines', line:{color:C.f, width:2.8}, name:'our filter', hovertemplate:'%{y:,.0f} EH/s'}];
  var L = base();
  L.shapes = [{type:'rect', xref:'x', yref:'paper', x0:D[P0], x1:D[N-1], y0:0, y1:1, fillcolor:C.pre, line:{width:0}, layer:'below'}];
  L.legend.itemclick = 'toggle'; L.legend.itemdoubleclick = 'toggleothers';
  L.dragmode = 'zoom';
  var i0 = Math.max(0, N-180);
  var yr = function(a0, a1){ var lo=Infinity, hi=-Infinity;
    for (var i=0;i<N;i++){ if (D[i]<a0 || D[i]>a1) continue; lo=Math.min(lo,H.raw[i]||Infinity,H.pure[i]); hi=Math.max(hi,H.raw[i]||-Infinity,H.pure[i]); }
    return [lo*0.96, hi*1.03]; };
  L.xaxis = Object.assign(L.xaxis, {range:[D[i0], D[N-1]], rangeslider:{visible:true, thickness:0.09, bgcolor:'#F7F7F8'}, type:'date'});
  L.yaxis = Object.assign(L.yaxis, {title:{text:'EH/s', font:{size:11}}, range:yr(D[i0], D[N-1]), fixedrange:false});
  L.margin.b = 30;
  head(L, 'Bitcoin network hashrate to '+H.asof_txt);
  L.annotations = [mark(0.12), {x:D[N-1], y:1, xref:'x', yref:'paper', xanchor:'right', yanchor:'top', showarrow:false, text:'two-sided version: last '+H.lag+' days not final', font:{size:11, color:C.two}}];
  Plotly.newPlot('ch-main', T, L, cfg('main'));
  var el = document.getElementById('ch-main'), busy = false;
  el.on('plotly_relayout', function(ev){
    if (busy) return;
    var r = el.layout.xaxis.range; if (!r) return;
    var a0 = String(r[0]).slice(0,10), a1 = String(r[1]).slice(0,10);
    busy = true; Plotly.relayout(el, {'yaxis.range': yr(a0, a1)}).then(function(){ busy = false; });
  });
})();

/* ---------- the typical daily move, year by year ---------- */
(function(){
  var yrs=[], acc={};
  for(var i=1;i<N;i++){ if(GR[i]===null || GP[i]===null) continue; var y=D[i].slice(0,4);
    if(!acc[y]){ acc[y]={n:0,a:0,aa:0,b:0,bb:0}; yrs.push(y); } var o=acc[y]; o.n++; o.a+=GR[i]; o.aa+=GR[i]*GR[i]; o.b+=GP[i]; o.bb+=GP[i]*GP[i]; }
  var sd=function(s,ss,n){ return Math.sqrt((ss-s*s/n)/(n-1)); };
  var sr=yrs.map(function(y){ var o=acc[y]; return sd(o.a,o.aa,o.n); }), sp=yrs.map(function(y){ var o=acc[y]; return sd(o.b,o.bb,o.n); });
  var GT=g(H.two), at={}; for(var i=1;i<N;i++){ if(GT[i]===null) continue; var y=D[i].slice(0,4); if(!at[y]) at[y]={n:0,a:0,aa:0}; at[y].n++; at[y].a+=GT[i]; at[y].aa+=GT[i]*GT[i]; }
  var stw=yrs.map(function(y){ var o=at[y]; return o?sd(o.a,o.aa,o.n):null; });
  var luck=100*Math.sqrt(2/144);
  var T=[{x:yrs, y:sr, type:'bar', name:'raw estimate', marker:{color:C.raw}, hovertemplate:'%{y:.1f}%'},
         {x:yrs, y:sp, type:'bar', name:'our filter', marker:{color:C.f}, hovertemplate:'%{y:.1f}%'},
         {x:[yrs[0], yrs[yrs.length-1]], y:[luck,luck], type:'scatter', mode:'lines', line:{color:C.ink, width:1.4, dash:'dash'}, name:'luck alone (Poisson)', hoverinfo:'skip'}];
  var L=base(); L.barmode='group'; L.hovermode='x unified';
  L.yaxis.title={text:'typical daily change, %', font:{size:11}}; L.yaxis.rangemode='tozero';
  head(L, 'Typical daily change by year: raw estimate, our filter and luck alone');
  L.annotations=[mark()];
  Plotly.newPlot('ch-years', T, L, cfg('daily-move-by-year'));
})();

/* ---------- the 2021 mining ban ---------- */
(function(){
  var a=D.indexOf('2021-05-10'), z=D.indexOf('2021-07-10');
  var s=function(arr){ return arr.slice(a,z+1); }, x=s(D);
  var T=[{x:x, y:s(H.raw), type:'scatter', mode:'markers', marker:{size:6, color:C.raw}, name:'raw estimate', hovertemplate:'%{y:,.0f} EH/s'},
         {x:x, y:s(H.ma7), type:'scatter', mode:'lines', line:{color:C.ma7, width:2, dash:'dash'}, name:'7-day average', hovertemplate:'%{y:,.0f} EH/s'},
         {x:x, y:s(H.pure), type:'scatter', mode:'lines', line:{color:C.f, width:3}, name:'our filter', hovertemplate:'%{y:,.0f} EH/s'}];
  var L=base(); L.hovermode='x unified'; L.yaxis.title={text:'EH/s', font:{size:11}}; L.yaxis.range=[50,200];
  head(L, 'Hashrate in the 2021 China mining ban: our filter, 7-day average and raw estimate');
  L.annotations=[mark()]; L.shapes=[];
  L.shapes.push({type:'line', xref:'paper', yref:'y', x0:0, x1:1, y0:H.ban.half, y1:H.ban.half, line:{color:C.dim, width:1, dash:'dot'}});
  L.annotations.push({x:1, y:H.ban.half, xref:'paper', yref:'y', xanchor:'right', yanchor:'bottom', showarrow:false, text:'half of the fall, '+Math.round(H.ban.half)+' EH/s', font:{size:11, color:C.dim}});
  var g=H.ban.gapD;
  L.shapes.push({type:'line', xref:'x', yref:'y', x0:g, x1:g, y0:H.ban.gapF, y1:H.ban.gap7, line:{color:C.ink, width:1.5}});
  L.annotations.push({x:g, y:H.ban.gap7, xref:'x', yref:'y', xanchor:'left', yanchor:'bottom', xshift:6, showarrow:false,
     text:(narrow?'':'7-day average ')+Math.round(H.ban.gap7)+' against our filter '+Math.round(H.ban.gapF)+' EH/s', font:{size:12, color:C.ink}});
  var j=D.indexOf('2021-06-09');
  if (j>=0){ L.shapes.push({type:'line', xref:'x', yref:'paper', x0:D[j], x1:D[j], y0:0, y1:1, line:{color:C.dim, width:1, dash:'dot'}});
    L.annotations.push({x:D[j], y:0.04, xref:'x', yref:'paper', xanchor:'right', yanchor:'bottom', xshift:-4, showarrow:false, text:narrow?'9 Jun':'9 Jun: block times point to a real fall', font:{size:11, color:C.dim}}); }
  Plotly.newPlot('ch-ban', T, L, cfg('mining-ban-2021'));
})();
};
if (typeof Plotly !== 'undefined') window.__draw(); else window.addEventListener('load', window.__draw);

document.querySelectorAll('.copy').forEach(function(b){ b.addEventListener('click', function(){
  var el=document.getElementById(b.dataset.copy); navigator.clipboard.writeText(el.innerText).then(function(){ var t0=b.innerText; b.innerText='Copied'; setTimeout(function(){ b.innerText=t0; },1500); }); }); });
</script>
</body>
</html>
""")


def write_site(df, meta, out, *, site_url, repo_root, css_src, tail_shift, fetched_utc, blocks=None):
    out = Path(out)
    st, d = compute_stats(df, meta)
    as_of_iso = str(st["as_of"].date())

    hours = hourly_counts(blocks, st["as_of"])
    if hours is not None and sum(hours) != st["n"]:
        hours = None
    if hours is None:   # a later run of the same day: reuse the published hours
        try:
            with urllib.request.urlopen(site_url + "data/lastday.json", timeout=20) as r:
                prev = json.loads(r.read().decode())
            if prev.get("date") == as_of_iso and sum(prev.get("hours", [])) == st["n"]:
                hours = prev["hours"]
        except Exception:
            hours = None

    def esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    if st["ev"] > 0:
        ev_lede = (f"On {fmt_date(st['ev_d'])} the raw estimate moved {fmt_pct(st['ev_g'])} in a day. That day entered "
                   f"the three-day average at a third of its size, and three days later, when it left the window, "
                   f"the average moved {fmt_pct(st['ev_g3'])}.")
    else:
        ev_lede = ("A lucky day enters a three-day average at a third of its size and leaves it three days later "
                   "in the opposite direction, so the average shows a move that never happened.")
    w_cmp = ("the value it takes on most days" if abs(st["w_last"] - st["w_typ"]) < 5e-3
             else f"against {st['w_typ']:.2f} on most days")

    v = {
        "DESC": "Bitcoin's daily hashrate without the block luck. Our filter uses the Poisson arrival of blocks to remove "
                "most of the luck in the raw daily estimate. We update the series daily, and it is free to download.",
        "SITE_URL": site_url, "PAPER_URL": PAPER_URL, "REPO_URL": REPO_URL, "JAN_URL": JAN_URL, "TEO_URL": TEO_URL,
        "AS_OF": fmt_date(st["as_of"]), "AS_OF_ISO": as_of_iso, "LAG": str(TWO_LAG),
        "PURE": fmt_eh(st["pure"]), "BAND_LO": f"{st['band_lo']:,.0f}", "BAND_HI": f"{st['band_hi']:,.0f}",
        "RAW": fmt_eh(st["raw"]),
        "RAW_GAP": (f"{abs(100 * (st['raw'] / st['pure'] - 1)):.1f}% " + ("below" if st["raw"] < st["pure"] else "above")), "N": str(st["n"]), "N_LO": str(st["n_lo"]), "N_HI": str(st["n_hi"]),
        "LAM": f"{st['lam']:.0f}",
        "LAM_WHY": (f"At the current difficulty our filtered hashrate implies {st['lam']:.0f} blocks a day, about {st['lam'] / 24:.0f} an hour, "
                    + ("because the hashrate is slightly below the level the current difficulty was set for." if st["lam"] < 143.5 else
                       "because the hashrate is slightly above the level the current difficulty was set for." if st["lam"] > 144.5 else
                       "in line with the target.")), "LAM_H": f"{st['lam'] / 24:.1f}", "H_LO": str(st["h_lo"]), "H_HI": str(st["h_hi"]),
        "D1": fmt_pct(st["d1"]), "D7": fmt_pct(st["d7"]), "G_BAND": f"{1.96 * PAPER['growth_unc']:.1f}%",
        "SD_RAW": PAPER["sd_raw"], "SD_FILTER": PAPER["sd_filter"], "NOISE_SHARE": PAPER["noise_share"],
        "LUCK_FULL": PAPER["luck_full"], "LUCK_2016": PAPER["luck_2016"],
        "SD_R": f"{st['sd']['raw']:.1f}%", "SD_3": f"{st['sd']['ma3']:.1f}%", "SD_7": f"{st['sd']['ma7']:.1f}%", "SD_F": f"{st['sd']['pure']:.1f}%", "SD_T": f"{st['sd']['pure_two_sided']:.1f}%",
        "ACF_FROM": "2016", "ACF_RAW": neg(st["acf_raw1"]), "ACF_MA3": neg(st["acf_ma3_3"]), "ACF_MA7": neg(st["acf_ma7_7"]),
        "ACF_PURE": neg(st["acf_pure_max"]), "P_ACF3": PAPER["ma_acf3"],
        "GAIN": f"{100 * st['gain']:.0f}%",
        "LK_N": str(st["n_hi"]), "LK_RAW": fmt_pct(st["lk"]["raw"], sign=False), "LK_M3": fmt_pct(st["lk"]["m3"], sign=False),
        "LK_M7": fmt_pct(st["lk"]["m7"], sign=False), "LK_F0": fmt_pct(st["lk"]["f0"], sign=False), "LK_F1": fmt_pct(st["lk"]["f1"]),
        "EV_LEDE": ev_lede,
        "W_LAST": f"{st['w_last']:.2f}", "W_CMP": w_cmp, "W_N": str(st["w_n"]),
        "HOURS_CLASS": "" if hours is not None else "hidden",
        "BAN_HI": f"{st['ban_hi']:.0f}", "BAN_HI_D": fmt_date(st["ban_hi_d"]),
        "BAN_LO": f"{st['ban_lo']:.0f}", "BAN_LO_D": fmt_date(st["ban_lo_d"]),
        "BAN_LO7_D": fmt_date(st["ban_lo7_d"]), "BAN_LAG7": str(st["ban_lag7"]), "BAN_HALF_F": fmt_date(st["ban_half_f"]), "BAN_HALF_7": fmt_date(st["ban_half_7"]),
        "BAN_HALF_LAG": str(st["ban_half_lag"]), "BAN_GAP_D": fmt_date(st["ban_gap_d"]), "BAN_GAP_7": f"{st['ban_gap_7']:.0f}", "BAN_GAP_F": f"{st['ban_gap_f']:.0f}", "BAN_FLAG_K": f"{100 * st['ban_flag_k']:.0f}%",
        "P_BAN": PAPER["ban"], "P_GAIN": PAPER["gain_day"], "P_WEEK": PAPER["gain_week"],
        "P_COV": PAPER["cov_quiet"], "P_COV1": PAPER["cov_jump1"], "P_COV2": PAPER["cov_jump2"], "P_TWO": PAPER["luck_two"],
        "Q_QUIET": f"{float(meta['q_quiet']):.2e}", "Q_JUMP": f"{Q_JUMP:.2e}",
        "CITE_TEXT": esc(CITE_TEXT), "CITE_BIB": esc(CITE_BIB),
        "UPDATED": fmt_date(pd.Timestamp.utcnow()),
        "PAYLOAD": payload(d, st, hours),
        "CSS": Path(css_src).read_text(),
    }
    html = TEMPLATE.substitute(v)
    assert "—" not in html, "em-dash in page"
    (out / "site").mkdir(parents=True, exist_ok=True)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(html)
    shutil.copy(css_src, out / "site" / "dashboard.css")

    latest = {
        "date": as_of_iso, "pure_eh_s": round(st["pure"], 2),
        "raw_eh_s": round(st["raw"], 2), "n_blocks": st["n"],
        "change_day_pct": round(st["d1"], 2), "change_week_pct": round(st["d7"], 2),
        "source": "Blockchair daily block dumps; filter of Wüstenfeld and Geldner",
        "cite": CITE_TEXT, "paper": PAPER_URL, "code": REPO_URL, "url": site_url,
    }
    (out / "data" / "latest.json").write_text(json.dumps(latest, ensure_ascii=False, indent=1))
    if hours is not None:
        (out / "data" / "lastday.json").write_text(json.dumps({"date": as_of_iso, "hours": hours}))
    (out / "data" / "README.txt").write_text(
        "Bitcoin Hashrate Filter: the daily Bitcoin hashrate without the Poisson luck of block discovery\n"
        f"{site_url}\n\n"
        "daily.csv has one row per UTC day since 2011. Columns:\n"
        "  date                  the UTC day\n"
        "  n_blocks              blocks found that day\n"
        "  difficulty            average difficulty that day\n"
        "  raw                   raw estimate of the hashrate from that day's blocks, in H/s\n"
        "  pure                  our filter, in H/s\n"
        "  pure_two_sided        two-sided version of our filter, in H/s (last 7 days not final)\n"
        "  ma3                   3-day average of the raw estimate, in H/s\n"
        "  weight                how strongly that day's block times point to a real change (0 to 1)\n"
        "  p_filt                variance of our filter's estimate in log terms (a 95% range is pure x exp(+-1.96 x sqrt(p_filt)))\n"
        "  two_sided_days_after  how many of the following 7 days the two-sided value already uses\n\n"
        "Published values never change, except pure_two_sided while two_sided_days_after is below 7.\n"
        "We assign each block to a UTC day by its median time past (the median time stamp of the last 11 blocks), the time the protocol uses for its rules.\n"
        "A 7-day average is raw.rolling(7, min_periods=1).mean().\n\n"
        f"Please cite: {CITE_TEXT}\n{PAPER_URL}\nCode: {REPO_URL}\n"
    )
    share_images(d, st, out, site_url)
