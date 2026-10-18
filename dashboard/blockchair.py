"""Fetch recent blocks from Blockchair's daily dumps for the dashboard.

Blockchair publishes one gzipped TSV per UTC day at
https://gz.blockchair.com/bitcoin/blocks/ — the same source the paper's
history was built from (see the repository README, "Input data"). A day's
dump carries every block whose header time falls on that day, with the
fields the filter needs: id, time, median_time, difficulty.

One wrinkle: the daily count and difficulty are assigned to days by
MEDIAN time, and a block stamped just after midnight can still carry a
median time on the day before. So closing median-time day D takes dump D
plus the stragglers in the first hours of day D+1 — and day D+1's dump is
published only when day D+1 ends. The caller decides how to source the
stragglers (the next day's dump if it exists, else the live API fallback).
"""

import gzip
import io
import time
import urllib.error
import urllib.request

import pandas as pd

DUMP_URL = "https://gz.blockchair.com/bitcoin/blocks/blockchair_bitcoin_blocks_{day}.tsv.gz"
TIMEOUT = 60
USER_AGENT = "hashrate-filter-dashboard (github.com/janwuestenfeld/hashrate-filter)"

COLUMNS = ["id", "time", "median_time", "difficulty"]


class DumpUnavailable(RuntimeError):
    pass


def _download(url: str, pace: float, tries: int = 4) -> bytes:
    delay = 2.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = r.read()
            time.sleep(pace)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise DumpUnavailable(f"no dump at {url} (404)") from e
            if attempt == tries - 1:
                raise DumpUnavailable(f"GET {url} failed: HTTP {e.code}") from e
        except (urllib.error.URLError, TimeoutError) as e:
            if attempt == tries - 1:
                raise DumpUnavailable(f"GET {url} failed: {e}") from e
        time.sleep(delay)
        delay = min(delay * 2.5, 45.0)
    raise DumpUnavailable(f"GET {url} failed after {tries} tries")


def fetch_dump(day: pd.Timestamp, pace: float = 0.3) -> pd.DataFrame:
    """One day's blocks from its Blockchair dump (header-time day).

    Raises DumpUnavailable if the dump is missing or not a valid gzip
    (a day whose dump has not been published yet).
    """
    day = pd.Timestamp(day)
    url = DUMP_URL.format(day=day.strftime("%Y%m%d"))
    blob = _download(url, pace)
    if len(blob) < 2 or blob[:2] != b"\x1f\x8b":
        raise DumpUnavailable(f"not a gzip file at {url} (dump not published yet?)")
    df = pd.read_csv(
        io.BytesIO(gzip.decompress(blob)),
        sep="\t",
        usecols=COLUMNS,
        parse_dates=["time", "median_time"],
    )
    df = df.rename(columns={"id": "height"})
    return df[["height", "time", "median_time", "difficulty"]]


def fetch_blocks_for_days(
    first_day: pd.Timestamp,
    last_day: pd.Timestamp,
    pace: float = 0.3,
) -> tuple:
    """Blocks for the given days from Blockchair's daily dumps.

    Returns (blocks, available): the block frame in the same schema as the
    mempool fetcher (height, time, median_time, difficulty, sorted by
    height), and a dict mapping each requested day to its dump's
    availability. Days whose dump is unavailable are reported so the
    caller can fall back for them; the caller also sources the
    after-midnight stragglers that close the last median-time day.
    """
    first_day, last_day = pd.Timestamp(first_day), pd.Timestamp(last_day)

    frames = []
    available = {}
    day = first_day
    while day <= last_day:
        try:
            frames.append(fetch_dump(day, pace))
            available[day] = True
        except DumpUnavailable:
            available[day] = False
        day += pd.Timedelta(days=1)

    if not frames:
        blocks = pd.DataFrame(columns=["height", "time", "median_time", "difficulty"])
    else:
        blocks = pd.concat(frames, ignore_index=True)
        blocks = blocks.drop_duplicates(subset="height").sort_values("height").reset_index(drop=True)
    return blocks, available
