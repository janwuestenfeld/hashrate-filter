"""Fetch recent blocks from the mempool.space API for the dashboard.

One row per block with the three fields the filter needs (header time,
median time, difficulty) plus the height. The API returns blocks in pages
of ten, newest first; fetching one day's ~150 blocks takes about fifteen
calls. Calls are paced and retried with backoff.

Only complete days are used: a calendar day is complete once the chain
tip's median time has passed into the following day, because median time
never decreases along the chain, so no later block can still land on that
day. The caller decides which days to build; this module just walks back
far enough to cover them, with one day of slack on each side (a block's
median time can differ from its header time by a couple of hours at most).
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

import pandas as pd

API = "https://mempool.space/api"
PAGE = 10  # blocks per /api/blocks/{height} call
TIMEOUT = 30
USER_AGENT = "hashrate-filter-dashboard (github.com/janwuestenfeld/hashrate-filter)"


class FetchError(RuntimeError):
    pass


def _get(path: str, pace: float, tries: int = 6) -> object:
    """GET JSON with pacing and exponential backoff."""
    url = f"{API}{path}"
    delay = 1.0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = json.loads(r.read().decode())
            time.sleep(pace)
            return data
        except urllib.error.HTTPError as e:
            if e.code == 429 or 500 <= e.code < 600:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                wait = float(retry_after) if retry_after else delay
            elif attempt == tries - 1:
                raise FetchError(f"GET {url} failed: HTTP {e.code}") from e
            else:
                wait = delay
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == tries - 1:
                raise FetchError(f"GET {url} failed: {e}") from e
            wait = delay
        time.sleep(wait)
        delay = min(delay * 2.5, 60.0)
    raise FetchError(f"GET {url} failed after {tries} tries")


def tip_height(pace: float = 0.3) -> int:
    return int(_get("/blocks/tip/height", pace))


def fetch_blocks_back_to(
    first_day: pd.Timestamp,
    pace: float = 0.3,
    cache_path: Optional[Path] = None,
    from_cache: bool = False,
) -> pd.DataFrame:
    """All blocks from `first_day` (UTC, naive) to the tip, with one day of
    slack on the early side, plus the tip state.

    Returns a frame with columns height, time, median_time (UTC datetimes)
    and difficulty, sorted by height, and the frame's attrs carry
    tip_height and tip_median_day (the chain tip's median-time day, the
    completeness watermark: days before it are complete).
    """
    first_day = pd.Timestamp(first_day)
    horizon = first_day - pd.Timedelta(days=1)

    if from_cache and cache_path and Path(cache_path).exists():
        raw = json.loads(Path(cache_path).read_text())
    else:
        tip = tip_height(pace)
        raw = []
        h = tip
        while True:
            page = _get(f"/blocks/{h}", pace)
            if not page:
                raise FetchError(f"empty page at height {h}")
            raw.extend(page)
            oldest = page[-1]
            if (
                pd.Timestamp(oldest["timestamp"], unit="s", tz="UTC") < horizon.tz_localize("UTC")
                and pd.Timestamp(oldest["mediantime"], unit="s", tz="UTC") < horizon.tz_localize("UTC")
            ):
                break
            h = oldest["height"] - 1
        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            Path(cache_path).write_text(json.dumps(raw))

    df = pd.DataFrame(
        {
            "height": [int(b["height"]) for b in raw],
            "time": pd.to_datetime([int(b["timestamp"]) for b in raw], unit="s", utc=True).tz_convert(None),
            "median_time": pd.to_datetime([int(b["mediantime"]) for b in raw], unit="s", utc=True).tz_convert(None),
            "difficulty": [float(b["difficulty"]) for b in raw],
        }
    )
    df = df.drop_duplicates(subset="height").sort_values("height").reset_index(drop=True)
    if df[["time", "median_time", "difficulty"]].isna().any().any():
        raise FetchError("missing values in fetched blocks")

    tip_med = df["median_time"].iloc[-1]
    df.attrs["tip_height"] = int(df["height"].iloc[-1])
    df.attrs["tip_median_day"] = tip_med.floor("D")
    df.attrs["fetched_utc"] = pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    return df


def complete_days(blocks: pd.DataFrame) -> List[pd.Timestamp]:
    """Calendar days (UTC) that are complete given the tip's median time.

    A day d is complete once the tip's median time is on d+1 or later:
    median time never decreases along the chain, so no future block can
    still carry a median time on d.
    """
    watermark = blocks.attrs["tip_median_day"]
    days = blocks["median_time"].dt.floor("D").unique()
    return sorted(day for day in days if day < watermark)


def fetch_blocks_since(cutoff: pd.Timestamp, pace: float = 0.3) -> pd.DataFrame:
    """Blocks with header time at or after `cutoff` (UTC, naive), walked
    back from the tip. Used for the after-midnight stragglers whose median
    time still belongs to the day before (they close the latest day)."""
    cutoff = pd.Timestamp(cutoff)
    h = tip_height(pace)
    raw = []
    while True:
        page = _get(f"/blocks/{h}", pace)
        if not page:
            raise FetchError(f"empty page at height {h}")
        for b in page:
            if pd.Timestamp(b["timestamp"], unit="s") >= cutoff:
                raw.append(b)
        if pd.Timestamp(page[0]["timestamp"], unit="s") < cutoff:
            break
        h = page[-1]["height"] - 1
    df = pd.DataFrame(
        {
            "height": [int(b["height"]) for b in raw],
            "time": pd.to_datetime([int(b["timestamp"]) for b in raw], unit="s"),
            "median_time": pd.to_datetime([int(b["mediantime"]) for b in raw], unit="s"),
            "difficulty": [float(b["difficulty"]) for b in raw],
        }
    )
    return df.drop_duplicates(subset="height").sort_values("height").reset_index(drop=True)
