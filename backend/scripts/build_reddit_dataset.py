"""
Reddit Sunset Dataset Builder
==============================

Fetches top posts from r/sunset (or another subreddit), joins each post's
timestamp with historical weather from Open-Meteo at a given location, and
produces a labeled CSV / Parquet file suitable for ML training.

Usage
-----
    python scripts/build_reddit_dataset.py \\
        --latitude 37.7749 \\
        --longitude -122.4194 \\
        --subreddit sunset \\
        --limit 500 \\
        --time-filter month \\
        --out data/reddit_dataset.csv

BIAS NOTICE
-----------
Reddit upvotes are an imperfect proxy for sunset beauty. They are influenced
by: time of posting, virality, subreddit activity patterns, photo composition,
photographer skill, and post title framing — not just the objective sunset.

Label normalisation (percentile rank within batch) partially mitigates
temporal bias, but the dataset should be treated as "engagement score proxy",
not a ground truth beauty label. Future work: add per-image aesthetic scoring
via a vision model, or per-post manual ratings.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Any, Iterator, Optional

import requests

# ── Bootstrap path so we can import app modules ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REDDIT_BASE = "https://www.reddit.com"
REDDIT_HEADERS = {"User-Agent": os.getenv("REDDIT_USER_AGENT", "sunset-predictor-scraper/1.0")}
OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"

ARCHIVE_HOURLY_VARS = ",".join([
    "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high",
    "visibility", "relative_humidity_2m", "dew_point_2m", "temperature_2m",
    "precipitation", "wind_speed_10m", "surface_pressure",
])

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}

# ---------------------------------------------------------------------------
# Reddit fetching
# ---------------------------------------------------------------------------


def iter_reddit_posts(
    subreddit: str,
    time_filter: str,
    limit: int,
    use_praw: bool = False,
) -> Iterator[dict[str, Any]]:
    """
    Yield post dicts from the subreddit using the public JSON API.

    Falls back to PRAW if credentials are available and use_praw=True.
    """
    if use_praw and _praw_credentials_available():
        yield from _iter_praw(subreddit, time_filter, limit)
    else:
        yield from _iter_public_json(subreddit, time_filter, limit)


def _praw_credentials_available() -> bool:
    return bool(os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET"))


def _iter_praw(subreddit: str, time_filter: str, limit: int) -> Iterator[dict[str, Any]]:
    """Use PRAW (authenticated Reddit API) to fetch posts."""
    import praw

    reddit = praw.Reddit(
        client_id=os.environ["REDDIT_CLIENT_ID"],
        client_secret=os.environ["REDDIT_CLIENT_SECRET"],
        user_agent=os.getenv("REDDIT_USER_AGENT", "sunset-predictor/1.0"),
    )
    sub = reddit.subreddit(subreddit)
    for submission in sub.top(time_filter=time_filter, limit=limit):
        yield {
            "post_id": submission.id,
            "title": submission.title,
            "subreddit": subreddit,
            "created_utc": int(submission.created_utc),
            "score": submission.score,
            "num_comments": submission.num_comments,
            "permalink": f"https://reddit.com{submission.permalink}",
            "url": submission.url,
            "is_image": _is_image_url(submission.url),
        }


def _iter_public_json(subreddit: str, time_filter: str, limit: int) -> Iterator[dict[str, Any]]:
    """
    Paginate the public Reddit JSON API (no auth required).

    Rate limit: ~1 req per 2 seconds to be polite.
    """
    fetched = 0
    after: Optional[str] = None

    while fetched < limit:
        batch = min(100, limit - fetched)
        url = f"{REDDIT_BASE}/r/{subreddit}/top.json"
        params: dict[str, Any] = {"t": time_filter, "limit": batch}
        if after:
            params["after"] = after

        try:
            resp = requests.get(url, params=params, headers=REDDIT_HEADERS, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Reddit API request failed: %s — stopping pagination", exc)
            break

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            post = child.get("data", {})
            yield {
                "post_id": post.get("id", ""),
                "title": post.get("title", ""),
                "subreddit": post.get("subreddit", subreddit),
                "created_utc": int(post.get("created_utc", 0)),
                "score": int(post.get("score", 0)),
                "num_comments": int(post.get("num_comments", 0)),
                "permalink": f"https://reddit.com{post.get('permalink', '')}",
                "url": post.get("url", ""),
                "is_image": _is_image_url(post.get("url", "")),
            }
            fetched += 1

        after = data.get("data", {}).get("after")
        if not after:
            break

        time.sleep(2)  # be polite

    logger.info("Fetched %d posts from r/%s", fetched, subreddit)


def _is_image_url(url: str) -> bool:
    if not url:
        return False
    path = url.split("?")[0].lower()
    return any(path.endswith(ext) for ext in IMAGE_EXTENSIONS) or "imgur.com" in url or "redd.it" in url


# ---------------------------------------------------------------------------
# Weather fetching
# ---------------------------------------------------------------------------


def fetch_historical_weather(
    lat: float, lon: float, target_date: date
) -> Optional[dict[str, float]]:
    """
    Fetch hourly weather from Open-Meteo archive and return the row nearest
    to the 18:00 UTC slot (approximate sunset for mid-latitude locations).

    Returns None if the API call fails.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ARCHIVE_HOURLY_VARS,
        "start_date": str(target_date),
        "end_date": str(target_date),
        "timezone": "UTC",
    }
    try:
        resp = requests.get(OPEN_METEO_ARCHIVE, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.debug("Archive API failed for %s: %s", target_date, exc)
        return None

    hourly = data.get("hourly", {})
    times: list[str] = hourly.get("time", [])
    if not times:
        return None

    # Find index closest to 18:00 UTC as a rough sunset proxy
    target_hour = 18
    target_dt = datetime(target_date.year, target_date.month, target_date.day, target_hour, tzinfo=timezone.utc)
    parsed_times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in times]
    idx = min(range(len(parsed_times)), key=lambda i: abs((parsed_times[i] - target_dt).total_seconds()))

    def get(key: str, default: float = 0.0) -> float:
        vals = hourly.get(key, [])
        if idx < len(vals) and vals[idx] is not None:
            return float(vals[idx])
        return default

    return {
        "cloud_cover": get("cloud_cover"),
        "cloud_cover_low": get("cloud_cover_low"),
        "cloud_cover_mid": get("cloud_cover_mid"),
        "cloud_cover_high": get("cloud_cover_high"),
        "visibility_m": get("visibility", 10000.0),
        "relative_humidity": get("relative_humidity_2m", 50.0),
        "dewpoint_c": get("dew_point_2m"),
        "temperature_c": get("temperature_2m"),
        "precipitation_mm": get("precipitation"),
        "wind_speed_kmh": get("wind_speed_10m"),
        "pressure_hpa": get("surface_pressure", 1013.0),
    }


# ---------------------------------------------------------------------------
# Label construction
# ---------------------------------------------------------------------------


def build_beauty_labels(scores: pd.Series, method: str = "percentile") -> pd.Series:
    """
    Convert raw Reddit upvote scores to a 0–100 beauty label proxy.

    Methods:
    - percentile : rank-based (default; most robust against outliers)
    - log1p      : log(1 + score) normalised to [0, 100]
    - zscore     : z-score clipped to ±3, then scaled to [0, 100]
    """
    if method == "percentile":
        return scores.rank(pct=True) * 100.0

    if method == "log1p":
        logged = np.log1p(scores.clip(lower=0))
        max_val = logged.max()
        if max_val == 0:
            return pd.Series([50.0] * len(scores), index=scores.index)
        return logged / max_val * 100.0

    if method == "zscore":
        mean, std = scores.mean(), scores.std()
        if std == 0:
            return pd.Series([50.0] * len(scores), index=scores.index)
        z = (scores - mean) / std
        z_clipped = z.clip(-3, 3)
        return (z_clipped + 3) / 6 * 100.0

    raise ValueError(f"Unknown label method: {method!r}. Choose: percentile, log1p, zscore")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def build_dataset(
    lat: float,
    lon: float,
    subreddit: str = "sunset",
    limit: int = 500,
    time_filter: str = "month",
    label_method: str = "percentile",
    out_path: str = "data/reddit_dataset.csv",
    fmt: str = "csv",
    min_score: int = 10,
    image_only: bool = False,
) -> None:
    logger.info(
        "Building dataset: r/%s, limit=%d, time_filter=%s, location=(%.4f, %.4f)",
        subreddit, limit, time_filter, lat, lon,
    )

    # ── Step 1: Fetch Reddit posts ────────────────────────────────────────
    posts = list(iter_reddit_posts(subreddit, time_filter, limit, use_praw=True))
    logger.info("Raw posts fetched: %d", len(posts))

    # ── Step 2: Filter ────────────────────────────────────────────────────
    posts = [p for p in posts if p["score"] >= min_score]
    if image_only:
        posts = [p for p in posts if p["is_image"]]
    logger.info("Posts after filtering: %d", len(posts))

    if not posts:
        logger.error("No posts remaining after filtering. Exiting.")
        return

    # ── Step 3: Join with historical weather ──────────────────────────────
    rows = []
    for i, post in enumerate(posts):
        if i % 50 == 0:
            logger.info("  Processing post %d / %d…", i, len(posts))

        post_date = datetime.fromtimestamp(post["created_utc"], tz=timezone.utc).date()
        weather = fetch_historical_weather(lat, lon, post_date)

        if weather is None:
            logger.debug("Skipping post %s — no weather data for %s", post["post_id"], post_date)
            continue

        row = {
            **post,
            "post_date": str(post_date),
            "location_lat": lat,
            "location_lon": lon,
            **weather,
        }
        rows.append(row)
        time.sleep(0.5)  # gentle throttle for archive API

    if not rows:
        logger.error("No rows with weather data. Check lat/lon and date range.")
        return

    df = pd.DataFrame(rows)
    logger.info("Rows with weather: %d", len(df))

    # ── Step 4: Build beauty label ────────────────────────────────────────
    df["beauty_label"] = build_beauty_labels(df["score"], method=label_method)
    df["label_method"] = label_method

    # ── Step 5: Save ──────────────────────────────────────────────────────
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "parquet":
        out = out.with_suffix(".parquet")
        df.to_parquet(out, index=False)
    else:
        df.to_csv(out, index=False)

    logger.info("Dataset saved to %s (%d rows, %d columns)", out, len(df), len(df.columns))
    logger.info("Label stats: min=%.1f mean=%.1f max=%.1f", df["beauty_label"].min(), df["beauty_label"].mean(), df["beauty_label"].max())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a labeled sunset dataset from Reddit + Open-Meteo weather."
    )
    parser.add_argument("--latitude", type=float, required=True)
    parser.add_argument("--longitude", type=float, required=True)
    parser.add_argument("--subreddit", default="sunset")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--time-filter", default="month",
                        choices=["hour", "day", "week", "month", "year", "all"])
    parser.add_argument("--label-method", default="percentile",
                        choices=["percentile", "log1p", "zscore"])
    parser.add_argument("--out", default="data/reddit_dataset.csv")
    parser.add_argument("--format", dest="fmt", default="csv", choices=["csv", "parquet"])
    parser.add_argument("--min-score", type=int, default=10,
                        help="Minimum Reddit score to include a post")
    parser.add_argument("--image-only", action="store_true",
                        help="Only include posts that link to images")

    args = parser.parse_args()

    build_dataset(
        lat=args.latitude,
        lon=args.longitude,
        subreddit=args.subreddit,
        limit=args.limit,
        time_filter=args.time_filter,
        label_method=args.label_method,
        out_path=args.out,
        fmt=args.fmt,
        min_score=args.min_score,
        image_only=args.image_only,
    )


if __name__ == "__main__":
    main()
