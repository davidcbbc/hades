from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, List, Mapping, Sequence, Tuple
from urllib.parse import urlparse

import pandas as pd
from dateutil import parser as dtparser

try:
    import requests
except ImportError:  # pragma: no cover - surface to UI instead
    requests = None

NEWS_DATA_PATH = Path("../news_data.json")
WATCHLISTS_PATH = Path("../watchlists.json")
SCREENSHOT_DIRS: List[Path] = [
    Path("shots"),
    Path("post_monitor_yolo_resources") / "screenshots",
]


def _parse_datetime(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    try:
        return pd.to_datetime(value, utc=True)
    except Exception:
        try:
            return pd.to_datetime(dtparser.parse(value))
        except Exception:
            return None


def load_posts(deduplicate: bool = True) -> pd.DataFrame:
    """Load news_data.json into a flattened DataFrame."""
    if not NEWS_DATA_PATH.exists():
        return pd.DataFrame()

    with NEWS_DATA_PATH.open("r", encoding="utf-8") as f:
        try:
            raw = json.load(f)
        except json.JSONDecodeError:
            return pd.DataFrame()

    records = []
    for entry in raw:
        payload = entry.get("data") or {}
        record = {
            "hash": entry.get("hash"),
            "pulled_at": entry.get("pulled_at"),
            "pulled_at_dt": _parse_datetime(entry.get("pulled_at")),
            "title": payload.get("title"),
            "author": payload.get("author"),
            "url": payload.get("url"),
            "domain": urlparse(payload.get("url", "")).netloc,
            "posting_time": payload.get("posting_time"),
            "posting_time_dt": _parse_datetime(payload.get("posting_time")),
            "short_summary": payload.get("short_summary"),
            "long_summary": payload.get("long_summary"),
            "sentiment": payload.get("sentiment"),
            "screenshot_path": payload.get("screenshot_path"),
        }
        records.append(record)

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df = df.sort_values("pulled_at_dt", ascending=False)
    if deduplicate and "hash" in df.columns:
        df = df.drop_duplicates(subset="hash", keep="first")
    return df.reset_index(drop=True)


def filter_posts_by_keywords(
    posts: pd.DataFrame,
    keywords: Sequence[str],
    domain_hints: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return rows that contain any keyword within text fields or domain hints."""
    if posts.empty:
        return posts

    text_columns = [col for col in ["title", "short_summary", "long_summary", "url"] if col in posts.columns]
    if text_columns:
        aggregated = posts[text_columns].fillna("").agg(" ".join, axis=1).str.lower()
    else:
        aggregated = pd.Series("", index=posts.index)

    mask = pd.Series(False, index=posts.index)
    for keyword in keywords:
        kw = keyword.strip().lower()
        if not kw:
            continue
        mask |= aggregated.str.contains(kw, regex=False)

    if domain_hints and "domain" in posts.columns:
        domains = posts["domain"].fillna("").str.lower()
        for hint in domain_hints:
            domain_kw = hint.strip().lower()
            if not domain_kw:
                continue
            mask |= domains.str.contains(domain_kw, regex=False)

    return posts[mask].reset_index(drop=True)


def resolve_screenshot_path(path: str | None) -> Path | None:
    """Return the first matching screenshot path if it exists."""
    if not path:
        return None

    candidate = Path(f"/home/capella/Desktop/Hades/{path}")
    if candidate.exists():
        return candidate

    for base in SCREENSHOT_DIRS:
        alt = base / candidate.name
        if alt.exists():
            return alt

    return None


def sentiment_counts(df: pd.DataFrame) -> pd.Series:
    if df.empty or "sentiment" not in df.columns:
        return pd.Series(dtype=int)
    return df["sentiment"].fillna("unknown").value_counts()


def _env_bool(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no"}


def export_post_to_misp(row: Mapping[str, Any] | pd.Series) -> Tuple[bool, str]:
    """Push a single post as a MISP event using REST API."""
    if requests is None:
        return False, "Missing dependency: install requests to enable MISP export."

    misp_url = (os.getenv("MISP_URL") or "").strip().rstrip("/")
    api_key = (os.getenv("MISP_API_KEY") or "").strip()
    if not misp_url or not api_key:
        return False, "MISP_URL and MISP_API_KEY must be set in the environment."

    distribution = os.getenv("MISP_DISTRIBUTION", "0")
    analysis = os.getenv("MISP_ANALYSIS", "2")
    threat_level = os.getenv("MISP_THREAT_LEVEL", "4")
    timeout = float(os.getenv("MISP_TIMEOUT", "10"))
    verify_ssl = _env_bool("MISP_VERIFY_SSL", True)

    tags_env = os.getenv("MISP_TAGS")
    tags: Sequence[str] = []
    if tags_env:
        tags = [tag.strip() for tag in tags_env.split(",") if tag.strip()]

    if isinstance(row, pd.Series):
        row = row.to_dict()

    title = (row.get("title") or row.get("url") or "Unknown post").strip()
    domain = (row.get("domain") or "unknown").strip()
    info = f"Hades Crawl • {domain} • {title}"

    attributes = []
    attribute_fields = [
        ("url", "link"),
        ("domain", "domain"),
        ("author", "text"),
        ("sentiment", "text"),
        ("posting_time", "datetime"),
        ("pulled_at", "datetime"),
        ("hash", "other"),
        ("short_summary", "text"),
        ("long_summary", "text"),
        ("screenshot_path", "text"),
    ]

    for field, attribute_type in attribute_fields:
        value = row.get(field)
        if not value:
            continue
        attributes.append(
            {
                "category": "External analysis",
                "type": attribute_type,
                "value": str(value),
                "comment": f"Extracted {field.replace('_', ' ')} from Hades crawler.",
            }
        )

    event_body: dict[str, Any] = {
        "info": info,
        "distribution": distribution,
        "analysis": analysis,
        "threat_level_id": threat_level,
        "Attribute": attributes,
    }

    if tags:
        event_body["Tag"] = [{"name": tag} for tag in tags]

    if row.get("pulled_at"):
        event_body.setdefault("published", False)

    payload = {"Event": event_body}
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": api_key,
    }

    endpoint = f"{misp_url}/events"

    try:
        response = requests.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=timeout,
            verify=verify_ssl,
        )
    except Exception as exc:  # pragma: no cover - external service
        return False, f"MISP export failed: {exc}"

    if response.status_code >= 300:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        return (
            False,
            f"MISP export returned {response.status_code}: {detail}",
        )

    return True, "Event exported to MISP."


def load_watchlists() -> list[dict[str, Any]]:
    """Return persisted watchlists from disk."""
    if not WATCHLISTS_PATH.exists():
        return []
    try:
        with WATCHLISTS_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(data, list):
        return [entry for entry in data if isinstance(entry, dict)]
    return []


def save_watchlists(watchlists: Sequence[Mapping[str, Any]]) -> None:
    """Persist watchlists to disk."""
    serializable: list[dict[str, Any]] = []
    for entry in watchlists:
        if not isinstance(entry, Mapping):
            continue
        serializable.append(
            {
                "name": str(entry.get("name") or "Untitled watchlist"),
                "keywords": [str(k) for k in entry.get("keywords", []) if str(k).strip()],
                "webhook_url": str(entry.get("webhook_url") or ""),
                "description": str(entry.get("description") or ""),
            }
        )

    WATCHLISTS_PATH.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _stringify_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return str(value)


def _collect_keyword_hits(row: Mapping[str, Any], keywords: Sequence[str]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    normalized = [
        (keyword, keyword.lower())
        for keyword in keywords
        if keyword and keyword.lower().strip()
    ]
    if not normalized:
        return hits

    for field, value in row.items():
        text = _stringify_value(value)
        if not text:
            continue
        lower_text = text.lower()
        for original, lowered in normalized:
            if lowered in lower_text:
                match_index = lower_text.find(lowered)
                snippet_start = max(match_index - 40, 0)
                snippet_end = min(match_index + len(lowered) + 40, len(text))
                snippet = text[snippet_start:snippet_end]
                if snippet_start > 0:
                    snippet = "…" + snippet
                if snippet_end < len(text):
                    snippet = snippet + "…"
                hits.append(
                    {
                        "keyword": original,
                        "field": field,
                        "snippet": snippet,
                    }
                )
    return hits


def find_watchlist_matches(posts: pd.DataFrame, keywords: Sequence[str]) -> list[dict[str, Any]]:
    """Return posts that match the provided keywords."""
    if posts.empty or not keywords:
        return []

    matches: list[dict[str, Any]] = []
    for _, series in posts.iterrows():
        row = series.to_dict()
        hits = _collect_keyword_hits(row, keywords)
        if hits:
            matches.append({"row": row, "hits": hits})
    return matches


def send_watchlist_alert(
    watchlist_name: str,
    webhook_url: str,
    matches: Sequence[Mapping[str, Any]],
) -> tuple[bool, str]:
    """Send a webhook alert containing keyword matches."""
    if not webhook_url:
        return False, "Webhook URL is not configured for this watchlist."
    if requests is None:
        return False, "Missing dependency: install requests to enable webhook alerts."

    timeout = float(os.getenv("WATCHLIST_WEBHOOK_TIMEOUT", "10"))
    verify_ssl = _env_bool("WATCHLIST_WEBHOOK_VERIFY_SSL", True)

    payload_matches = []
    for match in matches:
        row = match.get("row") or {}
        hits = match.get("hits") or []
        payload_matches.append(
            {
                "hash": row.get("hash"),
                "title": row.get("title"),
                "url": row.get("url"),
                "domain": row.get("domain"),
                "pulled_at": row.get("pulled_at"),
                "posting_time": row.get("posting_time"),
                "matched_keywords": sorted({hit["keyword"] for hit in hits if "keyword" in hit}),
                "evidence": hits,
            }
        )

    payload = {
        "watchlist": watchlist_name,
        "match_count": len(payload_matches),
        "matches": payload_matches,
    }

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=timeout,
            verify=verify_ssl,
        )
    except Exception as exc:  # pragma: no cover
        return False, f"Webhook delivery failed: {exc}"

    if response.status_code >= 300:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        return False, f"Webhook responded with {response.status_code}: {detail}"

    return True, "Alert sent to webhook."


def send_test_webhook(webhook_url: str) -> tuple[bool, str]:
    """Send a minimal payload to verify webhook connectivity."""
    if not webhook_url:
        return False, "Webhook URL is required."
    if requests is None:
        return False, "Missing dependency: install requests to enable webhook alerts."

    timeout = float(os.getenv("WATCHLIST_WEBHOOK_TIMEOUT", "10"))
    verify_ssl = _env_bool("WATCHLIST_WEBHOOK_VERIFY_SSL", True)

    payload = {
        "watchlist": "Test Watchlist",
        "match_count": 0,
        "message": "Webhook connectivity test from Hades Watchlists page.",
    }

    try:
        response = requests.post(
            webhook_url,
            json=payload,
            timeout=timeout,
            verify=verify_ssl,
        )
    except Exception as exc:  # pragma: no cover
        return False, f"Webhook delivery failed: {exc}"

    if response.status_code >= 300:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        return False, f"Webhook responded with {response.status_code}: {detail}"

    return True, "Test payload sent to webhook."
