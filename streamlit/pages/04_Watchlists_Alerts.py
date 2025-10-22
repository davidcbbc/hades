from __future__ import annotations

import re
from typing import Any

import pandas as pd
import streamlit as st

from utils import (
    find_watchlist_matches,
    load_posts,
    load_watchlists,
    save_watchlists,
    send_watchlist_alert,
    send_test_webhook,
)


def _parse_keywords(raw_input: str) -> list[str]:
    tokens = re.split(r"[\n,;]+", raw_input)
    return [token.strip() for token in tokens if token.strip()]


def _build_matches_table(matches: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for match in matches:
        row = match.get("row") or {}
        hits = match.get("hits") or []
        rows.append(
            {
                "title": row.get("title") or row.get("url"),
                "domain": row.get("domain"),
                "url": row.get("url"),
                "pulled_at": row.get("pulled_at"),
                "posting_time": row.get("posting_time"),
                "matched_keywords": ", ".join(sorted({hit["keyword"] for hit in hits if "keyword" in hit})),
            }
        )
    return pd.DataFrame(rows)


st.title("🛎️ Watchlists & Alerts")
st.caption("Maintain keyword watchlists, inspect matches, and trigger webhook notifications when new chatter appears.")

posts = load_posts(deduplicate=False)
watchlists = load_watchlists()

with st.expander("Create a new watchlist", expanded=not watchlists):
    with st.form("create_watchlist"):
        default_name = f"Watchlist {len(watchlists) + 1}"
        name = st.text_input("Name", value=default_name)
        keywords_input = st.text_area(
            "Keywords",
            placeholder="One keyword per line, or comma-separated.",
            help="Matches are case-insensitive and run across title, summary, sentiment, author, URL, and other fields.",
        )
        webhook_url = st.text_input(
            "Webhook URL",
            placeholder="https://example.com/webhook",
            help="Optional: a POST request with match details is sent here when you trigger alerts.",
        )
        description = st.text_area(
            "Notes",
            placeholder="Add context, e.g., which team owns this watchlist or why the keywords matter.",
        )
        submitted = st.form_submit_button("Add watchlist")

        if submitted:
            keywords = _parse_keywords(keywords_input)
            if not keywords:
                st.error("Provide at least one keyword.")
            else:
                watchlists.append(
                    {
                        "name": name.strip() or default_name,
                        "keywords": keywords,
                        "webhook_url": webhook_url.strip(),
                        "description": description.strip(),
                    }
                )
                save_watchlists(watchlists)
                st.success(f"Watchlist “{name or default_name}” added.")
                st.rerun()

if not watchlists:
    st.info("No watchlists yet. Add one to start tracking high-priority keywords across new crawls.")
else:
    st.subheader("Current watchlists")
    if posts.empty:
        st.warning("No crawl data available yet. Matches will appear once news_data.json is populated.")

    for idx, watchlist in enumerate(watchlists):
        watchlist_name = watchlist.get("name") or f"Watchlist {idx + 1}"
        keywords = watchlist.get("keywords") or []
        matches = find_watchlist_matches(posts, keywords)
        match_count = len(matches)
        match_keywords = sorted({hit["keyword"] for match in matches for hit in match.get("hits", [])})

        header = f"{watchlist_name} • {match_count} match{'es' if match_count != 1 else ''}"
        if match_keywords:
            header += f" • keywords: {', '.join(match_keywords)}"

        with st.expander(header, expanded=match_count > 0):
            if watchlist.get("description"):
                st.caption(watchlist["description"])

            st.markdown(
                f"- **Keywords:** {', '.join(keywords) if keywords else '—'}\n"
                f"- **Webhook:** {watchlist.get('webhook_url') or '—'}"
            )

            status_key = f"watchlist_status_{idx}"

            if match_count == 0:
                st.info("No matches found in the current crawl history.")
            else:
                table = _build_matches_table(matches)
                st.dataframe(table, hide_index=True, use_container_width=True)

            action_cols = st.columns(4)
            with action_cols[0]:
                send_disabled = match_count == 0 or not watchlist.get("webhook_url")
                if st.button("Send webhook alert", key=f"send_{idx}", disabled=send_disabled):
                    success, message = send_watchlist_alert(
                        watchlist_name=watchlist_name,
                        webhook_url=watchlist.get("webhook_url", ""),
                        matches=matches,
                    )
                    st.session_state[status_key] = (success, message)

            with action_cols[1]:
                test_status_key = f"watchlist_test_status_{idx}"
                test_disabled = not watchlist.get("webhook_url")
                if st.button("Test webhook", key=f"test_{idx}", disabled=test_disabled):
                    success, message = send_test_webhook(watchlist.get("webhook_url", ""))
                    st.session_state[test_status_key] = (success, message)

            with action_cols[2]:
                if st.button("Delete watchlist", key=f"delete_{idx}", type="secondary"):
                    del watchlists[idx]
                    save_watchlists(watchlists)
                    st.success(f"Deleted watchlist “{watchlist_name}”.")
                    st.experimental_rerun()

            with action_cols[3]:
                st.write("")  # spacer for layout

            with st.form(f"watchlist_form_{idx}"):
                updated_name = st.text_input("Name", value=watchlist_name)
                updated_keywords = st.text_area(
                    "Keywords",
                    value="\n".join(keywords),
                    help="One keyword per line. Matching is case-insensitive.",
                )
                updated_webhook = st.text_input(
                    "Webhook URL",
                    value=watchlist.get("webhook_url", ""),
                    help="Leave blank to disable webhook alerts for this watchlist.",
                )
                updated_description = st.text_area(
                    "Notes",
                    value=watchlist.get("description", ""),
                    help="Why does this watchlist exist? Who should be notified?",
                )
                save_changes = st.form_submit_button("Save changes")

                if save_changes:
                    parsed_keywords = _parse_keywords(updated_keywords)
                    if not parsed_keywords:
                        st.error("Provide at least one keyword before saving.")
                    else:
                        watchlists[idx] = {
                            "name": updated_name.strip() or watchlist_name,
                            "keywords": parsed_keywords,
                            "webhook_url": updated_webhook.strip(),
                            "description": updated_description.strip(),
                        }
                        save_watchlists(watchlists)
                        st.success(f"Updated watchlist “{updated_name or watchlist_name}”.")
                        st.experimental_rerun()

            for key in (status_key, test_status_key):
                if key in st.session_state:
                    success, message = st.session_state[key]
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
