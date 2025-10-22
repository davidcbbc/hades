from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from utils import (
    export_post_to_misp,
    filter_posts_by_keywords,
    load_posts,
    resolve_screenshot_path,
)

BREACH_KEYWORDS = [
    "breach",
    "breached",
    "data leak",
    "leak",
    "database",
    "dump",
    "credentials",
    "password",
    "exposed",
    "compromised",
]

BREACH_DOMAIN_HINTS = [
    "breach",
    "leak",
    "breached",
    "breachsta",
    "raidforum",
    "breachedforums",
]


def get_breach_posts() -> pd.DataFrame:
    posts = load_posts(deduplicate=False)
    if posts.empty:
        return posts

    filtered = filter_posts_by_keywords(posts, BREACH_KEYWORDS, BREACH_DOMAIN_HINTS)
    if filtered.empty:
        return filtered

    if "hash" in filtered.columns:
        filtered = filtered.drop_duplicates(subset="hash", keep="first")
    return filtered.reset_index(drop=True)


def main() -> None:
    st.title("🧾 Breached Data")
    st.caption("Monitor leaked datasets and breach disclosures captured by the crawlers.")

    posts = get_breach_posts()
    if posts.empty:
        st.info("No breach-related posts available yet.")
        return

    page_size = st.selectbox("Items per page", options=[5, 10, 20], index=1)
    total_pages = max(1, math.ceil(len(posts) / page_size))
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)

    start = (page - 1) * page_size
    end = start + page_size

    for idx, (_, row) in enumerate(posts.iloc[start:end].iterrows()):
        row_key = f"{row.get('hash') or row.get('url') or idx}-{start + idx}"
        st.markdown("---")
        title = row["title"] or row["url"] or "Untitled post"
        st.subheader(title)
        st.caption(f"Domain: {row['domain'] or 'unknown'} • Pulled at: {row['pulled_at'] or '—'}")

        cols = st.columns([3, 2])
        with cols[0]:
            st.markdown(
                f"""
**Author:** {row.get("author") or "—"}  
**Sentiment:** {row.get("sentiment") or "—"}  
**Posting time:** {row.get("posting_time") or "—"}  
**URL:** {row.get("url") or "—"}  
**Hash:** {row.get("hash") or "—"}
"""
            )
            st.markdown(f"**Short summary:** {row.get('short_summary') or '—'}")
            st.markdown("**Long summary:**")
            st.write(row.get("long_summary") or "—")

        with cols[1]:
            screenshot_path = resolve_screenshot_path(row.get("screenshot_path"))
            if screenshot_path:
                st.image(str(screenshot_path), caption=screenshot_path.name, use_container_width=True)
            else:
                st.info("No screenshot available for this entry.")

        button_key = f"export_misp_{row_key}"
        status_key = f"misp_status_{row_key}"
        if st.button("Export to MISP", key=button_key):
            success, message = export_post_to_misp(row)
            st.session_state[status_key] = (success, message)

        if status_key in st.session_state:
            success, message = st.session_state[status_key]
            if success:
                st.success(message)
            else:
                st.error(message)


if __name__ == "__main__":
    main()
