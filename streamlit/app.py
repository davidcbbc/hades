from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import load_posts, sentiment_counts

st.set_page_config(page_title="Hades Post Monitor", page_icon="🛰️", layout="wide")


def get_posts() -> pd.DataFrame:
    return load_posts()


def _format_timestamp(ts: pd.Timestamp | None) -> str:
    if ts is None or pd.isna(ts):
        return "—"
    if ts.tzinfo is None:
        return ts.strftime("%Y-%m-%d %H:%M:%S")
    return ts.tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S UTC")


def main() -> None:
    st.title("🛰️ Hades Dashboard")
    st.caption("Key performance indicators for the latest crawled posts captured by the monitoring agents.")

    posts = get_posts()
    if posts.empty:
        st.info("No crawl data available yet. Run a scan from the Launch Scan page to populate the dashboard.")
        return

    total_posts = len(posts)
    unique_domains = posts["domain"].nunique()
    unique_authors = posts["author"].dropna().nunique()
    last_pull = posts["pulled_at_dt"].max()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Unique Domains", unique_domains)
    col2.metric("Tracked Posts", total_posts)
    col3.metric("Distinct Authors", unique_authors)
    col4.metric("Last Crawl", _format_timestamp(last_pull))

    with st.expander("Sentiment distribution", expanded=True):
        sentiments = sentiment_counts(posts)
        if sentiments.empty:
            st.write("Sentiment data not available.")
        else:
            st.bar_chart(sentiments)

    st.markdown("### Latest Discoveries")
    latest_posts = posts.head(6)[
        [
            "title",
            "domain",
            "author",
            "sentiment",
            "pulled_at_dt",
            "posting_time",
            "url",
        ]
    ].copy()
    latest_posts.rename(
        columns={
            "pulled_at_dt": "pulled_at",
        },
        inplace=True,
    )
    latest_posts["pulled_at"] = latest_posts["pulled_at"].apply(_format_timestamp)
    st.dataframe(
        latest_posts,
        hide_index=True,
        use_container_width=True,
    )

    st.markdown("### Top Domains")
    domain_counts = posts["domain"].value_counts().head(10)
    if domain_counts.empty:
        st.write("No domains recorded yet.")
    else:
        st.dataframe(
            domain_counts.rename("count"),
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
