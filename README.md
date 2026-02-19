# Hades

AI-assisted post monitoring toolkit with:
- CLI crawlers for breach/vulnerability discussions
- a Streamlit dashboard for triage and exports
- optional watchlist webhook alerts and MISP export

## What Is In This Repo

- `posts_monitor_yolo.py`: breach-focused crawler (top 3 recent posts)
- `posts_monitor_instructed.py`: breach-focused crawler with optional `--nav_prompt` guidance
- `posts_monitor_yolo_vulns.py`: vulnerability/exploit-focused crawler
- `posts_monitor_yolo_tor.py`: breach crawler routed through Tor SOCKS proxy
- `posts_monitor_instructed_tor.py`: instructed crawler routed through Tor SOCKS proxy
- `streamlit/app.py`: main dashboard (KPIs + sentiment + latest discoveries)
- `streamlit/pages/02_Launch_Scan.py`: run supported crawler scripts from UI
- `streamlit/pages/03_Crawled_Websites.py`: breach-related filtered posts
- `streamlit/pages/04_Vulnerabilities.py`: vulnerability-related filtered posts
- `streamlit/pages/04_Watchlists_Alerts.py`: keyword watchlists + webhook alerts
- `streamlit/utils.py`: data loading/filtering + MISP/webhook helpers

## Requirements

- Python `>=3.12` (from `pyproject.toml`)
- dependencies from `uv.lock` / `pyproject.toml`
- Browser Use compatible browser runtime
- For Tor scripts: local SOCKS proxy on `localhost:9050`

## Setup

### 1) Install dependencies

Using `uv`:

```bash
uv sync
```

Using `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install browser-use streamlit pandas python-dateutil python-dotenv requests pydantic
```

### 2) Configure environment

Create `.env` in project root. Common keys used by this project:

```bash
# LLM/provider keys (as needed by your selected model/provider)
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_ENDPOINT=
AZURE_ENDPOINT=
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GOOGLE_API_KEY=
GROQ_API_KEY=

# Optional logging controls
BROWSER_USE_LOGGING_LEVEL=info
ANONYMIZED_TELEMETRY=false

# Optional MISP integration (Streamlit export button)
MISP_URL=
MISP_API_KEY=
MISP_DISTRIBUTION=0
MISP_ANALYSIS=2
MISP_THREAT_LEVEL=4
MISP_TIMEOUT=10
MISP_VERIFY_SSL=true
MISP_TAGS=

# Optional watchlist webhook settings
WATCHLIST_WEBHOOK_TIMEOUT=10
WATCHLIST_WEBHOOK_VERIFY_SSL=true
```

## CLI Usage

All monitor scripts support:

```bash
python <script>.py --url <target> [--once] [--interval 300] [--output news_data.json] [--debug]
```

Extra option for instructed variants:

```bash
--nav_prompt <path_to_text_file>
```

### Examples

Single run:

```bash
python posts_monitor_yolo.py --url https://example.com/forum --once --debug
```

Continuous mode:

```bash
python posts_monitor_instructed.py --url https://example.com/forum --interval 600
```

Tor-routed run:

```bash
python posts_monitor_yolo_tor.py --url https://example.com/forum --once
```

Vulnerability-focused run:

```bash
python posts_monitor_yolo_vulns.py --url https://example.com/forum --once
```

## Streamlit App

Run from repo root:

```bash
streamlit run streamlit/app.py
```

### Pages

- Dashboard: high-level KPIs and recent posts
- Launch Scan: UI wrapper for `posts_monitor_yolo.py` and `posts_monitor_instructed.py`
- Breached Data: breach/leak keyword filtering + screenshot preview + MISP export
- Vulnerabilities: vuln/exploit keyword filtering + screenshot preview + MISP export
- Watchlists & Alerts: create keyword lists, view matches, send webhook alerts

## Data Files

- `news_data.json`: rolling store (last 100 saved records)
- `shots/`: screenshots captured by crawler tools
- `watchlists.json`: persisted watchlist config

Expected `news_data.json` item shape:

```json
{
  "hash": "<md5(url)>",
  "pulled_at": "2026-02-19T00:00:00Z",
  "data": {
    "title": "...",
    "author": "...",
    "url": "...",
    "posting_time": "...",
    "short_summary": "...",
    "long_summary": "...",
    "sentiment": "positive|neutral|negative",
    "screenshot_path": "..."
  }
}
```

## Security Notes


- Tor scripts assume `localhost:9050`; change script args if your Tor endpoint differs.

## Troubleshooting

- If parsing fails, run with `--debug` to inspect raw model output.
- If Streamlit shows no data, verify `news_data.json` exists and is valid JSON.
- If MISP export fails, verify `MISP_URL` and `MISP_API_KEY`.
- If webhook alerts fail, validate webhook URL and SSL settings.
