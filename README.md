# Hades TOR Post Monitor

## Overview
`posts_monitor_instructed_tor2.py` drives a Browser Use agent (Gemini Flash via Azure OpenAI) through a TOR-enabled Chromium session to find the newest breach-related posts on a target site. The agent extracts structured metadata, captures full-page evidence, and stores results locally for both one-off runs and continuous monitoring.

## Key Features
- Navigates a provided domain (including subdomains) and asks the LLM agent to surface the three newest breach discussions.
- Captures evidence screenshots with a custom `maybe_capture_screenshot` tool that saves base64 payloads to `./shots/`.
- Persists each post to a rolling JSON log (`news_data.json`) with URL-based deduplication.
- Supports deterministic navigation hints via an optional instruction file.
- Offers verbose/headful mode for debugging alongside quiet headless operation for scheduled monitoring.

## Requirements
- Python 3.10+ with access to the project's dependencies (`browser-use`, `python-dotenv`, `python-dateutil`, `pydantic`, etc.).
- A running TOR SOCKS proxy on `localhost:9050` (adjust the code if your proxy uses another port).
- Azure OpenAI configuration for the `browser_use.llm.ChatAzureOpenAI` client (API key, endpoint, deployment name); define these in a `.env` file loaded at startup.

## Configuration
1. Create a `.env` file alongside the script and populate the Azure OpenAI variables expected by `ChatAzureOpenAI` (for example `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_DEPLOYMENT`).
2. Ensure the TOR proxy is reachable at `localhost:9050`. The browser session also disables DNS leaks and bypasses only loopback traffic.
3. (Optional) Prepare a navigation prompt text file that lists deterministic steps the agent should follow before free-form analysis.

## Usage
Run the script directly from the repo root:

```bash
python posts_monitor_instructed_tor2.py [--url TARGET_URL] [--once] [--interval SECONDS] \
  [--output PATH] [--debug] [--nav_prompt FILE]
```

- `--url`: Site root to investigate (default `https://www.techcrunch.com`).
- `--once`: Perform a single sweep and exit; omit for continuous monitoring.
- `--interval`: Seconds between scans when monitoring (default `300`).
- `--output`: JSON file that stores extracted posts (default `news_data.json`).
- `--debug`: Enables Browser Use logging, headful Chromium, and verbose console output.
- `--nav_prompt`: Injects navigation instructions from a text file before the agent begins free-form exploration.

### Single Extraction
```bash
python posts_monitor_instructed_tor2.py --once --url https://example.com/forum --debug
```
Prints a status line for the newest post, writes a screenshot to `./shots/`, and saves the record in `news_data.json`.

### Continuous Monitoring
```bash
python posts_monitor_instructed_tor2.py --url https://example.com/forum --interval 600
```
Loops forever, only displaying and persisting posts whose URLs were not seen previously.

## Output Layout
- `news_data.json`: Rolling log of the most recent 100 posts; each entry includes a URL hash, timestamp, and the agent's structured payload.
- `./shots/`: Evidence screenshots named after the post URL (or a provided filename).

## Troubleshooting
- **Empty or malformed JSON**: Enable `--debug` to inspect the agent transcript. The script already strips backticks and `<json>` tags before parsing, but unforeseen formats may still need manual cleanup.
- **Connection errors**: Confirm your TOR proxy port and that outbound access through it is permitted.
- **Agent navigation loops**: Provide a concise nav prompt file to anchor the agent before it explores freely.

## Extending
If you need to change the extraction logic, focus on `extract_latest_article()`. It assembles the agent prompt, runs the Browser Use session, and normalizes the LLM output before handing results to the CLI helpers.
