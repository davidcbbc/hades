#!/usr/bin/env python3
"""
News monitoring agent with browser-use + Gemini Flash.
Automatically extracts and analyzes the latest articles from any news website.
"""

import argparse
import sys
import asyncio
import hashlib
import json
import logging
import os
import time
from datetime import datetime
from typing import Literal
from dotenv import load_dotenv
import re
from browser_use.browser.events import ScreenshotEvent

from dateutil import parser as dtparser
from pydantic import BaseModel
from browser_use.llm import ChatAzureOpenAI
from browser_use import Tools, Agent, BrowserSession
from pydantic import BaseModel, Field
from browser_use.agent.views import ActionResult
from pathlib import Path
import base64

tools = Tools(exclude_actions=['evaluate'])

load_dotenv()

def setup_environment(debug: bool):
	if not debug:
		os.environ['BROWSER_USE_SETUP_LOGGING'] = 'false'
		os.environ['BROWSER_USE_LOGGING_LEVEL'] = 'critical'
		logging.getLogger().setLevel(logging.CRITICAL)
	else:
		os.environ['BROWSER_USE_SETUP_LOGGING'] = 'true'
		os.environ['BROWSER_USE_LOGGING_LEVEL'] = 'info'


parser = argparse.ArgumentParser(description='News extractor using Browser-Use + Gemini')
parser.add_argument('--url', default='https://www.techcrunch.com', help='News site root URL')
parser.add_argument('--interval', type=int, default=300, help='Seconds between checks in monitor mode')
parser.add_argument('--once', action='store_true', help='Run a single extraction and exit')
parser.add_argument('--output', default='news_data.json', help='Path to JSON file where articles are stored')
parser.add_argument('--debug', action='store_true', help='Verbose console output and non-headless browser')
parser.add_argument('--nav_prompt', help='Path to a text file with navigation steps to prepend to the Agent task')
args = parser.parse_args()

setup_environment(args.debug)


# ---------------------------------------------------------
# Tools
# ---------------------------------------------------------

def _safe_name(url: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", url)[:80]
    return f"{int(time.time())}_{slug or 'page'}.jpg"

@tools.action(
    description=(
        "Capture a screenshot **only** if the current page is a post"
        "post-detail (or otherwise high-value). Return the saved path."
    )
)
async def maybe_capture_screenshot(filename: str | None = None, browser_session=None) -> ActionResult:
    # Get URL for naming
    print("Tool CAPTURE_SCREENSHOT CALLED !!!")
    state = await browser_session.get_browser_state_summary()
    url = state.url or "about:blank"
    out_dir = Path("./shots")
    out_dir.mkdir(parents=True, exist_ok=True)
    outfile = out_dir / (filename or _safe_name(url))

    # Take the screenshot via the event system (sync result)
    evt = browser_session.event_bus.dispatch(ScreenshotEvent(full_page=True))
    await evt
    result = await evt.event_result(raise_if_any=True, raise_if_none=True)

    # result often contains base64 payload; support both dict or raw base64
    b64 = result.get("screenshot_base64") if isinstance(result, dict) else result
    outfile.write_bytes(base64.b64decode(b64))

    return ActionResult(
        extracted_content=f"📸 Saved screenshot for {url} → {outfile}",
        attachments=[str(outfile)]
    )


@tools.action(description="Capture a full-page screenshot and return the saved file path")
async def capture_sreenshot(path: str, browser_session: BrowserSession) -> ActionResult:
    """
    LLM calls this to capture evidence. The returned path is meant to be embedded in Markdown.
    """

	
	
    print("called capture_screenshot")

    # Access injected objects per docs (browser_session, cdp_client, etc.)
    #browser_session = locals().get("browser_session")
    if browser_session is None:
        return ActionResult(error="browser_session not available in tool context")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    page = await browser_session.get_current_page()
    shot = await page.screenshot()  # may be bytes OR base64


    if isinstance(shot, (bytes, bytearray)):
        data = bytes(shot)
    else:
        # assume base64-encoded string
        data = base64.b64decode(shot)

    with open(path, "wb") as f:
        f.write(data)

    return ActionResult(
        extracted_content=path,
        attachments=[path]   # shows in history; LLM can reference path in the report
    )



class Post(BaseModel):
	title: str
	author: str
	url: str
	posting_time: str
	short_summary: str
	long_summary: str
	sentiment: Literal['positive', 'neutral', 'negative']
	screenshot_path: str


# ---------------------------------------------------------
# Core extractor
# ---------------------------------------------------------


async def extract_latest_article(site_url: str, debug: bool = False) -> dict:
	"""Open site_url, navigate to the newest article and return structured JSON."""

	SYSTEM_EXT = """
	You are INTELLIGENCE-HUNTER, an OSINT analyst agent.

	GOALS
	- Investigate the provided target URL for leaks indicators.
	- Analyze the website and find where the latest Leaks are being discussed.
	- Analyze each post and gather information.

	RULES
	- DO NOT ever go beyond the original domain. Subdomains are valid.
	- When you determine the current page is a Post call the tool `maybe_capture_screenshot`. Do not over-call; one shot per relevant page.
	- Call `done` when finished.

	Keep it crisp;. Be decisive.
	"""
	nav_block = ""
	if getattr(args, "nav_prompt", None):
		try:
			with open(args.nav_prompt, "r", encoding="utf-8") as f:
				steps_text = f.read().strip()
			if steps_text:
				nav_block = (
					"FOLLOW THESE NAVIGATION STEPS FIRST (verbatim):\n"
					f"{steps_text}\n\n"
					"Then proceed with analysis.\n\n"
				)
		except Exception as e:
			if debug:
				print(f"[DEBUG] Could not read --nav_prompt file: {e}")

	shots = Path("post_monitor_yolo_resources") / "screenshots"

	prompt = (
		nav_block +
		f'Navigate to {site_url} and find the top 1 most recent posts (based on the creation date) related to data breaches.'
		f"Click on it to open the full post page. Once loaded the Post, call `maybe_capture_screenshot`, scroll & extract ALL required information: "
		f'1. title: The article headline '
		f'2. author: The author of the post'
		f'3. url: The full URL of the post page '
		f'4. posting_time: The publication date/time as shown on the page '
		f"5. short_summary: A 10-word overview of the post's content "
		f'6. long_summary: A 100-word detailed summary of the post, if possible '
		f"7. sentiment: Classify as 'positive', 'neutral', or 'negative' based on the post tone. "
		f'8. screenshot_path: The path to the taken screenshot'
		f'When done analyzing the top 1 most recent posts, call the done action with success=True and put ALL extracted data in the text field '
		f'as valid JSON in this exact format: '
		f'{{"title": "...", "url": "...", "posting_time": "...", "short_summary": "...", "long_summary": "...", "sentiment": "positive|neutral|negative"}}'
	)

	llm = ChatAzureOpenAI(model="gpt-4.1")
	browser_session = BrowserSession(headless=not debug)

	agent = Agent(task=prompt, llm=llm, browser_session=browser_session, vision_detail_level="high", extend_system_message=SYSTEM_EXT, tools=tools)

	if debug:
		print(f'[DEBUG] Starting extraction from {site_url}')
		start = time.time()

	result = await agent.run(max_steps=25)
	screenshots = result.screenshots()
	paths = result.screenshot_paths()

	raw = result.final_result() if result else None
	if debug:
		print(f'[DEBUG] Raw result type: {type(raw)}')
		print(f'[DEBUG] Raw result: {raw[:500] if isinstance(raw, str) else raw}')
		print(f'[DEBUG] Extraction time: {time.time() - start:.2f}s')
		#print(f'[DEBUG] Screenshots {screenshots}')

	if isinstance(raw, dict):
		return {'status': 'success', 'data': raw}

	text = str(raw).strip() if raw else ''

	if '<json>' in text and '</json>' in text:
		text = text.split('<json>', 1)[1].split('</json>', 1)[0].strip()

	if text.lower().startswith('here is'):
		brace = text.find('{')
		if brace != -1:
			text = text[brace:]

	if text.startswith('```'):
		text = text.lstrip('`\n ')
		if text.lower().startswith('json'):
			text = text[4:].lstrip()

	def _escape_newlines(src: str) -> str:
		out, in_str, esc = [], False, False
		for ch in src:
			if in_str:
				if esc:
					esc = False
				elif ch == '\\':
					esc = True
				elif ch == '"':
					in_str = False
				elif ch == '\n':
					out.append('\\n')
					continue
				elif ch == '\r':
					continue
			else:
				if ch == '"':
					in_str = True
			out.append(ch)
		return ''.join(out)

	cleaned = _escape_newlines(text)

	def _try_parse(txt: str):
		try:
			return json.loads(txt)
		except Exception as e:
			print(e)
			return None

	data = _try_parse(cleaned)

	# Fallback: grab first balanced JSON object
	if data is None:
		brace = 0
		start = None
		for i, ch in enumerate(text):
			if ch == '{':
				if brace == 0:
					start = i
				brace += 1
			elif ch == '}':
				brace -= 1
				if brace == 0 and start is not None:
					candidate = _escape_newlines(text[start : i + 1])
					data = _try_parse(candidate)
					if data is not None:
						break

	if isinstance(data, dict):
		return {'status': 'success', 'data': data}
	return {'status': 'error', 'error': f'JSON parse failed. Raw head: {text[:200]}'}



# ---------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------


def load_seen_hashes(file_path: str = 'news_data.json') -> set:
	"""Load already-saved article URL hashes from disk for dedup across restarts."""
	if not os.path.exists(file_path):
		return set()
	try:
		with open(file_path) as f:
			items = json.load(f)
		return {entry['hash'] for entry in items if 'hash' in entry}
	except Exception:
		return set()


def save_article(article: dict, file_path: str = 'news_data.json'):
	"""Append article to disk with a hash for future dedup."""
	payload = {
		'hash': hashlib.md5(article['url'].encode()).hexdigest(),
		'pulled_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
		'data': article,
	}

	existing = []
	if os.path.exists(file_path):
		try:
			with open(file_path) as f:
				existing = json.load(f)
		except Exception:
			existing = []

	existing.append(payload)
	# Keep last 100
	existing = existing[-100:]

	with open(file_path, 'w') as f:
		json.dump(existing, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------
# CLI functions
# ---------------------------------------------------------


def _fmt(ts_raw: str) -> str:
	"""Format timestamp string"""
	try:
		return dtparser.parse(ts_raw).strftime('%Y-%m-%d %H:%M:%S')
	except Exception:
		return datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')


async def run_once(url: str, output_path: str, debug: bool):
	"""Run a single extraction and exit"""
	res = await extract_latest_article(url, debug)

	if res['status'] == 'success':
		art = res['data']
		url_val = art.get('url', '')
		hash_ = hashlib.md5(url_val.encode()).hexdigest() if url_val else None
		if url_val:
			save_article(art, output_path)
		ts = _fmt(art.get('posting_time', ''))
		sentiment = art.get('sentiment', 'neutral')
		emoji = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}.get(sentiment, '🟡')
		summary = art.get('short_summary', art.get('summary', art.get('title', '')))
		if debug:
			print(json.dumps(art, ensure_ascii=False, indent=2))
			print()
		print(f'[{ts}] - {emoji} - {summary}')
		if not debug:
			print()  # Only add spacing in non-debug mode
		return hash_
	else:
		print(f'Error: {res["error"]}')
		return None


async def monitor(url: str, interval: int, output_path: str, debug: bool):
	"""Continuous monitoring mode"""
	seen = load_seen_hashes(output_path)
	print(f'Monitoring {url} every {interval}s')
	print()

	while True:
		try:
			res = await extract_latest_article(url, debug)

			if res['status'] == 'success':
				art = res['data']
				url_val = art.get('url', '')
				hash_ = hashlib.md5(url_val.encode()).hexdigest() if url_val else None
				if hash_ and hash_ not in seen:
					seen.add(hash_)
					ts = _fmt(art.get('posting_time', ''))
					sentiment = art.get('sentiment', 'neutral')
					emoji = {'positive': '🟢', 'negative': '🔴', 'neutral': '🟡'}.get(sentiment, '🟡')
					summary = art.get('short_summary', art.get('title', ''))
					save_article(art, output_path)
					if debug:
						print(json.dumps(art, ensure_ascii=False, indent=2))
					print(f'[{ts}] - {emoji} - {summary}')
					if not debug:
						print()  # Add spacing between articles in non-debug mode
			elif debug:
				print(f'Error: {res["error"]}')

		except Exception as e:
			if debug:
				import traceback

				traceback.print_exc()
			else:
				print(f'Unhandled error: {e}')

		await asyncio.sleep(interval)


def main():
	"""Main entry point"""
	if args.once:
		asyncio.run(run_once(args.url, args.output, args.debug))
	else:
		try:
			asyncio.run(monitor(args.url, args.interval, args.output, args.debug))
		except KeyboardInterrupt:
			print('\nStopped by user')


if __name__ == '__main__':
	main()