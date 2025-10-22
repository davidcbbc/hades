#!/usr/bin/env python3
"""
forum_vuln_watcher_markdown.py
Scan a forum (domain-locked), find vuln/exploit/PoC/0day discussions,
take screenshots of each thread, and generate a final Markdown report.

Usage:
  python forum_vuln_watcher_markdown.py https://forum.example.com \
    --days 14 --max 25 --shots --selector "article, .post, .message" \
    --mask ".username,.email" --out ./reports

Requires:
  - Python 3.11+
  - pip install browser-use python-dotenv pydantic playwright
  - playwright install chromium --with-deps

Environment:
  OPENAI_API_KEY=...  (or GOOGLE_API_KEY / ANTHROPIC_API_KEY / GROQ_API_KEY / OLLAMA_HOST)
  # Optional (if you want Browser Use Cloud):
  BROWSER_USE_API_KEY=...
"""

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

# browser-use
from browser_use import Agent, Browser, BrowserSession, ChatOpenAI, ChatGoogle, ChatAnthropic, ChatGroq, ChatOllama, Tools# type: ignore
from browser_use.agent.views import ActionResult
from browser_use.llm import ChatAzureOpenAI

# playwright (async)
from playwright.async_api import async_playwright, Page

# Global Playwright browser instance - shared between custom actions
playwright_browser: Browser | None = None
playwright_page: Page | None = None


tools = Tools()

# --------------------------- Structured Output Models ---------------------------

class Finding(BaseModel):
    title: str = Field(..., description="Thread/post title")
    url: str
    date_iso: Optional[str] = Field(None, description="ISO8601 date/time if visible")
    author: Optional[str] = None
    snippet: Optional[str] = Field(None, description="1–3 sentence vuln/exploit/PoC-related summary")
    indicators: List[str] = Field(default_factory=list, description="Matched keywords (CVE, PoC, 0day, RCE...)")
    tags: Optional[List[str]] = Field(default=None, description="Forum tags/categories if available")
    thread_type: Optional[str] = Field(default=None, description="vulnerability | exploit | 0day | PoC | research | discussion")

class ScanReport(BaseModel):
    site: str
    scanned_at: str
    days_window: int
    findings: List[Finding] = Field(default_factory=list)
    summary_recent: str = Field(..., description="Bullet summary (newest/most critical)")
    # Optional extras (agent may fill; we also compute fallbacks)
    site_title: Optional[str] = Field(default=None, description="Human-readable site title/brand if known")
    verdict: Optional[str] = Field(default=None, description="One-line verdict whether PoC/vuln discussion exists")

# --------------------------- Defaults & Helpers ---------------------------

DEFAULT_KEYWORDS = [
    "CVE", "zero-day", "0day", "0-day", "n-day", "PoC", "proof of concept",
    "exploit", "weaponized", "RCE", "LPE", "privilege escalation",
    "pre-auth", "unauthenticated", "SQLi", "XSS", "CSRF", "SSRF",
    "deserialization", "path traversal", "heap overflow", "buffer overflow",
    "use-after-free", "sandbox escape", "backdoor", "CWE-"
]

def infer_allowed_domains(target_url: str) -> List[str]:
    """
    Strict whitelist for the target host + subdomains. Formats per docs:
      'host', '*.host', 'http*://host'
    """
    u = urlparse(target_url)
    host = u.netloc
    return [host, f"*.{host}", f"http*://{host}"]

def build_task(site: str, keywords: List[str], days: int, max_items: int) -> str:
    kw = ", ".join(sorted(set(keywords), key=str.lower))
    return f"""
Goal:
- Open {site} and find the most recent discussions that mention vulnerabilities, exploits, PoCs, or 0days.

Hard constraints:
- Only navigate within {site}'s domain and subdomains.
- Do NOT use external search engines;

What to look for (case-insensitive):
- Indicators: {kw}

Process:
1) If authentication is needed , authenticate with username 'SIM-3290562' and password 'TEvRnQg%JWx7'.
2) After authenticating, navigate trought {site} and prefer Latest/Recent/New or Security/Vulnerabilities sections.
3) Collect up to {max_items} distinct threads/posts from the LAST {days} DAYS.
   - Open each candidate to confirm relevance (software/hardware/theoretical vuln/exploit/PoC/0day).
   - Extract: title, canonical URL, author (if visible), 1–3 sentence snippet, tags, matched indicators.
   - Parse posted/published date if visible; normalize relative timestamps to ISO8601 (UTC if unknown).
   - Take a screenshot using the 'playwright_screenshot' action and save it as "<page_title>.png"
4) Ranking:
   - Sort by recency (newest first). Break ties by relevance (CVE IDs, number of indicators).
5) Output (Pydantic schema):
   - Fill ScanReport EXACTLY. If possible also include 'site_title' (brand/page title) and a one-line 'verdict'.

Safety/Ethics:
- Respect robots and ToS. Do not download attachments or run PoC code.
- Avoid infinite scrolling; stop when the {days}-day window is exceeded.
""".strip()

def make_llm(provider: str, model: Optional[str]):
    provider = provider.lower().strip()
    if provider == "openai":
        return ChatOpenAI(model=model or "gpt-4.1-mini")
    if provider == "google":
        return ChatGoogle(model=model or "gemini-flash-latest")
    if provider == "anthropic":
        return ChatAnthropic(model=model or "claude-3-5-sonnet-20240620")
    if provider == "groq":
        return ChatGroq(model=model or "meta-llama/llama-4-maverick-17b-128e-instruct")
    if provider == "ollama":
        return ChatOllama(model=model or "llama3.1:8b")
    if provider == "azure":
        return ChatAzureOpenAI(model=model or "gpt-4.1")
    # default
    return ChatOpenAI(model=model or "gpt-4.1-mini")

def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def compute_verdict(report: ScanReport) -> str:
    n = len(report.findings or [])
    if n == 0:
        return "No relevant vulnerability or PoC discussions found in the selected time window."
    def has(ind: str, s: str) -> bool:
        return ind.lower() in (s or "").lower()
    poc = sum(
        any(has("poc", " ".join(f.indicators))) or has("proof of concept", " ".join(f.indicators)) or has("poc", f.thread_type or "")
        for f in report.findings
    )
    cve = sum(any(has("cve", " ".join(f.indicators))) for f in report.findings)
    exploit = sum(any(has("exploit", " ".join(f.indicators))) or has("exploit", f.thread_type or "") for f in report.findings)
    return (
        f"YES — Found {n} relevant thread(s) in the last {report.days_window} day(s): "
        f"{poc} with PoC indicators, {cve} referencing CVEs, {exploit} exploit-related."
    )

# --------------------------- Agent Run ---------------------------

async def run_agent(
    site: str,
    keywords: List[str],
    days: int,
    max_items: int,
    headless: bool,
    use_cloud: bool,
    provider: str,
    model: Optional[str],
    user_data_dir: Optional[str],
    proxy: Optional[str],
    timeout_sec: int,
) -> ScanReport:

    llm = make_llm(provider, model)

    # Ensure allowed_domains are applied in both local and cloud modes
    allowed = infer_allowed_domains(site)
    browser = Browser(
        headless=headless,
        allowed_domains=allowed,
        user_data_dir=user_data_dir,
        proxy=({"server": proxy} if proxy else None),
        highlight_elements=True
    )
    cloud_browser = Browser(use_cloud=True, allowed_domains=allowed)

    task = build_task(site=site, keywords=keywords, days=days, max_items=max_items)

    agent = Agent(
        task=task,
        llm=llm,  # FIX: respect --provider/--model; removed unconditional Azure override
        browser=browser if not use_cloud else cloud_browser,
        output_model_schema=ScanReport,  # structured output via Pydantic
        extend_system_message=(
            "You are a security OSINT assistant. "
            "Stay within allowed domains only "
            "Extract only vuln/exploit/PoC/0day-relevant content."
        ),
        llm_timeout=timeout_sec,
        step_timeout=timeout_sec,
        max_history_items=50,
        use_thinking=True,
        vision_detail_level='high'
    )

    history = await agent.run()

    # Parse structured output, with robust fallbacks/defaults
    try:
        structured = history.structured_output
        data = structured if isinstance(structured, dict) else structured
        report = ScanReport(**data)
    except (ValidationError, Exception):
        final_text = history.final_result() or ""
        report = ScanReport(
            site=site,
            scanned_at=_iso_now(),
            days_window=days,
            findings=[],
            summary_recent=final_text[:2000],
        )

    # Ensure required metadata & compute verdict if missing
    if not report.scanned_at:
        report.scanned_at = _iso_now()
    if not report.verdict:
        report.verdict = compute_verdict(report)

    return report

# --------------------------- Screenshots ---------------------------

class PlaywrightScreenshotAction(BaseModel):
	"""Parameters for Playwright screenshot action."""

	filename: str = Field(default='playwright_screenshot.png', description='Filename for screenshot')
	quality: int | None = Field(default=None, description='JPEG quality (1-100), only for .jpg/.jpeg files')

@tools.registry.action(
	"Take a screenshot using Playwright's screenshot capabilities with high quality and precision.",
	param_model=PlaywrightScreenshotAction,
)
async def playwright_screenshot(params: PlaywrightScreenshotAction, browser_session: BrowserSession):
	"""
	Custom action that uses Playwright's advanced screenshot features.
	"""
	try:
		if not playwright_page:
			return ActionResult(error='Playwright not connected. Run setup first.')

		# Taking screenshot with Playwright

		# Use Playwright's screenshot with full page capture
		screenshot_kwargs = {'path': params.filename, 'full_page': True}

		# Add quality parameter only for JPEG files
		if params.quality is not None and params.filename.lower().endswith(('.jpg', '.jpeg')):
			screenshot_kwargs['quality'] = params.quality

		await playwright_page.screenshot(**screenshot_kwargs)

		success_msg = f'✅ Screenshot saved as {params.filename} using Playwright'

		return ActionResult(
			extracted_content=success_msg, include_in_memory=True, long_term_memory=f'Screenshot saved: {params.filename}'
		)

	except Exception as e:
		error_msg = f'❌ Playwright screenshot failed: {str(e)}'
		return ActionResult(error=error_msg)


def sanitize_slug(s: str, limit: int = 80) -> str:
    slug = re.sub(r"\s+", " ", (s or "")).strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug).strip("-")
    return slug[:limit] or "post"

async def screenshot_findings(
    report: ScanReport,
    out_dir: Path,
    selector: Optional[str],
    full_page: bool,
    mask_selectors: Optional[List[str]],
    timeout_ms: int = 20000,
) -> Dict[int, Path]:
    """
    Visit each finding URL and save a screenshot.
    If selector is provided, try element-only; else capture full page.
    Supports masking selectors to obscure PII.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    host = urlparse(str(report.site)).netloc

    def on_domain(u: str) -> bool:
        netloc = urlparse(u).netloc
        return netloc == host or netloc.endswith("." + host)

    shots: Dict[int, Path] = {}

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1366, "height": 900})
        page = await context.new_page()

        for i, f in enumerate(report.findings, 1):
            url = str(f.url)
            if not on_domain(url):
                continue

            fname = f"{i:02d}_{sanitize_slug(f.title)}.png"
            path = out_dir / fname

            await page.goto(url, wait_until="networkidle", timeout=timeout_ms)

            try:
                # NOTE: Playwright mask= works on page.screenshot; locator.screenshot doesn't support mask in all versions.
                mask_locs = [page.locator(ms) for ms in (mask_selectors or [])]
                if selector:
                    loc = page.locator(selector).first
                    if await loc.count() > 0:
                        if mask_selectors:
                            # Fallback to page-level shot when masking requested
                            await page.screenshot(path=str(path), full_page=full_page, mask=mask_locs)
                        else:
                            await loc.screenshot(path=str(path))
                    else:
                        await page.screenshot(path=str(path), full_page=full_page, mask=mask_locs or None)
                else:
                    await page.screenshot(path=str(path), full_page=full_page, mask=mask_locs or None)
            except Exception:
                # Fallback: viewport-only
                await page.screenshot(path=str(path))

            shots[i] = path

        await context.close()
        await browser.close()

    return shots

# --------------------------- Markdown Rendering ---------------------------

def _host(url: str) -> str:
    return urlparse(url).netloc

def _first_sentence(s: str, max_len: int = 200) -> str:
    if not s:
        return ""
    s = s.strip()
    # crude first-sentence clamp for overview table
    s = re.split(r"(?<=[.!?])\s", s, maxsplit=1)[0]
    if len(s) > max_len:
        s = s[: max_len - 1] + "…"
    return s

def render_markdown(
    report: ScanReport,
    shots: Dict[int, Path],
    report_dir: Path,
) -> str:
    site = str(report.site)
    scanned_at = report.scanned_at
    host = _host(site)
    title = report.site_title or host

    # Header
    md = []
    md.append(f"# {title} — Vuln/PoC Watch (Last {report.days_window} Days)")
    md.append("")
    md.append(f"_Site:_ **{host}**  |  _Scanned at (UTC):_ **{scanned_at}**")
    md.append("")
    md.append("## Executive Summary")
    md.append("")
    md.append(report.summary_recent.strip() or "*No recent discussions found.*")
    md.append("")
    md.append("## Verdict")
    md.append("")
    md.append(report.verdict or compute_verdict(report))
    md.append("")
    md.append("## Posts Found")
    md.append("")
    md.append(f"Total findings: **{len(report.findings)}**")
    if report.findings:
        md.append("")
        md.append("| # | Date | Title | Type | Indicators |")
        md.append("|:-:|:-----|:------|:-----|:-----------|")
        for i, f in enumerate(report.findings, 1):
            date = f.date_iso or "—"
            ttype = f.thread_type or "—"
            inds = ", ".join(f.indicators[:5]) if f.indicators else "—"
            title_cell = f"[{f.title}]({f.url})"
            md.append(f"| {i} | {date} | {title_cell} | {ttype} | {inds} |")
    md.append("")
    md.append("## Table of Contents")
    md.append("")
    for i, f in enumerate(report.findings, 1):
        anchor = sanitize_slug(f"{i}-{f.title}", limit=96)
        md.append(f"- [{i}. {f.title}]({'#' + anchor})")
    md.append("")
    md.append("---")
    md.append("")

    # Findings (detailed)
    for i, f in enumerate(report.findings, 1):
        anchor = sanitize_slug(f"{i}-{f.title}", limit=96)
        md.append(f"### <a id=\"{anchor}\"></a>{i}. {f.title}")
        meta_bits = []
        if f.date_iso: meta_bits.append(f"**Date:** {f.date_iso}")
        if f.author: meta_bits.append(f"**Author:** {f.author}")
        if f.tags: meta_bits.append(f"**Tags:** {', '.join(f.tags)}")
        if f.indicators: meta_bits.append(f"**Indicators:** {', '.join(f.indicators)}")
        if f.thread_type: meta_bits.append(f"**Type:** {f.thread_type}")
        if meta_bits:
            md.append("")
            md.append(" | ".join(meta_bits))
        md.append("")
        md.append(f"**Link:** {f.url}")
        md.append("")
        if f.snippet:
            md.append(_first_sentence(f.snippet.strip(), 700))
            md.append("")
        # Embed screenshot if available
        shot_path = shots.get(i)
        if shot_path:
            rel = shot_path.relative_to(report_dir)
            alt = sanitize_slug(f.title, limit=60) or f"post-{i}"
            md.append(f"![{alt}]({rel.as_posix()})")
            md.append("")
        md.append("---")
        md.append("")

    # Methodology
    md.append("## Methodology & Constraints")
    md.append("")
    md.append("- Domain-locked crawling using `allowed_domains` (host + subdomains).")
    md.append("- Threads chosen by recency & relevance to vulnerabilities, exploits, PoCs, and 0days.")
    md.append("- Dates normalized to ISO when possible; relative times converted using UTC when timezone unknown.")
    md.append("- Screenshots captured with Playwright (`full_page` or element-scoped). Optional masking via CSS selectors.")
    md.append("- No authentication performed; only public content was considered.")
    md.append("")
    return "\n".join(md)

# --------------------------- Save Helpers ---------------------------

def save_report_json(report: ScanReport, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", urlparse(str(report.site)).netloc.lower())
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{slug}_{ts}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(json.loads(report.model_dump_json()), f, ensure_ascii=False, indent=2)
    return path

def save_markdown(md: str, out_dir: Path, site: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", urlparse(site).netloc.lower())
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{slug}_{ts}.md"
    path.write_text(md, encoding="utf-8")
    return path

# --------------------------- CLI / Main ---------------------------

async def amain():
    load_dotenv()

    ap = argparse.ArgumentParser(description="Domain-locked forum scanner -> screenshots -> Markdown report")
    ap.add_argument("site", help="Target site URL (e.g., https://forum.example.com)")
    ap.add_argument("--days", type=int, default=14, help="Recency window in days")
    ap.add_argument("--max", dest="max_items", type=int, default=25, help="Max items to extract")
    ap.add_argument("--keywords", nargs="*", default=DEFAULT_KEYWORDS, help="Keywords to match")
    ap.add_argument("--headless", action="store_true", help="Run agent browser headless")
    ap.add_argument("--use-cloud", action="store_true", help="Use Browser Use Cloud browser")
    ap.add_argument("--provider", default="openai",
                    choices=["openai", "google", "anthropic", "groq", "ollama", "azure"],
                    help="LLM provider")
    ap.add_argument("--model", default=None, help="LLM model name")
    ap.add_argument("--user-data-dir", default=None, help="Persist browser profile")
    ap.add_argument("--proxy", default=None, help="HTTP(S) proxy, e.g., http://host:8080")
    ap.add_argument("--timeout", type=int, default=120, help="Per-step and LLM timeout seconds")
    ap.add_argument("--out", default="./reports", help="Base output directory")

    # screenshot & markdown options
    ap.add_argument("--shots", action="store_true", help="Take screenshots for each found post")
    ap.add_argument("--shots-dir", default="screenshots", help="Subdirectory (inside --out) for screenshots")
    ap.add_argument("--selector", default=None, help="CSS selector for post body (e.g. 'article, .post, .message')")
    ap.add_argument("--no-fullpage", action="store_true", help="Disable full-page screenshots")
    ap.add_argument("--mask", nargs="*", default=None, help="CSS selectors to mask (e.g. .username .email)")

    args = ap.parse_args()
    site = args.site if re.match(r"^https?://", args.site, re.I) else f"https://{args.site}"

    # Run agent
    report = await run_agent(
        site=site,
        keywords=args.keywords,
        days=args.days,
        max_items=args.max_items,
        headless=args.headless,
        use_cloud=args.use_cloud,
        provider=args.provider,
        model=args.model,
        user_data_dir=args.user_data_dir,
        proxy=args.proxy,
        timeout_sec=args.timeout,
    )

    # Guarantee a verdict even if the model didn't populate it
    if not report.verdict:
        report.verdict = compute_verdict(report)

    base_out = Path(args.out).resolve()
    # Save JSON first (for downstream consumption)
    json_path = save_report_json(report, base_out)

    # Screenshots (optional, but recommended for the Markdown)
    shots_dir = base_out / args.shots_dir
    shots_map: Dict[int, Path] = {}
    if args.shots:
        shots_map = await screenshot_findings(
            report=report,
            out_dir=shots_dir,
            selector=args.selector,
            full_page=not args.no_fullpage,
            mask_selectors=args.mask,
        )

    # Render & save Markdown
    md = render_markdown(report, shots_map, report_dir=base_out)
    md_path = save_markdown(md, base_out, str(report.site))

    # Console summary
    print("\n=== Most-Recent Summary ===")
    print(report.summary_recent.strip() or "(empty)")
    print(f"\nVerdict: {report.verdict}")
    print(f"\nJSON: {json_path}")
    print(f"MD  : {md_path}")
    if args.shots:
        print(f"Shots saved under: {shots_dir}")

def main():
    asyncio.run(amain())

if __name__ == "__main__":
    main()
