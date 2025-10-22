from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]

SCRIPT_OPTIONS = {
    "Posts Monitor (YOLO)": "posts_monitor_yolo.py",
    "Posts Monitor (Instructed)": "posts_monitor_instructed.py",
}


def build_command(
    script: str,
    url: str,
    output: str,
    interval: int,
    run_once: bool,
    debug: bool,
    nav_prompt: str,
    extra_args: str,
) -> list[str]:
    command = [sys.executable, script, "--url", url, "--output", output]
    if run_once:
        command.append("--once")
    else:
        command.extend(["--interval", str(interval)])
    if debug:
        command.append("--debug")
    if nav_prompt:
        command.extend(["--nav_prompt", nav_prompt])
    if extra_args:
        command.extend(shlex.split(extra_args))
    return command


st.title("🚀 Launch Scan")
st.caption("Run the monitoring scripts directly from the UI. Use `--once` for quick spot checks.")

with st.form("scan_form", clear_on_submit=False):
    script_label = st.selectbox("Script", list(SCRIPT_OPTIONS.keys()))
    default_url = "https://www.techcrunch.com"
    url = st.text_input("Target URL", value=default_url)
    output = st.text_input("Output file", value="news_data.json")
    run_once = st.checkbox("Run once (--once)", value=True)
    interval = st.number_input("Polling interval (--interval)", min_value=60, value=300, step=60, disabled=run_once)
    debug = st.checkbox("Enable debug (--debug)")
    nav_prompt = st.text_input("Navigation prompt file (--nav_prompt)", value="")
    extra_args = st.text_input("Additional arguments", placeholder="e.g. --interval 900 --some-flag")

    submitted = st.form_submit_button("Run scan")

    if submitted:
        script_path = SCRIPT_OPTIONS[script_label]
        resolved_script = ROOT_DIR / script_path
        if not resolved_script.exists():
            st.error(f"Script not found: {resolved_script}")
        else:
            nav_arg = nav_prompt.strip()
            if nav_arg:
                nav_candidate = (ROOT_DIR / nav_arg).resolve()
                if not nav_candidate.exists():
                    st.warning(f"Navigation prompt file not found: {nav_candidate}")

            command = build_command(
                script=script_path,
                url=url.strip() or default_url,
                output=output.strip() or "news_data.json",
                interval=int(interval),
                run_once=run_once,
                debug=debug,
                nav_prompt=nav_prompt.strip(),
                extra_args=extra_args.strip(),
            )

            st.write("Executing:")
            st.code(" ".join(shlex.quote(part) for part in command))

            try:
                with st.spinner("Running scan..."):
                    completed = subprocess.run(
                        command,
                        cwd=ROOT_DIR,
                        check=False,
                        text=True,
                        capture_output=True,
                    )
            except Exception as exc:
                st.error(f"Failed to start scan: {exc}")
            else:
                if completed.returncode == 0:
                    st.success("Scan finished successfully.")
                else:
                    st.error(f"Scan exited with status {completed.returncode}.")

                if completed.stdout:
                    st.write("### Stdout")
                    st.code(completed.stdout)
                if completed.stderr:
                    st.write("### Stderr")
                    st.code(completed.stderr)
