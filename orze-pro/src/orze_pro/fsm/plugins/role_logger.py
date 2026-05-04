from __future__ import annotations
"""Role activity logger: parses orze.log for role events and writes to JSONL.

Captures:
- Role launches (research, professor, code_evolution, meta_research)
- Role completions (success/failure/timeout)
- Ideas ingested
- Experiments completed/failed
- Anomalies detected

Runs as a guard+action: guard checks if there are new log lines,
action parses them into structured JSONL events.
"""

import json
import logging
import re
import time
from pathlib import Path

from fsm.engine import guard, action, Context, _log_event

logger = logging.getLogger("fsm")

# Patterns to extract from orze.log
PATTERNS = [
    (r"Running (\w+) \[(\w+)\] \(cycle (\d+)\)", "role_start"),
    (r"(\w+) cycle (\d+) completed", "role_complete"),
    (r"\[ROLE TIMEOUT\] (\w+) after (.+) — killing", "role_timeout"),
    (r"(\w+) has failed (\d+) consecutive times", "role_failure_streak"),
    (r"\[COMPLETED\] (idea-\w+) on GPU (\d+)", "experiment_complete"),
    (r"\[ANOMALY\] (.+)", "anomaly"),
    (r"Ingested (\d+) new ideas from", "ideas_ingested"),
    (r"Launching (idea-\w+) on GPU (\d+): (.+)", "experiment_launch"),
    (r"Retrospection triggered: (\d+) completed", "retrospection"),
    (r"research cycle (\d+) exited .+ but ideas.md was not modified", "research_zero_output"),
]


@guard("new_log_lines")
def new_log_lines(ctx: Context) -> str | None:
    """True when orze.log has new content since last parse."""
    log_path = ctx.results_dir / "orze.log"
    if not log_path.exists():
        return None
    try:
        current_size = log_path.stat().st_size
        last_size = ctx.vars.get("_last_log_size", 0)
        if current_size > last_size:
            return f"{current_size - last_size} new bytes in orze.log"
    except Exception:
        pass
    return None


@action("parse_role_activity")
def parse_role_activity(ctx: Context):
    """Parse new orze.log lines into structured JSONL events."""
    log_path = ctx.results_dir / "orze.log"
    if not log_path.exists():
        return

    last_size = ctx.vars.get("_last_log_size", 0)
    try:
        current_size = log_path.stat().st_size
        if current_size <= last_size:
            return

        # Read only new bytes
        with open(log_path, "r", encoding="utf-8") as f:
            f.seek(last_size)
            new_lines = f.read()

        ctx.vars["_last_log_size"] = current_size
    except Exception:
        return

    event_count = 0
    for line in new_lines.strip().split("\n"):
        if not line:
            continue
        for pattern, event_type in PATTERNS:
            m = re.search(pattern, line)
            if m:
                event = {
                    "event": event_type,
                    "fsm": "role_logger",
                    "groups": m.groups(),
                    "line": line.strip()[:200],
                }
                _log_event(event)
                event_count += 1
                break

    if event_count > 0:
        ctx.vars["_total_events_parsed"] = ctx.vars.get("_total_events_parsed", 0) + event_count
