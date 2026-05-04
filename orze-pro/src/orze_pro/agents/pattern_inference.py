"""LLM-backed pattern inferrer for the orze metric harvester.

Regex defaults in `orze.engine.metric_harvester` work for common metric
names / log formats, but break on anything exotic — prose sentences
with the metric mid-line, TensorBoard-style `val/ndcg 0.8`, JSON
lines, per-fold breakdowns, etc. Rather than expanding the hardcoded
pattern table, we ask a fast LLM to look at the actual log output and
propose Python regex patterns keyed to that specific training script.

Results are cached by the harvester keyed by `(train_script, mtime)`
so inference runs at most once per script per edit — typically a
single Haiku call per new train script ever touched by the project.

CALLING SPEC:
    infer_metric_patterns(train_script, log_text, metric_name) -> list[str]
        Returns a list of Python regex strings whose group(1) is the
        numeric value of `metric_name`. Empty list on failure (no
        patterns found, LLM error, missing CLI) — the caller caches
        empties to avoid retry storms.
"""
from __future__ import annotations

from orze_pro._gate import require_license; require_license()

import json
import logging
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List

logger = logging.getLogger("orze_pro.pattern_inference")

# Budget: keep prompt small. Haiku handles plenty with head+tail slice.
_HEAD_LINES = 80
_TAIL_LINES = 120


def _sample_log(log_text: str) -> str:
    lines = log_text.splitlines()
    if len(lines) <= _HEAD_LINES + _TAIL_LINES:
        return log_text
    head = "\n".join(lines[:_HEAD_LINES])
    tail = "\n".join(lines[-_TAIL_LINES:])
    return (
        f"{head}\n"
        f"... ({len(lines) - _HEAD_LINES - _TAIL_LINES} lines elided) ...\n"
        f"{tail}"
    )


def _build_prompt(train_script: Path, log_sample: str,
                  metric_name: str) -> str:
    return f"""\
You are a log-format reverse-engineer. You inspect a training script's
stdout sample and produce Python regex patterns that capture a named
numeric metric from it, per evaluation / epoch.

Training script: {train_script.name}
Metric to capture: `{metric_name}`

Log sample:
```
{log_sample}
```

Produce 1-3 Python regex patterns that match lines where this metric
is reported. Each pattern MUST:

- Capture the numeric value as group(1). Use `([0-9]+\\.?[0-9]*)` or
  similar — do NOT include trailing punctuation in the capture.
- Be specific enough to avoid capturing unrelated numbers on other
  lines. Include a distinguishing token from the log format (the
  metric name, a prefix like `test_`, `val/`, a phrase like
  `came in at`, etc).
- Work when run with `re.IGNORECASE`.

If the metric does NOT appear in the log sample, return an empty list.

Return ONLY a JSON array of strings. No prose, no code fences, no
explanation. Example output for "map":

["test_mAP\\\\s*=\\\\s*([0-9]+\\\\.?[0-9]*)", "val_mAP\\\\s*:\\\\s*([0-9]+\\\\.?[0-9]*)"]
"""


def _extract_json_array(text: str) -> str:
    """Return the first balanced `[...]` block in `text`, or ''."""
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "[":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0 and start >= 0:
                return text[start:i + 1]
    return ""


def _parse_patterns(response: str) -> List[str]:
    """Extract the JSON array of patterns from an LLM response."""
    if not response:
        return []
    cleaned = re.sub(r"```(?:json)?\s*", "", response).replace("```", "").strip()

    # Try direct parse first — fastest path when the model did the right thing.
    data = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to balanced-bracket extraction if the model wrapped
        # the JSON in prose.
        block = _extract_json_array(cleaned)
        if block:
            try:
                data = json.loads(block)
            except json.JSONDecodeError:
                data = None

    if not isinstance(data, list):
        return []

    out: List[str] = []
    for item in data:
        if not isinstance(item, str):
            continue
        try:
            pat = re.compile(item, re.IGNORECASE)
        except re.error:
            continue
        if pat.groups < 1:
            continue
        out.append(item)
    return out


def infer_metric_patterns(train_script: Path,
                          log_text: str,
                          metric_name: str,
                          timeout: int = 60,
                          model: str = "haiku",
                          claude_bin: str = "") -> List[str]:
    """Call Claude CLI to infer regex patterns for a metric in a log format.

    Returns an empty list on any failure mode (no CLI, bad output, timeout).
    Safe to call from the orchestrator loop — blocking but rate-limited
    by the harvester's per-script cache.
    """
    if not log_text.strip():
        return []

    sample = _sample_log(log_text)
    prompt = _build_prompt(train_script, sample, metric_name)

    binary = claude_bin or shutil.which("claude") or "claude"
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)

    cmd = [
        binary, "-p", prompt,
        "--dangerously-skip-permissions",
        "--output-format", "text",
        "--model", model,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=timeout, env=env,
        )
    except FileNotFoundError:
        logger.debug("claude CLI not found — skipping pattern inference")
        return []
    except subprocess.TimeoutExpired:
        logger.info("pattern_inference timed out for %s/%s",
                    train_script.name, metric_name)
        return []
    except Exception as e:
        logger.debug("pattern_inference error: %s", e)
        return []

    response = (result.stdout or "").strip()
    patterns = _parse_patterns(response)
    if patterns:
        logger.info(
            "pattern_inference: %d pattern(s) learned for %s/%s",
            len(patterns), train_script.name, metric_name)
    else:
        logger.debug(
            "pattern_inference: LLM returned no usable patterns for %s/%s "
            "(response=%r)", train_script.name, metric_name, response[:200])
    return patterns
