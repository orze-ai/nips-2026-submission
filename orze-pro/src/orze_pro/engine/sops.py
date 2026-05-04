"""Built-in Standard Operating Procedures (SOPs) for the orze engine.

Tier 1 SOPs are hardcoded, always-on engine behaviors. They run without
orze-pro and protect the pipeline from common failure modes.

CALLING SPEC:
    validate_idea(train_script, idea_config, python) -> (bool, str)
        Check idea config keys against train script's argparse.
        Returns (is_valid, error_message). Cached by train_script mtime.

    analyze_failure_feedback(idea_dir, results_dir, cfg) -> None
        Read failure_analysis.json + log tail, write _failure_feedback.md.

    analyze_method(name, source_dir, results_dir) -> Optional[Path]
        Extract basic method spec from source code, write _methods/<name>.yaml.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, FrozenSet, Optional, Tuple

logger = logging.getLogger("orze")

# ---------------------------------------------------------------------------
# validate_idea — argparse pre-check (Tier 1 SOP #1)
# ---------------------------------------------------------------------------

# Cache: {(script_path, mtime): frozenset_of_valid_args}
_args_cache: Dict[Tuple[str, float], FrozenSet[str]] = {}

# Keys that are always valid (orze injects these, not part of train script)
_ORZE_INJECTED_ARGS = frozenset({
    "idea_id", "idea-id", "results_dir", "results-dir",
    "ideas_md", "ideas-md", "config", "gpu",
    "train_script",  # meta key, not passed to script
})


def _cache_key(train_script: str) -> Tuple[str, float]:
    """Return (resolved_path, mtime) for cache invalidation."""
    p = Path(train_script).resolve()
    try:
        return (str(p), p.stat().st_mtime)
    except OSError:
        return (str(p), 0.0)


def _get_valid_args(train_script: str, python: str) -> FrozenSet[str]:
    """Parse --help output to extract valid argument names. Cached by mtime.

    Runs `python train_script --help` and extracts --flag names from the
    output. This is cheap (~500ms, no GPU/data loading) because --help
    exits before any heavy imports via argparse's sys.exit(0).
    """
    key = _cache_key(train_script)
    cached = _args_cache.get(key)
    if cached is not None:
        return cached

    try:
        result = subprocess.run(
            [python, train_script, "--help"],
            capture_output=True, text=True, timeout=15,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
        text = result.stdout + result.stderr
    except (subprocess.TimeoutExpired, OSError) as e:
        logger.warning("validate_idea: --help failed for %s: %s", train_script, e)
        # Can't validate → allow everything
        _args_cache[key] = frozenset()
        return frozenset()

    # Extract --flag-name patterns from help text
    # Matches: --flag_name, --flag-name, --flag
    flags = set()
    for m in re.finditer(r'--([a-zA-Z][a-zA-Z0-9_-]*)', text):
        raw = m.group(1)
        # Normalize: --flag-name and --flag_name are the same arg
        flags.add(raw.replace("-", "_"))
        flags.add(raw.replace("_", "-"))
        flags.add(raw)  # keep original too

    # If the script uses --config (YAML-based config), also extract
    # cfg.get("key") patterns from source code. Many train scripts pass
    # all experiment config through a YAML file rather than CLI args.
    # Also scan sibling .py files (e.g. strategies/*.py, lora_trainer.py)
    # because train.py often dispatches config to downstream modules that
    # consume their own keys — those keys would otherwise be flagged as
    # "unrecognized args" even though they're valid.
    if "config" in flags:
        # cfg access patterns: <var>.get("k"), <var>["k"], <var>.setdefault("k"..),
        # <var>.pop("k"..).  Accept any of the common "config-dict" variable
        # names used by train scripts — defaulting the match to literal `cfg`
        # was rejecting ideas whenever the script renamed its merged config
        # (e.g. `hp`, `params`, `args`) inside a helper function.
        _CFG_VAR_NAMES = (
            r"(?:cfg|config|configs|hp|hparams|params|args|opts|"
            r"options|kwargs|settings|conf)"
        )
        cfg_key_re = re.compile(
            rf'''{_CFG_VAR_NAMES}(?:\.(?:get|setdefault|pop)\(|\[)\s*["']([a-zA-Z_][a-zA-Z0-9_]*)["']'''
        )
        # Dict-literal key patterns: "key": ...  or  'key': ...
        # Used when idea generators produce metadata keys (e.g. "training_proposal": True)
        # that are project vocabulary but not read via cfg.get.
        dict_key_re = re.compile(
            r'''["']([a-zA-Z_][a-zA-Z0-9_]*)["']\s*:\s*'''
        )
        script_dir = Path(train_script).resolve().parent
        # Scan the train script itself plus sibling .py files up to depth 2.
        # Cap at 200 files to keep worst-case cost bounded.
        scan_files = [Path(train_script)]
        try:
            for p in script_dir.rglob("*.py"):
                # Skip the venv / site-packages / build / __pycache__ noise.
                parts = set(p.parts)
                if parts & {"venv", "site-packages", "__pycache__",
                            "build", "dist", ".git"}:
                    continue
                # Depth cap relative to script_dir.
                try:
                    rel_depth = len(p.relative_to(script_dir).parts)
                except ValueError:
                    continue
                if rel_depth > 3:
                    continue
                scan_files.append(p)
                if len(scan_files) >= 200:
                    break
        except OSError:
            pass

        seen: set = set()
        for f in scan_files:
            if f in seen:
                continue
            seen.add(f)
            try:
                src = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for m in cfg_key_re.finditer(src):
                k = m.group(1)
                flags.add(k)
                flags.add(k.replace("_", "-"))
            for m in dict_key_re.finditer(src):
                k = m.group(1)
                flags.add(k)
                flags.add(k.replace("_", "-"))

    valid = frozenset(flags)
    _args_cache[key] = valid
    logger.debug("validate_idea: cached %d valid args for %s", len(valid), train_script)
    return valid


def validate_idea(train_script: str, idea_config: dict,
                  python: str = "") -> Tuple[bool, str]:
    """Validate idea config keys against train script's argparse.

    Policy:

    * If `--config <path>` is an argparse flag on the script, the project
      owns its YAML schema — orze cannot reliably introspect it via
      heuristic source scanning. Default to ALLOW; log a debug message
      listing keys the scan didn't recognize, but never block an idea on
      them. Blocking here was the single biggest first-user footgun: a
      renamed helper-variable (`hp` instead of `cfg`) rejected every idea
      and the engineer auto-fix couldn't help because the "bug" lived in
      the framework, not the script.

    * If the script does NOT take `--config` (every knob is a CLI flag),
      strict validation is well-defined and we keep it.

    Returns (is_valid, error_message). error_message is "" if valid.
    If the train script can't be parsed (missing, no --help), returns
    (True, "") to avoid blocking ideas unnecessarily.
    """
    if not python:
        python = sys.executable

    if not Path(train_script).exists():
        return False, f"train_script not found: {train_script}"

    valid_args = _get_valid_args(train_script, python)
    if not valid_args:
        # Couldn't parse --help → can't validate, allow through
        return True, ""

    def _variants(k):
        return {k, k.replace("-", "_"), k.replace("_", "-")}

    invalid = []
    for key in idea_config:
        normalized = str(key).replace("-", "_")
        if normalized in _ORZE_INJECTED_ARGS:
            continue
        if not _variants(key) & valid_args:
            invalid.append(key)

    if not invalid:
        return True, ""

    # Config-driven script: trust the YAML. The scan (in _get_valid_args)
    # already widened `valid_args` with project-side cfg.get/[] lookups;
    # anything still missing is almost always either a helper's renamed
    # config var or a dict-literal key the regex missed. Log and allow.
    if "config" in valid_args:
        logger.debug(
            "validate_idea: %s is config-driven; allowing unknown keys: %s",
            Path(train_script).name, ", ".join(invalid))
        return True, ""

    return False, f"unrecognized args for {Path(train_script).name}: {', '.join(invalid)}"


# ---------------------------------------------------------------------------
# analyze_failure_feedback — structured feedback (Tier 1 SOP #2)
# ---------------------------------------------------------------------------

def analyze_failure_feedback(idea_dir: Path, results_dir: Path,
                             cfg: dict) -> None:
    """Read failure_analysis.json + train_output.log tail, append
    actionable lesson to results/_failure_feedback.md.

    This file is consumed by research_context.py to inform future
    idea generation — closing the feedback loop between failures
    and research agents.
    """
    import json

    fa_path = idea_dir / "failure_analysis.json"
    log_path = idea_dir / "train_output.log"
    feedback_path = results_dir / "_failure_feedback.md"

    if not fa_path.exists():
        return

    try:
        fa = json.loads(fa_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return

    category = fa.get("category", "unknown")
    error = fa.get("what", fa.get("error", "unknown"))
    lesson = fa.get("lesson", "")
    idea_id = idea_dir.name

    # Read last 5 lines of training log for context
    log_tail = ""
    if log_path.exists():
        try:
            lines = log_path.read_text(encoding="utf-8").strip().splitlines()
            log_tail = "\n".join(lines[-5:])
        except OSError:
            pass

    # Build actionable feedback entry
    entry = (
        f"\n### {idea_id} — {category}\n"
        f"- **Error**: {error}\n"
    )
    if lesson:
        entry += f"- **Lesson**: {lesson}\n"
    if log_tail:
        entry += f"- **Log tail**: `{log_tail[:200]}`\n"

    # Append to feedback file (capped at 100 entries)
    try:
        existing = ""
        if feedback_path.exists():
            existing = feedback_path.read_text(encoding="utf-8")

        # Count existing entries
        n_entries = existing.count("### idea-") + existing.count("### mvit")
        if n_entries > 100:
            # Trim oldest entries (keep last 80)
            sections = re.split(r'\n(?=### )', existing)
            existing = "\n".join(sections[-80:])

        if not existing.startswith("# Failure Feedback"):
            existing = "# Failure Feedback\n\nActionable lessons from failed experiments.\n" + existing

        feedback_path.write_text(existing + entry, encoding="utf-8")
    except OSError as e:
        logger.warning("Failed to write failure feedback: %s", e)


# ---------------------------------------------------------------------------
# analyze_method — basic method extraction (Tier 1 SOP #3)
# ---------------------------------------------------------------------------

def analyze_method(name: str, source_dir: Path,
                   results_dir: Path) -> Optional[Path]:
    """Extract basic method spec from source code directory.

    Writes results/_methods/<name>.yaml with whatever can be extracted
    without LLM assistance (argparse defs, loss class names, optimizer
    configs). The professor (Tier 2) can enrich it later.

    Returns path to written file, or None on failure.
    """
    import yaml

    if not source_dir.exists():
        logger.warning("analyze_method: source_dir not found: %s", source_dir)
        return None

    methods_dir = results_dir / "_methods"
    methods_dir.mkdir(exist_ok=True)
    out_path = methods_dir / f"{name}.yaml"

    spec: dict = {
        "name": name,
        "source": str(source_dir),
        "extracted_at": __import__("datetime").datetime.now().isoformat(),
    }

    # Scan Python files for key patterns
    py_files = list(source_dir.rglob("*.py"))[:50]  # cap scan
    loss_classes = set()
    optimizers = set()
    argparse_args = set()

    for py_file in py_files:
        try:
            text = py_file.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # Find loss function/class definitions
        for m in re.finditer(r'class\s+(\w*[Ll]oss\w*)|def\s+(\w*loss\w*)', text):
            loss_classes.add(m.group(1) or m.group(2))

        # Find optimizer references
        for m in re.finditer(r'(Adam|AdamW|SGD|RMSprop|LAMB)\b', text):
            optimizers.add(m.group(1))

        # Find argparse add_argument calls
        for m in re.finditer(r'add_argument\(\s*["\']--([a-zA-Z_-]+)', text):
            argparse_args.add(m.group(1))

    if loss_classes:
        spec["loss_classes"] = sorted(loss_classes)
    if optimizers:
        spec["optimizers"] = sorted(optimizers)
    if argparse_args:
        spec["available_args"] = sorted(argparse_args)

    # Look for config files
    config_files = list(source_dir.rglob("*.yaml")) + list(source_dir.rglob("*.yml"))
    if config_files:
        spec["config_files"] = [str(f.relative_to(source_dir)) for f in config_files[:10]]

    try:
        out_path.write_text(yaml.dump(spec, default_flow_style=False, sort_keys=False),
                            encoding="utf-8")
        logger.info("Method spec written to %s", out_path)
        return out_path
    except OSError as e:
        logger.warning("Failed to write method spec: %s", e)
        return None
