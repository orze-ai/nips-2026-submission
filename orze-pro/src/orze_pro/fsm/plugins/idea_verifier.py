from __future__ import annotations
"""Idea gatekeeper: two-tier filtering to prevent GPU waste without killing good ideas.

Tier 1 (rule-based, instant, conservative):
  - EXACT duplicate: same config hash + same seed as a completed experiment
  - Known-dead combos from RESEARCH_RULES.md (share_all_norms, tie_gate, d=2/hd=2)
  These are safe to filter — they are literally the same experiment repeated.

Tier 2 (LLM agent, periodic, nuanced):
  - Reviews borderline ideas that Tier 1 didn't catch
  - Has full context: leaderboard, recent code changes, experiment distribution
  - Can reason about whether a "similar" config exercises a new code path
  - Decides: APPROVE, SKIP (with reason), or PRIORITIZE (boost priority)

Design principle: NEVER filter an idea that might exercise a new code path.
False negatives (letting a bad idea through) cost 25min of GPU time.
False positives (killing a good idea) cost an entire research direction.
"""

import hashlib
import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

from fsm.engine import guard, action, Context

logger = logging.getLogger("fsm")


# ---------------------------------------------------------------------------
# Tier 1: Rule-based. Only catches EXACT duplicates and proven-dead configs.
# Conservative by design — if in doubt, let it through.
# ---------------------------------------------------------------------------

# Proven-dead configs loaded from RESEARCH_RULES.md at runtime.
# Fallback to empty list if the file doesn't define them.
PROVEN_DEAD = []  # populated by _load_proven_dead()


def _load_proven_dead(results_dir: Path) -> list:
    """Parse proven-dead config combos from RESEARCH_RULES.md.

    Looks for a YAML code block under a heading containing 'dead' or 'proven'
    (case-insensitive). Each list entry is a dict of config keys/values.
    Falls back to empty list if not found — Tier 2 (professor) handles nuance.
    """
    rules_path = results_dir.parent / ".orze" / "rules" / "RESEARCH_RULES.md"
    # Fallback to legacy location
    if not rules_path.exists():
        rules_path = results_dir.parent / "RESEARCH_RULES.md"
    if not rules_path.exists():
        return []
    try:
        import yaml
        text = rules_path.read_text()
        # Find YAML blocks after "dead" or "proven" headings
        pattern = r'(?i)(?:dead|proven)[^\n]*\n```ya?ml\n(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        dead = []
        for match in matches:
            parsed = yaml.safe_load(match)
            if isinstance(parsed, list):
                dead.extend(parsed)
        return dead
    except Exception:
        return []


def _config_fingerprint(config: dict) -> str:
    """Full config hash INCLUDING seed — only exact duplicates match."""
    cfg = {k: v for k, v in sorted(config.items())}
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]


def _has_novel_keys(config: dict, known_keys: set) -> bool:
    """True if config has keys not seen in any completed experiment.

    Novel keys likely exercise new code paths added by code evolution.
    NEVER filter ideas with novel keys.
    """
    return bool(set(config.keys()) - known_keys)


def _tier1_check(idea_id: str, config: dict,
                 completed_fingerprints: set,
                 known_keys: set,
                 results_dir: Path = None) -> Optional[str]:
    """Returns rejection reason or None. ONLY rejects certainties."""

    # Load proven-dead patterns from RESEARCH_RULES.md (cached per call site)
    global PROVEN_DEAD
    if not PROVEN_DEAD and results_dir:
        PROVEN_DEAD = _load_proven_dead(results_dir)

    # Reject proven-dead configs FIRST — these are architecturally broken
    # regardless of whether they have novel keys
    for dead in PROVEN_DEAD:
        if all(config.get(k) == v for k, v in dead.items()):
            return f"proven dead: {dead}"

    # Novel config keys likely exercise new code paths from code evolution.
    # Don't reject these as duplicates — they may behave differently.
    if _has_novel_keys(config, known_keys):
        return None

    # Reject exact duplicates (same config + same seed)
    fp = _config_fingerprint(config)
    if fp in completed_fingerprints:
        return f"exact duplicate (fingerprint {fp})"

    return None


# ---------------------------------------------------------------------------
# Tier 2 context builder: gives the LLM agent full picture
# ---------------------------------------------------------------------------

def _build_gatekeeper_context(results_dir: Path, queued_ideas: list) -> str:
    """Build context for the LLM gatekeeper with full experiment distribution."""
    lines = ["# Idea Gatekeeper Context\n"]

    # Leaderboard summary
    report = results_dir / "report.md"
    if report.exists():
        content = report.read_text(encoding="utf-8")
        # Extract top 10 from report
        table_lines = [l for l in content.split("\n") if l.startswith("|") and "idea-" in l]
        if table_lines:
            lines.append("## Top Results")
            for l in table_lines[:10]:
                lines.append(l)
            lines.append("")

    # Experiment distribution
    lake_path = results_dir.parent / "idea_lake.db"
    if not lake_path.exists():
        lake_path = results_dir / "idea_lake.db"
    if lake_path.exists():
        try:
            conn = sqlite3.connect(str(lake_path), timeout=5)

            # Approach family distribution
            rows = conn.execute(
                "SELECT approach_family, COUNT(*), "
                "SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) "
                "FROM ideas GROUP BY approach_family"
            ).fetchall()
            if rows:
                lines.append("## Approach Distribution")
                for family, total, completed in rows:
                    lines.append(f"- {family or 'other'}: {total} total, {completed} completed")
                lines.append("")

            # Accuracy distribution
            rows = conn.execute(
                "SELECT idea_id, title, eval_metrics FROM ideas "
                "WHERE status='completed' AND eval_metrics IS NOT NULL "
                "ORDER BY rowid DESC LIMIT 20"
            ).fetchall()
            if rows:
                lines.append("## Recent Completions")
                for iid, title, metrics_str in rows:
                    try:
                        m = json.loads(metrics_str) if metrics_str else {}
                        acc = m.get("accuracy", "?")
                        lines.append(f"- {iid}: {title[:50]} → acc={acc}")
                    except Exception:
                        pass
                lines.append("")
            conn.close()
        except Exception:
            pass

    # Recent code changes (from git)
    try:
        import subprocess
        result = subprocess.run(
            ["git", "log", "--oneline", "-5", "--", "train.py"],
            capture_output=True, text=True, timeout=5,
            cwd=str(results_dir.parent))
        if result.stdout.strip():
            lines.append("## Recent train.py Changes")
            lines.append(result.stdout.strip())
            lines.append("")
    except Exception:
        pass

    # Past Professor decisions (for consistency)
    decisions_file = results_dir / "_professor_decisions.jsonl"
    if decisions_file.exists():
        try:
            recent = decisions_file.read_text(encoding="utf-8").strip().splitlines()[-20:]
            if recent:
                lines.append("## Your Recent Decisions (stay consistent)")
                for line in recent:
                    try:
                        d = json.loads(line)
                        lines.append(f"- {d['idea_id']}: {d['decision']} — {d.get('reason', '')}")
                    except Exception:
                        continue
                lines.append("")
        except Exception:
            pass

    # The ideas to review
    lines.append("## Ideas to Review")
    for idea in queued_ideas:
        lines.append(f"\n### {idea['id']}: {idea['title']}")
        if idea.get("hypothesis"):
            lines.append(f"Hypothesis: {idea['hypothesis']}")
        lines.append(f"Config keys: {', '.join(sorted(idea['config'].keys()))}")
        # Highlight novel keys
        if idea.get("novel_keys"):
            lines.append(f"**NOVEL KEYS**: {', '.join(idea['novel_keys'])}")
        lines.append(f"Seed: {idea['config'].get('seed', '?')}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Guards & Actions
# ---------------------------------------------------------------------------

@guard("has_garbage_ideas")
def has_garbage_ideas(ctx: Context) -> str | None:
    """True when queued ideas contain Tier 1 filterable garbage."""
    lake_path = ctx.results_dir.parent / "idea_lake.db"
    if not lake_path.exists():
        lake_path = ctx.results_dir / "idea_lake.db"
    if not lake_path.exists():
        return None

    # Build completed fingerprints and known keys
    completed_fingerprints = set()
    known_keys = set()
    try:
        import yaml
        conn = sqlite3.connect(str(lake_path), timeout=5)
        rows = conn.execute(
            "SELECT config FROM ideas WHERE status = 'completed'"
        ).fetchall()
        for (config_str,) in rows:
            if not config_str:
                continue
            try:
                config = yaml.safe_load(config_str)
                if isinstance(config, dict):
                    completed_fingerprints.add(_config_fingerprint(config))
                    known_keys.update(config.keys())
            except Exception:
                continue

        # Check queued ideas
        rows = conn.execute(
            "SELECT idea_id, config FROM ideas WHERE status = 'queued'"
        ).fetchall()
        conn.close()
    except Exception:
        return None

    garbage_count = 0
    for idea_id, config_str in rows:
        if not config_str:
            continue
        try:
            config = yaml.safe_load(config_str)
            if not isinstance(config, dict):
                continue
        except Exception:
            continue
        if _tier1_check(idea_id, config, completed_fingerprints, known_keys, results_dir=ctx.results_dir):
            garbage_count += 1

    if garbage_count > 0:
        return f"{garbage_count} exact duplicates/dead configs in queue (of {len(rows)} queued)"
    return None


@action("filter_garbage_ideas")
def filter_garbage_ideas(ctx: Context):
    """Tier 1: Skip only exact duplicates and proven-dead configs."""
    lake_path = ctx.results_dir.parent / "idea_lake.db"
    if not lake_path.exists():
        lake_path = ctx.results_dir / "idea_lake.db"
    if not lake_path.exists():
        return

    completed_fingerprints = set()
    known_keys = set()
    try:
        import yaml
        conn = sqlite3.connect(str(lake_path), timeout=5)
        rows = conn.execute(
            "SELECT config FROM ideas WHERE status = 'completed'"
        ).fetchall()
        for (config_str,) in rows:
            if not config_str:
                continue
            try:
                config = yaml.safe_load(config_str)
                if isinstance(config, dict):
                    completed_fingerprints.add(_config_fingerprint(config))
                    known_keys.update(config.keys())
            except Exception:
                continue

        rows = conn.execute(
            "SELECT idea_id, config FROM ideas WHERE status = 'queued'"
        ).fetchall()
    except Exception:
        return

    filtered = 0
    for idea_id, config_str in rows:
        if not config_str:
            continue
        try:
            config = yaml.safe_load(config_str)
            if not isinstance(config, dict):
                continue
        except Exception:
            continue

        reason = _tier1_check(idea_id, config, completed_fingerprints, known_keys)
        if reason:
            try:
                conn.execute(
                    "UPDATE ideas SET status = 'skipped' WHERE idea_id = ?",
                    (idea_id,))
                idea_dir = ctx.results_dir / idea_id
                idea_dir.mkdir(exist_ok=True)
                (idea_dir / "metrics.json").write_text(json.dumps({
                    "status": "SKIPPED",
                    "skip_reason": reason,
                    "skipped_by": "tier1_filter",
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }, indent=2), encoding="utf-8")
                filtered += 1
                logger.info("Tier1 filtered %s: %s", idea_id, reason)
            except Exception as e:
                logger.error("Failed to filter %s: %s", idea_id, e)

    if filtered > 0:
        try:
            conn.commit()
        except Exception:
            pass
    try:
        conn.close()
    except Exception:
        pass

    ctx.vars["last_filtered_count"] = filtered
    ctx.vars["total_filtered"] = ctx.vars.get("total_filtered", 0) + filtered
    if filtered > 0:
        logger.info("Tier1 filtered %d exact duplicates/dead configs (total: %d)",
                     filtered, ctx.vars["total_filtered"])


# ---------------------------------------------------------------------------
# Tier 2: LLM Professor — nuanced review with full context
# ---------------------------------------------------------------------------

def _find_lake_path(results_dir: Path) -> Optional[Path]:
    """Locate idea_lake.db (may be in results_dir or its parent)."""
    for candidate in [results_dir.parent / "idea_lake.db",
                      results_dir / "idea_lake.db"]:
        if candidate.exists():
            return candidate
    return None


def _get_unreviewed_ideas(results_dir: Path, limit: int = 20) -> list:
    """Get queued ideas not yet reviewed by the Professor."""
    lake_path = _find_lake_path(results_dir)
    if not lake_path:
        return []

    decisions_file = results_dir / "_professor_decisions.jsonl"
    reviewed_ids = set()
    if decisions_file.exists():
        for line in decisions_file.read_text(encoding="utf-8").splitlines():
            try:
                reviewed_ids.add(json.loads(line)["idea_id"])
            except Exception:
                continue

    import yaml
    ideas = []
    known_keys = set()
    try:
        conn = sqlite3.connect(str(lake_path), timeout=5)
        rows = conn.execute(
            "SELECT idea_id, title, config, hypothesis FROM ideas "
            "WHERE status = 'queued' ORDER BY rowid DESC LIMIT ?",
            (limit * 2,),  # over-fetch to account for already-reviewed
        ).fetchall()
        # Build known keys for novel-key detection (same connection)
        completed = conn.execute(
            "SELECT config FROM ideas WHERE status = 'completed'"
        ).fetchall()
        conn.close()
        for (config_str,) in completed:
            if config_str:
                try:
                    c = yaml.safe_load(config_str)
                    if isinstance(c, dict):
                        known_keys.update(c.keys())
                except Exception:
                    pass
    except Exception:
        return []

    for idea_id, title, config_str, hypothesis in rows:
        if idea_id in reviewed_ids:
            continue
        config = {}
        if config_str:
            try:
                config = yaml.safe_load(config_str)
                if not isinstance(config, dict):
                    config = {}
            except Exception:
                config = {}
        novel = set(config.keys()) - known_keys if config else set()
        ideas.append({
            "id": idea_id,
            "title": title or "",
            "config": config,
            "hypothesis": hypothesis or "",
            "novel_keys": sorted(novel),
        })
        if len(ideas) >= limit:
            break

    return ideas


def _load_professor_prompt(results_dir: Path, ideas: list) -> str:
    """Build the full Professor prompt: compose bundled SOPs + context.

    The professor's bundled static SOPs live at orze_pro/sops/
    professor_*.skill.md. We compose them in the same order the
    auto-injection uses so the idea-verifier reasoning matches what
    the live professor role actually sees.
    """
    try:
        from orze.skills.loader import compose_skills
    except ImportError:
        logger.error("orze.skills.loader not available — upgrade orze")
        return ""

    role_cfg = {
        "skills": [
            "@sop:professor_base",
            "@sop:professor_web_search",
            "@sop:professor_cross_domain_query",
            "@sop:professor_idea_review",
            "@sop:professor_diversity_enforcement",
            "@sop:professor_gap_closure",
            "@sop:professor_strategy_review",
            "@sop:professor_regression_detection",
            "@sop:professor_steering",
        ],
    }
    prompt_text = compose_skills(role_cfg, results_dir.parent,
                                 template_vars=None)
    if not prompt_text:
        logger.error("No professor SOPs composed")
        return ""

    # Interpolate template vars
    status = {}
    status_path = results_dir / "status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    prompt_text = prompt_text.replace("{results_dir}", str(results_dir))
    prompt_text = prompt_text.replace("{ideas_file}", "ideas.md")
    prompt_text = prompt_text.replace("{cycle}", str(status.get("iteration", "?")))
    prompt_text = prompt_text.replace("{completed}", str(status.get("completed", "?")))
    prompt_text = prompt_text.replace("{queued}", str(status.get("queue_depth", "?")))
    prompt_text = prompt_text.replace("{gpu_count}", str(len(status.get("free_gpus", []))))

    # Append gatekeeper context
    context = _build_gatekeeper_context(results_dir, ideas)
    return prompt_text + "\n\n---\n\n" + context


def _detect_backend() -> tuple:
    """Auto-detect which LLM backend is available. Returns (backend, api_key)."""
    for env_var, backend in [
        ("GEMINI_API_KEY", "gemini"),
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("OPENAI_API_KEY", "openai"),
    ]:
        key = os.environ.get(env_var, "")
        if key:
            return backend, key
    return "", ""


def _parse_professor_decisions(response: str) -> list:
    """Parse Professor LLM response into decision dicts."""
    # Try JSON array first
    try:
        # Strip markdown fences if present
        cleaned = response.strip()
        cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
        decisions = json.loads(cleaned)
        if isinstance(decisions, list):
            return [d for d in decisions
                    if isinstance(d, dict) and "idea_id" in d and "decision" in d]
    except (json.JSONDecodeError, ValueError):
        pass

    # Fall back to JSONL (one decision per line)
    decisions = []
    for line in response.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        try:
            d = json.loads(line)
            if isinstance(d, dict) and "idea_id" in d and "decision" in d:
                decisions.append(d)
        except (json.JSONDecodeError, ValueError):
            continue
    return decisions


def _apply_professor_decisions(results_dir: Path, decisions: list):
    """Apply Professor decisions: log all, skip rejected, note priorities."""
    lake_path = _find_lake_path(results_dir)

    # Append all decisions to JSONL log
    decisions_file = results_dir / "_professor_decisions.jsonl"
    with open(decisions_file, "a", encoding="utf-8") as f:
        for d in decisions:
            d["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            f.write(json.dumps(d) + "\n")

    conn = None
    if lake_path:
        try:
            conn = sqlite3.connect(str(lake_path), timeout=5)
        except Exception:
            conn = None

    skipped = 0
    prioritized = 0
    for d in decisions:
        idea_id = d["idea_id"]
        decision = d.get("decision", "").upper()
        reason = d.get("reason", "")

        if decision == "SKIP":
            # Mark as skipped in lake
            if conn:
                try:
                    conn.execute(
                        "UPDATE ideas SET status = 'skipped' WHERE idea_id = ?",
                        (idea_id,))
                except Exception:
                    pass
            # Write metrics.json
            idea_dir = results_dir / idea_id
            idea_dir.mkdir(exist_ok=True)
            (idea_dir / "metrics.json").write_text(json.dumps({
                "status": "SKIPPED",
                "skip_reason": f"professor: {reason}",
                "skipped_by": "professor",
                "confidence": d.get("confidence", 0),
                "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, indent=2), encoding="utf-8")
            skipped += 1
            logger.info("Professor SKIP %s: %s", idea_id, reason)

        elif decision == "PRIORITIZE":
            if conn:
                try:
                    conn.execute(
                        "UPDATE ideas SET priority = 'critical' WHERE idea_id = ?",
                        (idea_id,))
                except Exception:
                    pass
            prioritized += 1
            logger.info("Professor PRIORITIZE %s: %s", idea_id, reason)

    if conn:
        try:
            conn.commit()
            conn.close()
        except Exception:
            pass

    logger.info("Professor reviewed %d ideas: %d skipped, %d prioritized, %d approved",
                len(decisions), skipped, prioritized,
                len(decisions) - skipped - prioritized)


# ---------------------------------------------------------------------------
# Tier 2 Guards & Actions
# ---------------------------------------------------------------------------

@guard("has_unreviewed_ideas")
def has_unreviewed_ideas(ctx: Context) -> str | None:
    """True when queued ideas exist that the Professor hasn't reviewed yet."""
    ideas = _get_unreviewed_ideas(ctx.results_dir, limit=20)
    if ideas:
        return f"{len(ideas)} unreviewed ideas in queue"
    return None


@guard("has_llm_api_key")
def has_llm_api_key(ctx: Context) -> str | None:
    """True when at least one LLM API key is available."""
    backend, key = _detect_backend()
    if backend:
        return f"LLM available: {backend}"
    return None


@action("run_professor_review")
def run_professor_review(ctx: Context):
    """Tier 2: Call LLM Professor to review unreviewed queued ideas."""
    ideas = _get_unreviewed_ideas(ctx.results_dir, limit=20)
    if not ideas:
        logger.info("Professor: no ideas to review")
        return

    backend, api_key = _detect_backend()
    if not backend:
        logger.warning("Professor: no LLM API key available, skipping Tier 2")
        return

    prompt = _load_professor_prompt(ctx.results_dir, ideas)
    if not prompt:
        logger.error("Professor: failed to build prompt")
        return

    logger.info("Professor: reviewing %d ideas via %s", len(ideas), backend)

    # Import and call the LLM — reuse research_llm backends
    try:
        from orze_pro.agents.research_llm import call_llm
        response = call_llm(prompt, backend, api_key=api_key)
    except Exception as e:
        logger.error("Professor LLM call failed: %s", e)
        return

    if not response:
        logger.warning("Professor: LLM returned empty response")
        return

    decisions = _parse_professor_decisions(response)
    if not decisions:
        logger.warning("Professor: could not parse decisions from response (%d chars)",
                       len(response))
        return

    _apply_professor_decisions(ctx.results_dir, decisions)
    ctx.vars["last_review_count"] = len(decisions)
    ctx.vars["total_reviewed"] = ctx.vars.get("total_reviewed", 0) + len(decisions)
