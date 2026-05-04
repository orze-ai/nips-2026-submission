"""Agent role lifecycle for Orze.

CALLING SPEC:
    build_claude_cmd(role_cfg, template_vars) -> list[str] | None
    build_research_cmd(role_cfg, template_vars, cfg) -> list[str]
    run_role_step(role_name, role_cfg, ctx) -> None
    run_all_roles(ctx) -> None
    run_role_once(role_name, ctx) -> None
    sweep_stray(ctx, role_name, cycle_num, cycle_start_ts) -> list[str]

    ctx is a RoleContext dataclass containing:
        cfg, results_dir, gpu_ids, active_roles, role_states,
        failure_counts, fix_counts, iteration
"""

from __future__ import annotations

from orze_pro._gate import require_license; require_license()
import fnmatch
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from orze.engine.process import RoleProcess, _new_process_group
from orze.engine.launcher import _format_args
from orze.engine.scheduler import get_unclaimed, _count_statuses
from orze.engine.failure import get_skipped_ideas
from orze.engine.roles import (
    check_active_roles,
    OUTCOME_OK,
    OUTCOME_TIMEOUT,
    OUTCOME_ERROR,
    OUTCOME_SOFT_FAILURE,
    OUTCOME_RATE_LIMITED,
)
from orze.core.fs import _fs_lock, _fs_unlock
from orze.core.ideas import parse_ideas
from orze.core.config import orze_path
from orze.reporting.notifications import notify

logger = logging.getLogger(__name__)

# Hard ceiling on cooldown_override. The exponential backoff
# ``base_cooldown * 2 ** (err - 4)`` previously had no upper bound; in
# the 2026-04 silent campaign professor.consecutive_errors hit ~340 and
# cooldown_override reached 5.1×10⁹⁹s — float overflow territory. Cap
# at 24h: by then either an operator has noticed or the role is dead.
MAX_COOLDOWN_OVERRIDE_S = 86400

# Repeated-stub heuristic threshold. When the last
# ``_REPEATED_STUB_WINDOW`` cycle log files for a role are byte-identical
# AND each is at most ``_REPEATED_STUB_MAX_BYTES`` bytes, treat it as a
# structural failure (e.g. ``Not logged in · Please run /login``) and
# emit a ``role_circuit_breaker`` notification regardless of whether the
# subprocess returned a quota signal.
_REPEATED_STUB_WINDOW = 5
_REPEATED_STUB_MAX_BYTES = 200


def _clamp_cooldown_override(role_state: dict, role_name: str) -> None:
    """Clamp ``cooldown_override`` to ``MAX_COOLDOWN_OVERRIDE_S`` in place.

    Logs once-per-role at WARNING when a historical bad value is clamped
    on state load (idempotent: re-clamping a value already at the cap is
    silent). Tolerant of corrupt non-numeric values — those are dropped.
    """
    co = role_state.get("cooldown_override")
    if co is None:
        return
    try:
        co_f = float(co)
    except (TypeError, ValueError):
        role_state.pop("cooldown_override", None)
        logger.warning(
            "role '%s': dropping non-numeric cooldown_override=%r from state",
            role_name, co,
        )
        return
    if co_f > MAX_COOLDOWN_OVERRIDE_S:
        role_state["cooldown_override"] = MAX_COOLDOWN_OVERRIDE_S
        if not role_state.get("_cooldown_clamped_warned"):
            logger.warning(
                "role '%s': clamped cooldown_override %.3g s -> %d s "
                "(MAX_COOLDOWN_OVERRIDE_S). Historical exponential-backoff "
                "overflow scar — see 2026-04 post-mortem.",
                role_name, co_f, MAX_COOLDOWN_OVERRIDE_S,
            )
            role_state["_cooldown_clamped_warned"] = True


def clamp_all_cooldowns(role_states: dict) -> None:
    """Walk a role_states dict and clamp every cooldown_override.

    Called by the orchestrator on state-load to defang historical bad
    values (e.g. 5.1×10⁹⁹ from pre-cap orze-pro builds).
    """
    if not isinstance(role_states, dict):
        return
    for rname, rs in role_states.items():
        if isinstance(rs, dict):
            _clamp_cooldown_override(rs, rname)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class RoleContext:
    cfg: dict
    results_dir: Path
    gpu_ids: list
    active_roles: dict    # mutated in-place
    role_states: dict     # mutated in-place
    failure_counts: dict  # read-only
    fix_counts: dict      # read-only
    iteration: int


# ---------------------------------------------------------------------------
# Stray file sweeper
# ---------------------------------------------------------------------------

SWEEP_PATTERNS = ("*.py", "*.sh", "*.json", "*.log", "*.npz", "*.pkl")
SWEEP_ALLOWLIST_DEFAULTS = {
    "README.md", "GOAL.md", "orze.yaml", "ideas.md", "pyproject.toml", "setup.py", "setup.cfg",
    "Makefile", "LICENSE", "CHANGELOG.md", "nohup.out", "train.py", "train_decoder.py",
    # Pre-scripts are infrastructure files — never sweep them
    "ensure_idea_dir.py",
    # orze runtime sidecars at project root — framework writes these itself.
    # Sweeping them breaks the feature that owns them (e.g. research dedup
    # loses its ingest cursor every cycle and re-ingests the whole ideas.md).
    "ideas.md.ingest_state.json",
}
# Built-in glob protections for orze-generated sidecars at project root.
# Kept separate from user_allow globs so framework-owned files survive even
# when the project hasn't customized sweep_allowlist.
SWEEP_ALLOWLIST_GLOBS_DEFAULT = (
    "ideas.md.safe*",      # role_runner ideas.md backups
    "ideas.md.corrupt.*",  # corruption-detector quarantined copies
)
SWEEP_ALLOWED_DIRS = {
    "src", "tests", "configs", "rules", "scripts", ".orze", "orze_results",
    ".git", "build", "dist", "__pycache__", ".venv", "venv"
}


def sweep_stray(ctx, role_name: str, cycle_num: int, cycle_start_ts: float) -> list[str]:
    """Move role-generated files at project root into quarantine. Returns list of moved paths."""
    cfg = ctx.cfg
    if not cfg.get("sweep_stray", True):
        return []
    project_root = Path(cfg["_project_root"])
    user_allow = cfg.get("sweep_allowlist", []) or []
    # Also protect pre_script and eval_script configured in orze.yaml
    pre_script = cfg.get("pre_script", "")
    eval_script = cfg.get("eval_script", "")
    extra_protected = set()
    for s in (pre_script, eval_script):
        if s:
            extra_protected.add(Path(s).name)
    allow_files = set(SWEEP_ALLOWLIST_DEFAULTS) | extra_protected | {p for p in user_allow if not any(c in p for c in "*?[")}
    allow_globs = list(SWEEP_ALLOWLIST_GLOBS_DEFAULT) + [p for p in user_allow if any(c in p for c in "*?[")]
    moved = []
    analysis_exts = {".py", ".sh"}
    try:
        for entry in project_root.iterdir():
            if entry.is_dir():
                continue
            if entry.name in allow_files:
                continue
            if any(fnmatch.fnmatch(entry.name, p) for p in allow_globs):
                continue
            if entry.suffix.lower() == ".md":
                continue  # allow root markdown
            if entry.name.startswith("."):
                continue
            try:
                if entry.stat().st_mtime <= cycle_start_ts:
                    continue
            except OSError:
                continue
            if not any(fnmatch.fnmatch(entry.name, p) for p in SWEEP_PATTERNS):
                continue
            if entry.suffix.lower() in analysis_exts:
                tgt_dir = orze_path(cfg, "methods") / role_name / f"cycle_{cycle_num:03d}"
            else:
                tgt_dir = orze_path(cfg, "stray") / role_name / f"cycle_{cycle_num:03d}"
            tgt_dir.mkdir(parents=True, exist_ok=True)
            try:
                entry.rename(tgt_dir / entry.name)
                moved.append(str(entry))
            except OSError:
                shutil.move(str(entry), str(tgt_dir / entry.name))
                moved.append(str(entry))
    except Exception as e:
        logger.warning("sweep_stray failed: %s", e)
        return moved
    if moved:
        sweeps_log = Path(cfg["_orze_dir"]) / "state" / "sweeps.jsonl"
        sweeps_log.parent.mkdir(parents=True, exist_ok=True)
        with open(sweeps_log, "a") as fp:
            fp.write(json.dumps({"ts": time.time(), "role": role_name, "cycle": cycle_num, "moved": moved}) + "\n")
        # notify root_pollution
        try:
            from orze.engine.intervention_detect import should_notify
            state_file = Path(cfg["_orze_dir"]) / "state" / "interventions.json"
            key = f"root_pollution:{role_name}:-"
            if should_notify(state_file, key):
                notify("needs_intervention", {
                    "role": role_name,
                    "idea_id": None,
                    "reason": "root_pollution",
                    "evidence": f"role wrote {len(moved)} file(s) to project root",
                    "log_tail": "\n".join(moved[:20]),
                    "host": socket.gethostname(),
                    "pid": os.getpid(),
                }, cfg)
        except Exception:
            pass
    return moved


# ---------------------------------------------------------------------------
# Execution receipts (forward-compatible with orze versions that lack the
# skills.receipts module — silently no-op in that case)
# ---------------------------------------------------------------------------

def _receipt_discover_outputs(role_cfg: dict, project_root: Path,
                              template_vars: Optional[dict] = None) -> dict:
    """Return {skill_id: [produced_paths]} for the skills this role uses.

    Handles both SOP tiers:
    - '@sop:<name>' refs resolve via orze_pro.skills.bundled (static tier)
    - './path.md' or 'path.md' refs resolve to project-local files (dynamic tier)
    - Generic '@<name>' orze built-ins have no SOP metadata and are skipped.

    Template variables like '{ideas_file}' and '{results_dir}' that appear
    in frontmatter ``produces`` entries are substituted against the same
    template_vars dict that the composed prompt uses. Without substitution,
    ``snapshot_mtimes`` would stat literal ``{ideas_file}`` paths that
    never exist, and no skill would ever record evidence of execution.

    Returns {} on any failure (fail-open — receipts must never kill a role).
    """
    try:
        from orze.skills.loader import parse_frontmatter
    except ImportError:
        return {}
    skills_list = role_cfg.get("skills") or []
    if not isinstance(skills_list, list):
        return {}

    tvars = template_vars or {}

    def _subst(s: str) -> str:
        for k, v in tvars.items():
            s = s.replace(f"{{{k}}}", str(v))
        return s

    def _extract(meta: dict) -> tuple:
        sid = meta.get("id")
        produces = meta.get("produces") or []
        if isinstance(produces, str):
            produces = [produces]
        return sid, [_subst(str(p)) for p in produces]

    out: dict = {}
    for ref in skills_list:
        ref = str(ref).strip()

        if ref.startswith("@sop:"):
            name = ref[len("@sop:"):]
            try:
                from orze_pro.skills.bundled import load_bundled_skill
                text, _path = load_bundled_skill(name)
                meta, _ = parse_frontmatter(text)
            except (ImportError, FileNotFoundError, OSError):
                continue
            sid, produces = _extract(meta)
            if sid and produces:
                out[str(sid)] = produces
            continue

        if ref.startswith("@"):
            continue  # generic orze built-in, no produces metadata

        path = Path(ref)
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            continue
        try:
            meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
        except OSError:
            continue
        sid, produces = _extract(meta)
        if sid and produces:
            out[str(sid)] = produces
    return out


def _receipt_snapshot_before_launch(role_name: str, role_cfg: dict,
                                    ctx: "RoleContext",
                                    template_vars: Optional[dict] = None) -> None:
    """Capture declared outputs and mtime snapshot into role_state.

    ``template_vars`` are used to substitute placeholders like
    ``{ideas_file}`` that appear in SOP frontmatter ``produces`` entries —
    the same tokens the composed prompt resolves. Without this, declared
    outputs would be literal ``{ideas_file}`` paths that never exist and
    ``skills_evidenced`` would always be empty for affected skills
    (notably ``sop-res-base`` and ``sop-ce-base``).

    Silent no-op if orze is too old to have the receipts module.
    """
    try:
        from orze.skills.receipts import snapshot_mtimes
    except ImportError:
        return
    try:
        project_root = ctx.results_dir.parent
        declared = _receipt_discover_outputs(role_cfg, project_root,
                                             template_vars=template_vars)
        state = ctx.role_states.setdefault(role_name, {})
        state["_receipt_declared"] = declared
        state["_receipt_mtime_snap"] = snapshot_mtimes(declared, project_root)
        state["_receipt_started_at"] = time.time()
    except Exception as e:  # receipts must never kill a role
        logger.debug("Receipt snapshot skipped for %s: %s", role_name, e)


def _receipt_write_after_completion(role_name: str,
                                    ctx: "RoleContext") -> None:
    """Compute evidenced skills and write the receipt JSON. No-op on any error."""
    try:
        from orze.skills.receipts import (
            Receipt, compute_evidenced_skills, write_receipt,
        )
    except ImportError:
        return
    try:
        state = ctx.role_states.get(role_name) or {}
        declared = state.get("_receipt_declared")
        if declared is None:
            return  # no snapshot captured (legacy role or import failed earlier)
        snap = state.get("_receipt_mtime_snap") or {}
        project_root = ctx.results_dir.parent
        evidenced = compute_evidenced_skills(declared, snap, project_root)
        cycle = int(state.get("cycles", 0))
        receipt = Receipt(
            role=role_name,
            cycle=cycle,
            started_at=float(state.get("_receipt_started_at", 0.0)),
            ended_at=time.time(),
            skills_declared=list(declared.keys()),
            skills_evidenced=evidenced,
            outputs=declared,
        )
        rpath = orze_path(ctx.cfg, "receipts", f"{role_name}_cycle{cycle:03d}.json")
        write_receipt(receipt, rpath)
        # Clear snapshot state so a re-run starts clean
        for k in ("_receipt_declared", "_receipt_mtime_snap",
                  "_receipt_started_at"):
            state.pop(k, None)
    except Exception as e:
        logger.debug("Receipt write skipped for %s: %s", role_name, e)


# ---------------------------------------------------------------------------
# SOP-derived role behavior
# ---------------------------------------------------------------------------

def _default_role_timeout(skills_count: int) -> int:
    """Timeout budget derived from SOP count: 10min base + 4min per skill.

    Replaces per-role hardcoded timeouts. A role composing N SOPs gets
    600 + 240*N seconds — enough for a typical 4-minute Claude CLI pass
    through each skill plus setup/teardown. Project-level ``orze.yaml``
    entries with an explicit ``timeout:`` still win (inject-time only).
    """
    return 600 + 240 * max(0, int(skills_count))


# Round-2 B1/B2: auto-derive timeout and stall_minutes from skill count
# at *role-step* time when the user hasn't set them explicitly. The
# existing `_default_role_timeout` only fires at inject-time for
# auto-injected roles, so user-defined claude roles still inherited the
# 600s default, which was too short for data_analyst (6 skills) and
# professor (10 skills) and caused them to be killed mid-cycle.
#
# Spec (round-2 plan):
#   timeout       = max(300, 60 * len(skills))     for mode: claude
#   stall_minutes = max(5,   2 * len(skills))      for mode: claude
#
# `mode: script` keeps stall_minutes=5 by default. Explicit values in
# orze.yaml always win; auto-derivation logs at INFO once per role.
_AUTO_DERIVED_LOGGED: set = set()


def _resolve_role_timeout(role_name: str, role_cfg: dict) -> int:
    explicit = role_cfg.get("timeout")
    if explicit is not None:
        return int(explicit)
    if role_cfg.get("mode") == "claude":
        n = len(role_cfg.get("skills") or [])
        derived = max(300, 60 * n)
        key = f"timeout:{role_name}"
        if key not in _AUTO_DERIVED_LOGGED:
            _AUTO_DERIVED_LOGGED.add(key)
            logger.info(
                "Role %r: auto-timeout=%ds (%d skills) — set timeout: in "
                "orze.yaml to override.", role_name, derived, n)
        return derived
    return 600


def _resolve_role_stall_minutes(role_name: str, role_cfg: dict,
                                global_default: int) -> int:
    explicit = role_cfg.get("stall_minutes")
    if explicit is not None:
        return int(explicit)
    if role_cfg.get("mode") == "claude":
        n = len(role_cfg.get("skills") or [])
        derived = max(5, 2 * n)
        key = f"stall:{role_name}"
        if key not in _AUTO_DERIVED_LOGGED:
            _AUTO_DERIVED_LOGGED.add(key)
            logger.info(
                "Role %r: auto-stall_minutes=%dm (%d skills) — set "
                "stall_minutes: under the role to override.",
                role_name, derived, n)
        return derived
    return int(global_default)


def _role_writes_ideas(ctx: "RoleContext", role_name: str,
                       role_cfg: Optional[dict] = None) -> bool:
    """True iff any SOP produces path equal to ``ideas_file``.

    Derived from SOP ``produces`` frontmatter — no role-name allowlist.
    Template vars like ``{ideas_file}`` are substituted before comparison
    so a SOP declaring ``produces: ['{ideas_file}']`` matches.
    """
    if role_cfg is None:
        role_cfg = (ctx.cfg.get("roles") or {}).get(role_name) or {}
    ideas_file = str(ctx.cfg.get("ideas_file", "ideas.md"))
    try:
        project_root = ctx.results_dir.parent
        tvars = {
            "ideas_file": ideas_file,
            "results_dir": str(ctx.results_dir),
            "role_name": role_name,
        }
        declared = _receipt_discover_outputs(role_cfg, project_root,
                                             template_vars=tvars)
    except Exception:
        return True  # fail-open — behave like legacy research role
    for outputs in declared.values():
        for out in outputs:
            if out == ideas_file or Path(out).name == Path(ideas_file).name:
                return True
    return False


# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def build_claude_cmd(
    role_cfg: dict,
    template_vars: dict,
    cfg: Optional[dict] = None,
) -> Optional[List[str]]:
    """Build a Claude CLI command for mode: claude.

    Requires role_cfg['skills']. Every Claude role composes its prompt
    from SOP skills — bundled static (@sop:<name>) and/or project-local
    dynamic (./path.skill.md).
    """
    from orze.skills.loader import compose_skills
    project_root = Path(template_vars.get("results_dir", ".")).parent
    prompt = compose_skills(role_cfg, project_root, template_vars=template_vars)
    if not prompt:
        logger.warning("%s produced no composed prompt (skills: %r)",
                       template_vars.get("role_name", "?"),
                       role_cfg.get("skills"))
        return None

    # Tier 2 SOP: inject structured YAML writing instructions for professor
    if template_vars.get("role_name") == "professor":
        try:
            from orze_pro.engine.sop_tier2 import build_sop_instructions
            results_dir = Path(template_vars.get("results_dir", "orze_results"))
            sop_section = build_sop_instructions(results_dir)
            if sop_section:
                prompt += sop_section
        except Exception:
            pass

    # `orze-claude` is the subscription-first / API-key-fallback shim
    # shipped with base orze (see orze/shims/claude.py). Defaulting to
    # the shim means pipelines on Claude Code subscription hosts no
    # longer fail every `mode: claude` cycle once the subscription cap
    # is hit — the shim transparently retries with ANTHROPIC_API_KEY
    # when it detects a quota-exhaustion message. A user who explicitly
    # sets `claude_bin: claude` in a role opts out of the shim.
    claude_bin = role_cfg.get("claude_bin") or "orze-claude"
    # Resolve to absolute path. The default "orze-claude" is a console
    # script installed into the venv that runs orze; subprocesses orze
    # spawns inherit the parent's PATH which usually does NOT include
    # venv/bin (orze is commonly started via `./venv/bin/python3 -m
    # orze.cli` rather than with the venv activated). Without this
    # resolution, every `mode: claude` role cycle would fail with
    # "FileNotFoundError: orze-claude".
    #
    # Don't use Path(sys.executable).resolve().parent — when
    # venv/bin/python is a symlink to /usr/bin/python3, resolve()
    # follows it and ``parent`` becomes /usr/bin. sys.prefix is the
    # venv root (set by venv activation, correct both for activated
    # and `./venv/bin/python` invocations).
    if claude_bin and not os.path.isabs(claude_bin):
        resolved = shutil.which(claude_bin)
        if not resolved:
            cand = Path(sys.prefix) / "bin" / claude_bin
            if cand.exists():
                resolved = str(cand)
        if resolved:
            claude_bin = resolved
    cmd = [claude_bin, "-p", prompt]

    # --model (e.g., sonnet, opus, haiku)
    model = role_cfg.get("model")
    if model:
        cmd.extend(["--model", model])

    # --mcp-config: write MCP server config to a temp file if configured.
    # Role config format:
    #   mcp_servers:
    #     <name>:
    #       type: http
    #       url: https://...
    mcp_servers = role_cfg.get("mcp_servers") or {}
    if mcp_servers and cfg is not None:
        import json as _json
        role_name = template_vars.get("role_name", "role")
        mcp_config_path = orze_path(cfg, "mcp", f"_{role_name}_mcp_config.json")
        mcp_config_path.write_text(
            _json.dumps({"mcpServers": mcp_servers}, indent=2),
            encoding="utf-8",
        )
        cmd.extend(["--mcp-config", str(mcp_config_path)])

    # --allowedTools (default: local tools + web for professors)
    default_tools = "Read,Write,Edit,Glob,Grep,Bash,WebSearch,WebFetch"
    allowed_tools = role_cfg.get("allowed_tools") or default_tools
    # If MCP servers are configured, allow all their tools automatically
    # via the mcp__<server> wildcard pattern.
    if mcp_servers:
        mcp_allowed = ",".join(f"mcp__{name}" for name in mcp_servers.keys())
        allowed_tools = f"{allowed_tools},{mcp_allowed}"
    cmd.extend(["--allowedTools", str(allowed_tools)])

    # --output-format
    output_format = role_cfg.get("output_format") or "text"
    cmd.extend(["--output-format", str(output_format)])

    # --dangerously-skip-permissions (default on for auto-injected roles).
    # These roles run headless under the orze daemon — permission prompts
    # in `claude -p` mode have no stdin to answer, so the per-command
    # permission layer stalls the role instead of protecting anything.
    # The real sandbox is --allowedTools (above) + the role's orze.yaml
    # mcp_servers config. Opt out per-role with:
    #     roles:
    #       professor:
    #         dangerously_skip_permissions: false
    if role_cfg.get("dangerously_skip_permissions", True):
        cmd.append("--dangerously-skip-permissions")

    # Any extra CLI args
    cmd.extend(_format_args(role_cfg.get("claude_args") or [],
                            template_vars))

    return cmd

def build_research_cmd(
    role_cfg: dict,
    template_vars: dict,
    cfg: dict,
) -> List[str]:
    """Build command for mode: research (built-in LLM research agent).

    Minimal config:
        research_gemini:
          mode: research
          backend: gemini       # gemini, openai, anthropic, ollama, custom
          model: gemini-3.1-pro-preview  # optional
          endpoint: http://...  # optional, for ollama/custom
          skills:                       # composed into the agent's prompt
            - "@sop:research_base"
            - ./RESEARCH_RULES.md       # project-specific constraints
          env:
            GEMINI_API_KEY: "..."
    """
    python = cfg.get("python", sys.executable)
    # research_agent.py lives in the agents directory
    agent_script = Path(__file__).parent.parent / "agents" / "research.py"

    cmd = [python, str(agent_script)]
    cmd.extend(["-c", str(cfg.get("_config_path", "orze.yaml"))])
    cmd.extend(["--backend", role_cfg["backend"]])
    cmd.extend(["--cycle", str(template_vars["cycle"])])
    cmd.extend(["--ideas-md", str(template_vars["ideas_file"])])
    cmd.extend(["--results-dir", str(template_vars["results_dir"])])

    if role_cfg.get("model"):
        cmd.extend(["--model", str(role_cfg["model"])])
    if role_cfg.get("endpoint"):
        cmd.extend(["--endpoint", str(role_cfg["endpoint"])])
    if role_cfg.get("num_ideas"):
        cmd.extend(["--num-ideas", str(role_cfg["num_ideas"])])
    from orze.skills.loader import compose_skills
    project_root = Path(template_vars.get("results_dir", ".")).parent
    composed = compose_skills(role_cfg, project_root, template_vars=None)
    if composed:
        tmp = orze_path(cfg, "tmp", f"skill_composed_{template_vars.get('role_name', 'research')}.md")
        tmp.write_text(composed, encoding="utf-8")
        cmd.extend(["--rules-file", str(tmp)])

    # Pass lake DB path - now from orze_dir
    lake_path = Path(cfg.get("idea_lake_db") or (Path(cfg["_orze_dir"]) / "idea_lake.db"))
    if lake_path.exists():
        cmd.extend(["--lake-db", str(lake_path)])

    # Pass retrospection file if it exists - now in knowledge/
    retro_file = orze_path(cfg, "knowledge", "retrospection.md")
    if retro_file.exists():
        cmd.extend(["--retrospection-file", str(retro_file)])

    # Auto-include the entire knowledge/ directory so any role (e.g.
    # data_analyst) that drops a *.md finding there flows into research
    # context with no per-project wiring. The research agent dedups
    # against the retrospection_file path so retrospection isn't loaded
    # twice.
    knowledge_dir = orze_path(cfg, "knowledge")
    if knowledge_dir.is_dir():
        cmd.extend(["--knowledge-dir", str(knowledge_dir)])

    return cmd

# ---------------------------------------------------------------------------
# Step / launch
# ---------------------------------------------------------------------------

def run_role_step(role_name: str, role_cfg: dict, ctx: RoleContext) -> None:
    """Launch agent role if not running and cooldown elapsed (non-blocking).

    Supports three modes:
      - mode: script  -- run a Python script
      - mode: claude  -- run Claude CLI with a rules/prompt file
      - mode: research -- run built-in LLM research agent
    """
    # Skip if already running
    if role_name in ctx.active_roles:
        return

    # Triggered roles only run when a trigger file exists
    explicitly_triggered = False
    if role_cfg.get("triggered_by"):
        trigger_file = orze_path(ctx.cfg, "triggers", f"_trigger_{role_name}")
        if not trigger_file.exists():
            # Deadlock breaker: if the triggering role (e.g. fsm, professor)
            # is stalled, auto-trigger this role so downstream work can
            # proceed despite the upstream being stuck. Applies to any role
            # with `triggered_by:` by default (these are the deadlock-prone
            # ones); can be opted out via `auto_trigger_on_stall: false`
            # per role in orze.yaml. Roles without `triggered_by:` cannot
            # be auto-triggered here (no upstream to watch).
            trigger_source = role_cfg["triggered_by"]
            source_state = ctx.role_states.get(trigger_source, {})
            stall_count = source_state.get("consecutive_zero_output", 0)
            stall_threshold = role_cfg.get("auto_trigger_stall_threshold", 10)
            auto_trigger = role_cfg.get("auto_trigger_on_stall", True)
            if stall_count >= stall_threshold and auto_trigger:
                logger.warning("Auto-triggering %s: %s stalled for %d cycles",
                               role_name, trigger_source, stall_count)
                explicitly_triggered = True
            else:
                return
        else:
            # Consume the trigger (role will be launched below)
            try:
                trigger_file.unlink(missing_ok=True)
                logger.info("Role '%s' triggered by %s", role_name,
                            role_cfg["triggered_by"])
                explicitly_triggered = True
            except OSError:
                pass

    # Thinker: trigger-condition based activation
    if role_cfg.get("trigger_conditions") and not role_cfg.get("triggered_by"):
        trigger_reason = _check_thinker_trigger(role_name, role_cfg, ctx)
        if not trigger_reason:
            return  # no trigger condition met
        # Check min cooldown (manual triggers bypass)
        role_state = ctx.role_states.setdefault(
            role_name, {"cycles": 0, "last_run_time": 0.0})
        is_manual = trigger_reason.startswith("manual:")
        min_cd = role_cfg.get("min_cooldown", 1800)
        if not is_manual and time.time() - role_state["last_run_time"] < min_cd:
            return
        logger.info("Thinker triggered: %s", trigger_reason)
        role_state["_trigger_reason"] = trigger_reason
        research_cycles = ctx.role_states.get("research", {}).get("cycles", 0)
        role_state["_last_trigger_cycle"] = research_cycles
        # Reset plateau baseline so it doesn't re-trigger immediately
        try:
            report_path = ctx.results_dir / "report.md"
            if report_path.exists():
                text = report_path.read_text(encoding="utf-8")
                completed = len(re.findall(r"^\|\s*\d+\s*\|", text,
                                           re.MULTILINE))
                role_state["_best_metric_at"] = completed
        except Exception:
            pass

    mode = role_cfg.get("mode", "script")
    if mode == "script" and not role_cfg.get("script"):
        return
    if mode == "claude" and not role_cfg.get("skills"):
        return
    if mode == "research" and not role_cfg.get("backend"):
        return

    # Skip research roles if retrospection paused them
    if role_name == "research" or role_cfg.get("pausable", False):
        from orze.engine.retrospection import is_research_paused
        if is_research_paused(ctx.results_dir):
            if not ctx.role_states.get(role_name, {}).get("_pause_logged"):
                logger.info("Role '%s' paused by retrospection (.pause_research sentinel)",
                            role_name)
                ctx.role_states.setdefault(role_name, {})["_pause_logged"] = True
            return

    # Per-role cooldown (with adaptive producer-consumer matching)
    role_state = ctx.role_states.setdefault(
        role_name, {"cycles": 0, "last_run_time": 0.0})
    # Defang any historical bad cooldown_override loaded from
    # ``.orze_state_<host>.json`` written by a pre-cap orze-pro build.
    # See MAX_COOLDOWN_OVERRIDE_S.
    _clamp_cooldown_override(role_state, role_name)
    cooldown = role_state.get("cooldown_override",
                              role_cfg.get("cooldown", 300))
    elapsed = time.time() - role_state["last_run_time"]

    # Hot-reload: if GOAL.md changed since last run, skip cooldown
    goal_changed = False
    goal_path = Path(ctx.cfg.get("goal_file", "GOAL.md"))
    if goal_path.exists():
        try:
            mtime = goal_path.stat().st_mtime
            prev = role_state.get("_goal_mtime", 0.0)
            if mtime > prev:
                if prev > 0:  # don't trigger on first read
                    goal_changed = True
                    logger.info("GOAL.md changed — triggering %s immediately",
                                role_name)
                role_state["_goal_mtime"] = mtime
        except OSError:
            pass

    # Professor uses fixed cycle_interval — never scaled by convergence
    # or queue depth. It must run regularly to steer research direction.
    if role_name == "professor":
        prof_interval = role_cfg.get("cycle_interval", 600)
        # Check for _trigger_professor file (written by orze result add)
        trigger_file = orze_path(ctx.cfg, "triggers", "_trigger_professor")
        prof_triggered = trigger_file.exists()
        if prof_triggered:
            try:
                trigger_reason = trigger_file.read_text(encoding="utf-8").strip()
                trigger_file.unlink(missing_ok=True)
                role_state["_trigger_reason"] = trigger_reason
                logger.info("Professor triggered: %s", trigger_reason[:100])
            except OSError:
                prof_triggered = False
        if elapsed < prof_interval and not goal_changed and not prof_triggered:
            return
        logger.info("Professor ready: elapsed=%.0fs, interval=%ds", elapsed, prof_interval)

    # Generic trigger-file consumption for non-`triggered_by` roles.
    # When another role drops `_trigger_<role_name>` in the triggers dir
    # to convey instructions (e.g. professor → engineer), surface its
    # contents as the trigger_reason so the SOP receives the actual task
    # rather than the default "scheduled" string. Without this, the
    # trigger file accumulates on disk and the role spins through
    # "no actionable trigger" cycles. Only applies to roles that don't
    # already declare `triggered_by` (those have their own consumption
    # path earlier in this function).
    if not role_cfg.get("triggered_by") and role_name != "professor":
        eng_trigger = orze_path(ctx.cfg, "triggers", f"_trigger_{role_name}")
        if eng_trigger.exists():
            try:
                trigger_reason = eng_trigger.read_text(encoding="utf-8").strip()
                eng_trigger.unlink(missing_ok=True)
                if trigger_reason:
                    role_state["_trigger_reason"] = trigger_reason
                    logger.info("%s triggered via _trigger_%s: %s",
                                role_name, role_name, trigger_reason[:100])
            except OSError:
                pass

    # Adaptive cooldown: if queue is nearly empty, skip cooldown to
    # keep GPUs fed. Only applies to the research role.
    queue_starving = False
    if role_name == "research" and elapsed >= 60:
        try:
            ideas = parse_ideas(ctx.cfg["ideas_file"])
            skipped = get_skipped_ideas(
                ctx.failure_counts,
                ctx.cfg.get("max_idea_failures", 0))
            n_unclaimed = len(get_unclaimed(
                ideas, ctx.results_dir, skipped,
                lake=getattr(ctx, 'lake', None)))
            n_gpus = len(ctx.gpu_ids)

            # --- Hard queue cap: skip research entirely ---
            max_queue = ctx.cfg.get("max_queue_size", 500)
            if n_unclaimed > max_queue:
                logger.info(
                    "Queue full (%d > %d) — skipping research",
                    n_unclaimed, max_queue)
                return

            # --- Adaptive cooldown: scale with queue depth ---
            # When queue is deep, slow down to save API costs.
            # When shallow, keep configured cooldown or trigger early.
            if n_unclaimed < n_gpus * 2:
                queue_starving = True
                logger.info(
                    "Queue low (%d unclaimed, %d GPUs) — "
                    "triggering research early", n_unclaimed, n_gpus)
            elif n_unclaimed > n_gpus * 8:
                # Queue is deep — double the cooldown
                cooldown = cooldown * 2
                logger.debug(
                    "Queue deep (%d unclaimed, %d GPUs) — "
                    "cooldown extended to %ds", n_unclaimed, n_gpus,
                    cooldown)

            # --- Convergence slowdown ---
            # If primary metric hasn't improved in N completed ideas,
            # multiply cooldown. Uses completed count as a stable
            # monotonic signal (not wall-clock time).
            patience = ctx.cfg.get("convergence_patience", 0)
            if patience > 0:
                best_val = role_state.get("_best_metric_val")
                best_at = role_state.get("_best_metric_at", 0)
                counts = _count_statuses(ideas, ctx.results_dir)
                n_completed = counts.get("COMPLETED", 0)

                # Read current best from completed rows
                primary = ctx.cfg["report"].get(
                    "primary_metric", "test_accuracy")
                sort_desc = ctx.cfg["report"].get(
                    "sort", "descending") == "descending"
                cur_best = None
                for d in ctx.results_dir.iterdir():
                    if not d.is_dir() or not d.name.startswith("idea-"):
                        continue
                    mp = d / "metrics.json"
                    if not mp.exists():
                        continue
                    try:
                        m = json.loads(
                            mp.read_text(encoding="utf-8"))
                        if m.get("status") != "COMPLETED":
                            continue
                        v = m.get(primary)
                        if v is None:
                            continue
                        if cur_best is None:
                            cur_best = v
                        elif sort_desc and v > cur_best:
                            cur_best = v
                        elif not sort_desc and v < cur_best:
                            cur_best = v
                    except Exception:
                        continue

                if cur_best is not None:
                    improved = False
                    if best_val is None:
                        improved = True
                    elif sort_desc and cur_best > best_val:
                        improved = True
                    elif not sort_desc and cur_best < best_val:
                        improved = True

                    if improved:
                        role_state["_best_metric_val"] = cur_best
                        role_state["_best_metric_at"] = n_completed
                    elif n_completed - best_at >= patience:
                        stale = n_completed - best_at
                        multiplier = 1 + (stale // patience)
                        cooldown = int(cooldown * multiplier)
                        logger.info(
                            "Convergence: no improvement in %d ideas "
                            "(best=%s at %d) — cooldown %ds",
                            stale, best_val, best_at, cooldown)
        except Exception:
            pass

    if (role_name != "professor" and elapsed < cooldown
            and not queue_starving and not goal_changed
            and not explicitly_triggered):
        # Log when an explicit trigger file existed but we're about to
        # skip — helps diagnose "why isn't role X firing?" in prod.
        # Only visible at DEBUG normally; promoted to INFO if the
        # role is in circuit-breaker backoff.
        cooldown_override = role_state.get("cooldown_override")
        if cooldown_override:
            logger.info(
                "Role '%s' in circuit-breaker backoff — cooldown %.0fs, "
                "elapsed %.0fs (need goal_changed or success to reset)",
                role_name, cooldown, elapsed)
        else:
            logger.debug(
                "Role '%s' cooldown: %.0fs elapsed < %.0fs",
                role_name, elapsed, cooldown)
        return

    # Round-2 B1: auto-derive timeout for mode: claude when not set
    # explicitly. See _resolve_role_timeout for the formula.
    timeout = _resolve_role_timeout(role_name, role_cfg)

    # Per-role cross-machine lock
    lock_dir = orze_path(ctx.cfg, "locks", role_name)
    if not _fs_lock(lock_dir, stale_seconds=timeout + 60):
        logger.debug("%s lock held by another host, skipping", role_name)
        return

    # Template variables (shared across all roles)
    ideas = parse_ideas(ctx.cfg["ideas_file"])
    counts = _count_statuses(ideas, ctx.results_dir)
    template_vars = {
        "ideas_file": ctx.cfg["ideas_file"],
        "results_dir": str(ctx.results_dir),
        "cycle": role_state["cycles"] + 1,
        "gpu_count": len(ctx.gpu_ids),
        "completed": counts.get("COMPLETED", 0),
        "queued": counts.get("QUEUED", 0),
        "role_name": role_name,
        "trigger_reason": role_state.get("_trigger_reason", "scheduled"),
        "timestamp": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
    }

    # If the role composes via skills:, inject the per-cycle trigger
    # evaluation context so skill-level gates (periodic_research_cycles,
    # on_plateau) see real values instead of defaults. The underscore
    # prefix signals 'transient, not user-authored'. Overwritten each
    # cycle; the next role run gets fresh numbers.
    if role_cfg.get("skills"):
        research_cycles = (ctx.role_states.get("research", {}) or {}
                           ).get("cycles", 0)
        plateau_patience = 0
        try:
            report_path = ctx.results_dir / "report.md"
            if report_path.exists():
                rtext = report_path.read_text(encoding="utf-8")
                n_completed = len(re.findall(
                    r"^\|\s*\d+\s*\|", rtext, re.MULTILINE))
                best_at = role_state.get("_best_metric_at", 0)
                plateau_patience = max(0, n_completed - best_at)
        except Exception:
            pass
        role_cfg["_trigger_context"] = {
            "research_cycles": int(research_cycles),
            "last_activation_cycle": int(
                role_state.get("_skill_last_activation_cycle", 0)),
            "plateau_patience": int(plateau_patience),
        }

    # Build command based on mode
    if mode == "claude":
        cmd = build_claude_cmd(role_cfg, template_vars, ctx.cfg)
        if not cmd:
            _fs_unlock(lock_dir)
            return
    elif mode == "research":
        cmd = build_research_cmd(role_cfg, template_vars, ctx.cfg)
    else:
        python = ctx.cfg.get("python", sys.executable)
        import shlex as _shlex
        script_tokens = _shlex.split(role_cfg["script"])
        cmd = [python, *script_tokens]
        cmd.extend(_format_args(role_cfg.get("args") or [], template_vars))

    # Environment
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)  # Allow nested Claude CLI sessions
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    for k, v in (ctx.cfg.get("train_extra_env") or {}).items():
        env[k] = str(v)
    for k, v in (role_cfg.get("env") or {}).items():
        env[k] = str(v)
    # Ensure orze and orze_pro are importable by built-in roles (both
    # `mode: research` and `mode: script` with an orze-shipped runner like
    # fsm/runner.py) even when cfg.python points to a project venv that
    # doesn't have them installed.
    #
    # Cover both package locations — orze-pro's own src (where this file
    # lives) makes `orze_pro` importable; orze's site-packages dir (where
    # `import orze` resolves at the orchestrator) makes `orze` importable
    # under the venv's python. Without the second entry, `fsm/runner.py`'s
    # `from orze.fsm.runner import main` fails every FSM cycle.
    if mode in ("research", "script"):
        extra_paths = [str(Path(__file__).parent.parent.parent)]
        try:
            import orze as _orze_mod  # type: ignore
            extra_paths.append(str(Path(_orze_mod.__file__).resolve().parent.parent))
        except ImportError:
            pass
        existing = env.get("PYTHONPATH", "")
        existing_parts = existing.split(os.pathsep) if existing else []
        prepend = [p for p in extra_paths if p and p not in existing_parts]
        if prepend:
            env["PYTHONPATH"] = os.pathsep.join(prepend + existing_parts) if existing else os.pathsep.join(prepend)

    # Inject ORZE_* environment variables for all roles
    cycle_num = role_state["cycles"] + 1
    role_scratch = orze_path(ctx.cfg, "tmp", role_name) / f"cycle_{cycle_num:03d}"
    role_scratch.mkdir(parents=True, exist_ok=True)
    env.update({
        "ORZE_DIR": ctx.cfg["_env_ORZE_DIR"],
        "ORZE_RESULTS_DIR": ctx.cfg["_env_ORZE_RESULTS_DIR"],
        "ORZE_IDEAS_FILE": ctx.cfg["_env_ORZE_IDEAS_FILE"],
        "ORZE_RULES_DIR": ctx.cfg["_env_ORZE_RULES_DIR"],
        "ORZE_METHODS_DIR": ctx.cfg["_env_ORZE_METHODS_DIR"],
        "ORZE_KNOWLEDGE_DIR": ctx.cfg["_env_ORZE_KNOWLEDGE_DIR"],
        "ORZE_FEEDBACK_DIR": ctx.cfg["_env_ORZE_FEEDBACK_DIR"],
        "ORZE_ROLE_SCRATCH": str(role_scratch),
    })

    # Set cwd for claude-mode roles to role scratch
    popen_cwd = str(role_scratch) if mode == "claude" else None

    # Per-role log directory under .orze/logs/
    log_dir = orze_path(ctx.cfg, "logs", role_name)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{socket.gethostname()}-{os.getpid()}-cycle_{cycle_num:03d}.log"

    logger.info("Running %s [%s] (cycle %d)...",
                 role_name, mode, cycle_num)

    # Execution receipt: snapshot declared outputs before launch.
    # Pass template_vars so '{ideas_file}'-style placeholders in SOP
    # ``produces`` metadata resolve to real paths — otherwise mtime
    # tracking stats literal placeholder strings and evidence is lost.
    _receipt_snapshot_before_launch(role_name, role_cfg, ctx,
                                    template_vars=template_vars)

    # Protect ideas.md: snapshot size before research role runs
    ideas_file = Path(ctx.cfg.get("ideas_file", "ideas.md"))
    ideas_pre_size = 0
    ideas_pre_count = 0
    ideas_md_mtime_pre = 0.0
    if ideas_file.exists():
        _stat = ideas_file.stat()
        ideas_pre_size = _stat.st_size
        ideas_md_mtime_pre = _stat.st_mtime
        ideas_pre_count = len(re.findall(
            r"^## idea-[a-z0-9]+:", ideas_file.read_text(encoding="utf-8"),
            re.MULTILINE))
        # Backup ideas.md to .orze/backups/ with timestamp
        try:
            ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
            backup_path = orze_path(ctx.cfg, "backups", f"ideas.md-{ts}.md")
            shutil.copy2(str(ideas_file), str(backup_path))
            logger.debug("ideas.md backup: %d bytes, %d ideas → %s",
                         ideas_pre_size, ideas_pre_count, backup_path.name)
        except OSError as _backup_err:
            logger.debug("ideas.md backup failed: %s", _backup_err)

    # Launch non-blocking
    log_fh = None
    cycle_start_ts = time.time()  # Record start time for stray file sweeper
    try:
        log_fh = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
            cwd=popen_cwd,
            preexec_fn=_new_process_group,
        )
        # Whether this role should trip the "exit 0 but ideas.md unchanged"
        # soft-failure signal is derived from SOP ``produces`` metadata:
        # if any skill declares it writes ``{ideas_file}``, the check
        # applies. Previously hardcoded against a role-name allowlist.
        writes_ideas = _role_writes_ideas(ctx, role_name, role_cfg)
        # Round-2 B2: per-role stall override (auto-derived for claude
        # roles when not set). Round-2 B3: warmup tolerance.
        stall_min_override = _resolve_role_stall_minutes(
            role_name, role_cfg,
            int(ctx.cfg.get("role_stall_minutes", 5)))
        warmup_s = float(role_cfg.get("stall_warmup_seconds", 60))
        ctx.active_roles[role_name] = RoleProcess(
            role_name=role_name,
            process=proc,
            start_time=cycle_start_ts,
            log_path=log_path,
            timeout=timeout,
            lock_dir=lock_dir,
            cycle_num=cycle_num,
            _log_fh=log_fh,
            ideas_pre_size=ideas_pre_size,
            ideas_pre_count=ideas_pre_count,
            ideas_md_mtime_pre=ideas_md_mtime_pre,
            writes_ideas_file=writes_ideas,
            stall_minutes_override=stall_min_override,
            stall_warmup_seconds=warmup_s,
        )
        # Store start time in role_state for sweep_stray
        role_state["_cycle_start_ts"] = cycle_start_ts
    except Exception as e:
        logger.warning("%s launch error: %s", role_name, e)
        if log_fh and not log_fh.closed:
            log_fh.close()
        _fs_unlock(lock_dir)

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

_professor_bootstrapped = False


# Canonical skill manifests for auto-injected roles. Each list enumerates
# every bundled static SOP the role composes, in the order it was migrated
# from the monolithic rules file. Projects that want per-project behavior
# add their own dynamic skills under <project>/skills/ and extend the list
# in orze.yaml — they do NOT modify these defaults.
_PROFESSOR_SKILLS = [
    "@sop:professor_base",
    "@sop:professor_paper_lake",
    "@sop:professor_web_search",
    "@sop:professor_cross_domain_query",
    "@sop:professor_idea_review",
    "@sop:professor_diversity_enforcement",
    "@sop:professor_gap_closure",
    "@sop:professor_strategy_review",
    "@sop:professor_regression_detection",
    "@sop:professor_steering",
]

_DATA_ANALYST_SKILLS = [
    "@sop:data_analyst_base",
    "@sop:data_analyst_error_analysis",
    "@sop:data_analyst_visualization",
    "@sop:data_analyst_insights",
    "@sop:data_analyst_anomaly_hypotheses",
]

_ENGINEER_SKILLS = [
    "@sop:engineer_base",
    "@sop:engineer_implement",
    "@sop:engineer_fix_bugs",
]

_THINKER_SKILLS = [
    "@sop:thinker_synthesis",
    "@sop:thinker_base",
    "@sop:thinker_phase_a_reformulation",
    "@sop:thinker_phase_b_root_cause",
    "@sop:thinker_axiom_removal",
    "@sop:thinker_phase_c_constraints",
    "@sop:thinker_phase_d_cross_domain",
    "@sop:thinker_phase_e_proposals",
    "@sop:thinker_phase_f_implementation",
]


def _maybe_bootstrap_professor(ctx: RoleContext) -> None:
    """Auto-inject the professor role when GOAL.md is present.

    The professor composes from bundled static SOPs (_PROFESSOR_SKILLS).
    No rules-file copying, no LLM-driven bootstrap — projects that want
    project-specific professor behavior author dynamic skills under
    <project>/skills/ and add them to orze.yaml themselves.
    """
    global _professor_bootstrapped
    if _professor_bootstrapped:
        return
    _professor_bootstrapped = True

    roles = ctx.cfg.get("roles") or {}
    prof_cfg = roles.get("professor")

    if not isinstance(prof_cfg, dict):
        goal_file = ctx.cfg.get("goal_file", "GOAL.md")
        if not Path(goal_file).exists():
            return
        prof_cfg = {
            "mode": "claude",
            "model": "claude-opus-4-6",
            "skills": list(_PROFESSOR_SKILLS),
            "cycle_interval": 600,
        }
        ctx.cfg.setdefault("roles", {})["professor"] = prof_cfg
        logger.info("Auto-enabled professor role (GOAL.md found) "
                    "— %d static SOPs", len(_PROFESSOR_SKILLS))


_analyst_injected = False


def _maybe_inject_analyst(ctx: RoleContext) -> None:
    """Auto-inject the data_analyst + engineer roles when professor is active.

    Both roles compose from bundled static SOPs — no file copying.
    """
    global _analyst_injected
    if _analyst_injected:
        return
    _analyst_injected = True

    roles = ctx.cfg.get("roles") or {}
    if "professor" not in roles:
        return  # no professor, no need for analyst

    if "data_analyst" not in roles:
        ctx.cfg.setdefault("roles", {})["data_analyst"] = {
            "mode": "claude",
            "model": "claude-opus-4-6",
            "skills": list(_DATA_ANALYST_SKILLS),
            "triggered_by": "professor",
            "allowed_tools": "Read,Write,Edit,Glob,Grep,Bash",
            "timeout": _default_role_timeout(len(_DATA_ANALYST_SKILLS)),
        }
        logger.info("Auto-enabled data_analyst role — %d static SOPs",
                    len(_DATA_ANALYST_SKILLS))
        trigger = orze_path(ctx.cfg, "triggers", "_trigger_data_analyst")
        if not trigger.exists():
            trigger.write_text("initial_audit\n", encoding="utf-8")
            logger.info("Created initial DA trigger for first-run audit")

    if "engineer" not in roles:
        ctx.cfg["roles"]["engineer"] = {
            "mode": "claude",
            "model": "claude-opus-4-6",
            "skills": list(_ENGINEER_SKILLS),
            "triggered_by": "professor",
            "allowed_tools": "Read,Write,Edit,Glob,Grep,Bash",
            "timeout": _default_role_timeout(len(_ENGINEER_SKILLS)),
        }
        logger.info("Auto-enabled engineer role — %d static SOPs",
                    len(_ENGINEER_SKILLS))


_thinker_injected = False


def _maybe_inject_thinker(ctx: RoleContext) -> None:
    """Auto-inject the thinker role when professor is active.

    Composes from bundled static SOPs — the seven phases of the thinker
    reasoning flow plus the synthesis + axiom-removal creativity SOPs.
    """
    global _thinker_injected
    if _thinker_injected:
        return
    _thinker_injected = True

    roles = ctx.cfg.get("roles") or {}
    if "thinker" in roles:
        return
    if "professor" not in roles:
        return

    ctx.cfg.setdefault("roles", {})["thinker"] = {
        "mode": "claude",
        "model": "claude-opus-4-6",
        "skills": list(_THINKER_SKILLS),
        "allowed_tools": "Read,Write,Glob,Grep,Bash",
        "timeout": 1800,  # 30min — thinker needs time for analysis + code
        "trigger_conditions": {
            "plateau_patience": 20,
            "failure_cascade": 10,
            "periodic_interval": 10,
        },
        "min_cooldown": 1800,
    }
    logger.info("Auto-enabled thinker role — %d static SOPs",
                len(_THINKER_SKILLS))


def _check_thinker_trigger(role_name: str, role_cfg: dict,
                           ctx: RoleContext) -> str:
    """Check if the thinker should activate. Returns trigger reason or empty string."""
    conditions = role_cfg.get("trigger_conditions", {})
    results_dir = ctx.results_dir

    # Manual trigger (file-based)
    trigger_file = orze_path(ctx.cfg, "triggers", f"_trigger_{role_name}")
    if trigger_file.exists():
        reason = trigger_file.read_text(encoding="utf-8").strip()
        trigger_file.unlink(missing_ok=True)
        return f"manual: {reason}"

    # Count total research cycles
    research_state = ctx.role_states.get("research", {})
    research_cycles = research_state.get("cycles", 0)

    # Periodic trigger
    periodic = conditions.get("periodic_interval", 0)
    if periodic > 0 and research_cycles > 0:
        thinker_state = ctx.role_states.get(role_name, {})
        last_cycle = thinker_state.get("_last_trigger_cycle", 0)
        if research_cycles - last_cycle >= periodic:
            return (f"periodic: {research_cycles} research cycles completed "
                    f"(interval={periodic})")

    # Plateau detection: best metric hasn't improved
    patience = conditions.get("plateau_patience", 0)
    if patience > 0:
        try:
            report_path = results_dir / "report.md"
            if report_path.exists():
                text = report_path.read_text(encoding="utf-8")
                # Count completed experiments (data rows start with "| N |")
                completed = len(re.findall(r"^\|\s*\d+\s*\|", text,
                                           re.MULTILINE))
                thinker_state = ctx.role_states.get(role_name, {})
                best_at = thinker_state.get("_best_metric_at", 0)
                if completed - best_at >= patience and completed > patience:
                    return (f"plateau: no improvement in {completed - best_at} "
                            f"experiments (patience={patience})")
        except Exception:
            pass

    # Failure cascade: consecutive experiments below baseline
    cascade = conditions.get("failure_cascade", 0)
    if cascade > 0:
        consecutive_failures = research_state.get("consecutive_failures", 0)
        if consecutive_failures >= cascade:
            return (f"failure cascade: {consecutive_failures} consecutive "
                    f"failures (threshold={cascade})")

    return ""


_telemetry_started = False


def _maybe_start_telemetry(ctx: RoleContext) -> None:
    """Start background telemetry reporter on first call. No-op if disabled or already running."""
    global _telemetry_started
    if _telemetry_started:
        return
    _telemetry_started = True
    try:
        from orze_pro.telemetry import start_telemetry
        start_telemetry(ctx.cfg, str(ctx.results_dir))
    except Exception as e:
        logger.debug("Telemetry start failed (non-fatal): %s", e)


_bot_launched = False
_bot_proc = None


def _maybe_launch_bot(ctx: RoleContext) -> None:
    """Auto-launch the Telegram/chat bot when bot: config is present.

    The bot runs as a long-lived daemon process alongside the orchestrator.
    Launched once on first call; restarted if the process dies.
    """
    global _bot_launched
    bot_cfg = ctx.cfg.get("bot") or ctx.cfg.get("telegram_bot")
    if not isinstance(bot_cfg, dict):
        return

    config_path = ctx.cfg.get("_config_path", "orze.yaml")

    # Check if bot process is already running (stored module-level, not in role_states
    # which gets JSON-serialized and Popen is not serializable)
    global _bot_proc
    if _bot_proc is not None and _bot_proc.poll() is None:
        return  # still alive

    if _bot_launched and _bot_proc is not None and _bot_proc.poll() is not None:
        logger.warning("Bot process died (exit=%s), restarting...",
                       _bot_proc.returncode)

    _bot_launched = True
    try:
        cmd = [sys.executable, "-m", "orze_pro.agents.bot", "-c", str(config_path)]
        bot_log = Path(ctx.cfg["_orze_dir"]) / "bot.log"
        bot_log.parent.mkdir(parents=True, exist_ok=True)
        proc = subprocess.Popen(
            cmd,
            stdout=open(bot_log, "a"),
            stderr=subprocess.STDOUT,
            preexec_fn=_new_process_group,
        )
        _bot_proc = proc
        logger.info("Auto-launched bot (PID %d)", proc.pid)
    except Exception as e:
        logger.warning("Failed to auto-launch bot (non-fatal): %s", e)


def run_all_roles(ctx: RoleContext) -> None:
    """Check active roles and launch new ones (non-blocking)."""
    # Auto-migrate layout if needed (runs once, fast path via version check)
    try:
        from orze.engine.migrate import _ensure_migrated
        _ensure_migrated(
            ctx.cfg.get("_project_root"),
            ctx.cfg.get("_orze_dir"),
            ctx.results_dir
        )
    except Exception as e:
        logger.warning("Auto-migration failed (non-fatal): %s", e)

    # Bootstrap professor rules on first call
    _maybe_bootstrap_professor(ctx)

    # Auto-inject data analyst when professor is active
    _maybe_inject_analyst(ctx)

    # Auto-inject thinker for creative reasoning on plateau
    _maybe_inject_thinker(ctx)

    # Auto-launch bot if configured
    _maybe_launch_bot(ctx)

    # Start telemetry reporter (background thread, once)
    _maybe_start_telemetry(ctx)

    # Check active roles
    finished = check_active_roles(
        ctx.active_roles,
        ideas_file=ctx.cfg.get("ideas_file", "ideas.md"),
        role_stall_minutes=int(ctx.cfg.get("role_stall_minutes", 5)))

    # Collect per-role results for consolidated notification
    role_contributions: Dict[str, int] = {}  # label -> new_ideas count
    any_ideas_modified = False

    for role_name, outcome in finished:
        role_state = ctx.role_states.setdefault(
            role_name, {"cycles": 0, "last_run_time": 0.0})
        role_state["last_run_time"] = time.time()
        role_state["cycles"] = role_state.get("cycles", 0) + 1

        # Intervention detection: check log for blocked patterns
        try:
            from orze.engine.intervention_detect import detect, should_notify
            log_dir = orze_path(ctx.cfg, "logs", role_name)
            cycle_num = role_state["cycles"]
            log_path = log_dir / f"{socket.gethostname()}-{os.getpid()}-cycle_{cycle_num:03d}.log"
            if log_path.exists():
                log_text = log_path.read_text(errors="replace")
                tail = "\n".join(log_text.splitlines()[-200:])
                extra = ctx.cfg.get("intervention_patterns") or {}
                hit = detect(tail, extra)
                if hit:
                    reason, evidence = hit
                    idea_id = role_state.get("current_idea_id")
                    key = f"{reason}:{role_name}:{idea_id or '-'}"
                    state_file = Path(ctx.cfg["_orze_dir"]) / "state" / "interventions.json"
                    if should_notify(state_file, key):
                        notify("needs_intervention", {
                            "role": role_name,
                            "idea_id": idea_id,
                            "reason": reason,
                            "evidence": evidence,
                            "log_tail": "\n".join(tail.splitlines()[-40:]),
                            "host": socket.gethostname(),
                            "pid": os.getpid(),
                        }, ctx.cfg)
        except Exception as e:
            logger.warning("intervention detect failed: %s", e)

        # Repeated-stub heuristic (post-mortem 2026-04). A run of N
        # byte-identical, tiny cycle logs is a structural failure (e.g.
        # ``Not logged in · Please run /login``) that the normal
        # OUTCOME_ERROR path doesn't always catch — Claude Code can
        # exit 0 while emitting the stub. Fire ``role_circuit_breaker``
        # unconditionally on first detection, then gate via
        # ``_repeated_stub_alerted_at`` until output diverges (cleared
        # below when a non-stub blob is observed).
        try:
            log_dir = orze_path(ctx.cfg, "logs", role_name)
            log_files = sorted(
                (p for p in log_dir.iterdir() if p.is_file()),
                key=lambda p: p.stat().st_mtime, reverse=True,
            )[:_REPEATED_STUB_WINDOW]
            if len(log_files) >= _REPEATED_STUB_WINDOW:
                blobs = [p.read_bytes() for p in log_files]
                tiny = all(len(b) <= _REPEATED_STUB_MAX_BYTES for b in blobs)
                identical = len(set(blobs)) == 1
                if tiny and identical:
                    if not role_state.get("_repeated_stub_alerted_at"):
                        sample = blobs[0].decode(
                            "utf-8", errors="replace").strip()
                        logger.error(
                            "%s: last %d cycle logs byte-identical and "
                            "<=%d bytes — structural failure "
                            "(degraded_repeated_stub). Sample: %r",
                            role_name, _REPEATED_STUB_WINDOW,
                            _REPEATED_STUB_MAX_BYTES, sample,
                        )
                        notify("role_circuit_breaker", {
                            "role": role_name,
                            "reason": "degraded_repeated_stub",
                            "sample": sample[:500],
                            "window": _REPEATED_STUB_WINDOW,
                            "message": (
                                f"{role_name}: last "
                                f"{_REPEATED_STUB_WINDOW} cycles produced "
                                f"the same <={_REPEATED_STUB_MAX_BYTES}-byte "
                                f"output — likely silent auth/login failure"
                            ),
                        }, ctx.cfg)
                        role_state["_repeated_stub_alerted_at"] = time.time()
                else:
                    # Output diverged — re-arm the alert for next streak.
                    role_state.pop("_repeated_stub_alerted_at", None)
        except Exception as e:
            logger.warning("repeated-stub detect failed: %s", e)

        # Clear one-shot trigger reason so the next cycle doesn't replay
        # it as a "ghost trigger". _trigger_reason is set from a file
        # signal (_trigger_professor, _trigger_thinker, etc.) that we
        # already consume+delete on read; keeping it in role_state meant
        # every subsequent regularly-scheduled professor cycle re-ran
        # with the stale reason, e.g. "BLOCKED PORTFOLIO ..." for 37+
        # cycles after the actual block was resolved.
        role_state.pop("_trigger_reason", None)

        # Execution receipt: record which declared SOPs showed evidence
        # of execution (declared output files changed mtime). Must run
        # AFTER ``cycles`` is incremented so the receipt is stamped with
        # the cycle number of the run that just finished.
        _receipt_write_after_completion(role_name, ctx)

        # Sweep stray files from project root to .orze/stray/ or .orze/methods/
        cycle_start_ts = role_state.get("_cycle_start_ts", 0)
        if cycle_start_ts:
            try:
                moved = sweep_stray(ctx, role_name, role_state["cycles"], cycle_start_ts)
                if moved:
                    logger.info("%s cycle %d swept %d stray files",
                                role_name, role_state["cycles"], len(moved))
            except Exception as e:
                logger.warning("sweep_stray failed for %s: %s", role_name, e)
            # Clean up the transient timestamp
            role_state.pop("_cycle_start_ts", None)

        # Professor auto-triggers DA after each OK cycle
        if role_name == "professor" and outcome == OUTCOME_OK:
            roles = ctx.cfg.get("roles") or {}
            if "data_analyst" in roles:
                trigger = orze_path(ctx.cfg, "triggers", "_trigger_data_analyst")
                trigger.write_text(f"professor_cycle_{role_state['cycles']}\n",
                                   encoding="utf-8")
                logger.info("Professor triggered data_analyst")

        role_cfg_ref = (ctx.cfg.get("roles") or {}).get(role_name, {})
        writes_ideas = _role_writes_ideas(ctx, role_name, role_cfg_ref)

        if outcome == OUTCOME_OK:
            role_state["consecutive_errors"] = 0
            role_state["consecutive_timeouts"] = 0
            role_state["consecutive_zero_output"] = 0
            role_state["consecutive_rate_limits"] = 0
            # Legacy alias kept for back-compat with any caller still
            # reading ``consecutive_failures`` (schedulers, dashboards).
            role_state["consecutive_failures"] = 0
            role_state.pop("cooldown_override", None)
            logger.info("%s cycle %d completed successfully",
                        role_name, role_state.get("cycles", 0))

            if writes_ideas:
                ideas_now = parse_ideas(ctx.cfg["ideas_file"])
                prev_count = role_state.get("_prev_idea_count", 0)
                new_ideas = max(0, len(ideas_now) - prev_count)
                role_state["_prev_idea_count"] = len(ideas_now)
                any_ideas_modified = True
                model = role_cfg_ref.get("model", role_name)
                label = model
                for prefix in ("claude-", "gemini-"):
                    if label.startswith(prefix):
                        label = label[len(prefix):]
                        break
                label = label.split("-")[0].split(".")[0]
                role_contributions[label] = (
                    role_contributions.get(label, 0) + new_ideas)
            else:
                role_contributions.setdefault(role_name, 0)

        elif outcome == OUTCOME_SOFT_FAILURE:
            # Exit 0 but no ideas.md modification — track separately so
            # a persistently silent role is visible without being treated
            # as a hard failure (no backoff, no error counter bump).
            zero = role_state.get("consecutive_zero_output", 0) + 1
            role_state["consecutive_zero_output"] = zero
            if zero >= 5:
                logger.error("%s has produced zero output for %d "
                             "consecutive cycles — role is stalled",
                             role_name, zero)
                notify("role_stalled", {
                    "role": role_name,
                    "consecutive_zero_output": zero,
                    "message": f"{role_name} succeeded {zero} times "
                               f"without modifying ideas file",
                }, ctx.cfg)
            role_contributions.setdefault(role_name, 0)

        elif outcome == OUTCOME_TIMEOUT:
            # Killed after exceeding its timeout budget. Bump cooldown
            # by a constant 2× rather than exponential — a timeout is
            # usually "role needs longer", not "role is broken".
            tmo = role_state.get("consecutive_timeouts", 0) + 1
            role_state["consecutive_timeouts"] = tmo
            base_cooldown = role_cfg_ref.get("cooldown", 300)
            role_state["cooldown_override"] = min(
                base_cooldown * 2, MAX_COOLDOWN_OVERRIDE_S)
            logger.warning(
                "%s timed out (#%d consecutive) — cooldown bumped to "
                "%ds. Consider increasing the role's timeout budget "
                "(SOP count=%d suggests %ds).",
                role_name, tmo, role_state["cooldown_override"],
                len(role_cfg_ref.get("skills") or []),
                _default_role_timeout(len(role_cfg_ref.get("skills") or [])))

        elif outcome == OUTCOME_RATE_LIMITED:
            # LLM billing / rate-limit hit — transient provider signal,
            # not a role bug. Do NOT bump consecutive_errors (would trip
            # the circuit-breaker exponential backoff). Track separately
            # so dashboards can surface it, and set a short cooldown
            # nudge so we don't immediately re-spawn a doomed cycle.
            rl = role_state.get("consecutive_rate_limits", 0) + 1
            role_state["consecutive_rate_limits"] = rl
            base_cooldown = role_cfg_ref.get("cooldown", 300)
            role_state["cooldown_override"] = min(
                max(role_state.get("cooldown_override", 0), base_cooldown),
                MAX_COOLDOWN_OVERRIDE_S)
            logger.warning(
                "%s rate-limited (#%d consecutive) — holding cooldown "
                "at %ds, not counting toward error threshold",
                role_name, rl, base_cooldown)
            role_contributions.setdefault(role_name, 0)

        else:  # OUTCOME_ERROR
            err = role_state.get("consecutive_errors", 0) + 1
            role_state["consecutive_errors"] = err
            role_state["consecutive_failures"] = err  # legacy alias
            if err >= 5:
                logger.error("%s has errored %d consecutive times — "
                             "check config or script", role_name, err)
                base_cooldown = role_cfg_ref.get("cooldown", 300)
                # Cap exponential growth — see MAX_COOLDOWN_OVERRIDE_S.
                multiplied = min(
                    base_cooldown * (2 ** (err - 4)),
                    MAX_COOLDOWN_OVERRIDE_S)
                role_state["cooldown_override"] = multiplied
                notify("role_circuit_breaker", {
                    "role": role_name,
                    "consecutive_failures": err,
                    "cooldown": multiplied,
                    "message": f"{role_name} errored {err} consecutive "
                               f"times — cooldown doubled to {multiplied}s",
                }, ctx.cfg)
            elif err >= 3:
                logger.warning("%s has errored %d consecutive times",
                               role_name, err)

    # Send one consolidated role_summary notification
    if finished and any_ideas_modified:
        ideas_now = parse_ideas(ctx.cfg["ideas_file"])
        n_queued = len(get_unclaimed(ideas_now, ctx.results_dir, set(),
                                     lake=getattr(ctx, 'lake', None)))
        total_new = sum(role_contributions.values())
        # Build breakdown string: "5 <opus> + 10 <gemini>"
        parts = [f"{n} <{lbl}>" for lbl, n in role_contributions.items() if n > 0]
        breakdown = " + ".join(parts) if parts else f"{total_new}"
        notify("role_summary", {
            "role": "researcher",
            "new_ideas": total_new,
            "breakdown": breakdown,
            "queued": n_queued,
        }, ctx.cfg)

    # Launch new roles if not running
    for role_name, role_cfg in (ctx.cfg.get("roles") or {}).items():
        if isinstance(role_cfg, dict):
            run_role_step(role_name, role_cfg, ctx)


def run_role_once(role_name: str, ctx: RoleContext) -> None:
    """Run a single agent role synchronously, then exit."""
    roles = ctx.cfg.get("roles") or {}
    if role_name not in roles:
        logger.error("Role '%s' not found in config. Available: %s",
                     role_name, list(roles.keys()))
        return
    role_cfg = roles[role_name]
    if not isinstance(role_cfg, dict):
        logger.error("Role '%s' config is not a dict", role_name)
        return

    logger.info("Running role '%s' once...", role_name)
    run_role_step(role_name, role_cfg, ctx)

    # Wait for it to finish
    while role_name in ctx.active_roles:
        time.sleep(2)
        finished = check_active_roles(
            ctx.active_roles,
            ideas_file=ctx.cfg.get("ideas_file", "ideas.md"),
            role_stall_minutes=int(ctx.cfg.get("role_stall_minutes", 5)))
        for rn, outcome in finished:
            if outcome == OUTCOME_OK:
                logger.info("Role '%s' completed successfully", rn)
            elif outcome == OUTCOME_SOFT_FAILURE:
                logger.warning("Role '%s' exited 0 but produced no output", rn)
            elif outcome == OUTCOME_TIMEOUT:
                logger.warning("Role '%s' timed out", rn)
            elif outcome == OUTCOME_RATE_LIMITED:
                logger.warning("Role '%s' hit LLM rate-limit (transient)", rn)
            else:
                logger.warning("Role '%s' failed", rn)
