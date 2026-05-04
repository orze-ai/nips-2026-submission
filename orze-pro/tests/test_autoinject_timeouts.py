"""Test auto-injected strategy roles + SOP-derived runtime behavior."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from orze_pro.engine.role_runner import (
    _default_role_timeout,
    _maybe_inject_analyst,
    _maybe_inject_thinker,
    _DATA_ANALYST_SKILLS,
    _ENGINEER_SKILLS,
    _THINKER_SKILLS,
)
import orze_pro.engine.role_runner as rr_mod


def _reset_inject_flags():
    """Reset module-level once-only flags so tests can re-inject."""
    rr_mod._analyst_injected = False
    rr_mod._thinker_injected = False


def _ctx(cfg: dict, tmp_path: Path):
    """Minimal RoleContext stub — only .cfg and .results_dir are read."""
    cfg.setdefault("_orze_dir", str(tmp_path / ".orze"))
    cfg.setdefault("_project_root", str(tmp_path))
    cfg.setdefault("_env_ORZE_RESULTS_DIR", str(tmp_path / "orze_results"))
    c = MagicMock()
    c.cfg = cfg
    c.results_dir = tmp_path
    return c


def test_default_role_timeout_formula():
    """Timeout = 600 + 240 × N(skills); zero-skill role gets base 600s."""
    assert _default_role_timeout(0) == 600
    assert _default_role_timeout(1) == 840
    assert _default_role_timeout(5) == 1800
    assert _default_role_timeout(9) == 2760


def test_data_analyst_timeout_follows_formula(tmp_path):
    """Auto-injected DA timeout matches 600 + 240 × len(DA skills)."""
    _reset_inject_flags()
    cfg = {"roles": {"professor": {"mode": "claude"}}}
    _maybe_inject_analyst(_ctx(cfg, tmp_path))

    da = cfg["roles"]["data_analyst"]
    expected = _default_role_timeout(len(_DATA_ANALYST_SKILLS))
    assert da["timeout"] == expected
    assert len(da["skills"]) == len(_DATA_ANALYST_SKILLS)


def test_engineer_timeout_follows_formula(tmp_path):
    """Auto-injected engineer timeout matches 600 + 240 × len(engineer skills)."""
    _reset_inject_flags()
    cfg = {"roles": {"professor": {"mode": "claude"}}}
    _maybe_inject_analyst(_ctx(cfg, tmp_path))

    eng = cfg["roles"]["engineer"]
    expected = _default_role_timeout(len(_ENGINEER_SKILLS))
    assert eng["timeout"] == expected
    assert len(eng["skills"]) == len(_ENGINEER_SKILLS)


def test_thinker_keeps_its_timeout(tmp_path):
    """Thinker injection still sets its own explicit timeout (30min)."""
    _reset_inject_flags()
    cfg = {"roles": {"professor": {"mode": "claude"}}}
    _maybe_inject_thinker(_ctx(cfg, tmp_path))

    th = cfg["roles"]["thinker"]
    assert th["timeout"] == 1800
    assert len(th["skills"]) == len(_THINKER_SKILLS)


def test_inject_does_not_override_user_config(tmp_path):
    """If user explicitly sets data_analyst in orze.yaml, leave it alone."""
    _reset_inject_flags()
    cfg = {"roles": {
        "professor": {"mode": "claude"},
        "data_analyst": {"mode": "claude", "timeout": 300},
    }}
    _maybe_inject_analyst(_ctx(cfg, tmp_path))

    assert cfg["roles"]["data_analyst"]["timeout"] == 300


# -----------------------------------------------------------------------------
# --dangerously-skip-permissions default (autonomous roles run headless)
# -----------------------------------------------------------------------------

from orze_pro.engine.role_runner import build_claude_cmd


def _base_role_cfg():
    return {"mode": "claude", "skills": ["@sop:professor_base"]}


def _tmpl_vars(tmp_path):
    return {"results_dir": str(tmp_path), "role_name": "professor"}


def test_skip_permissions_on_by_default(tmp_path):
    """Auto-injected roles run headless under the daemon — the default
    must pass --dangerously-skip-permissions so the permission layer
    can't stall on prompts no human will answer."""
    cmd = build_claude_cmd(_base_role_cfg(), _tmpl_vars(tmp_path))
    assert "--dangerously-skip-permissions" in cmd


def test_skip_permissions_opt_out(tmp_path):
    """Projects with stricter sandboxing set the key explicitly false."""
    cfg = _base_role_cfg()
    cfg["dangerously_skip_permissions"] = False
    cmd = build_claude_cmd(cfg, _tmpl_vars(tmp_path))
    assert "--dangerously-skip-permissions" not in cmd


def test_skip_permissions_opt_in_is_idempotent(tmp_path):
    """Explicit True matches the default — flag appears exactly once."""
    cfg = _base_role_cfg()
    cfg["dangerously_skip_permissions"] = True
    cmd = build_claude_cmd(cfg, _tmpl_vars(tmp_path))
    assert cmd.count("--dangerously-skip-permissions") == 1
