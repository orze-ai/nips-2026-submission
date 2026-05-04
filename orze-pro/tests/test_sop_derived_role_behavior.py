"""Tests for SOP-derived role behavior: outcome-based counters + writes_ideas_file."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from orze.engine.roles import (
    OUTCOME_OK,
    OUTCOME_TIMEOUT,
    OUTCOME_ERROR,
    OUTCOME_SOFT_FAILURE,
)
from orze_pro.engine.role_runner import (
    RoleContext,
    _role_writes_ideas,
    run_all_roles,
)


def _ctx(cfg: dict, tmp_path: Path) -> RoleContext:
    return RoleContext(
        cfg=cfg,
        results_dir=tmp_path,
        gpu_ids=[0],
        active_roles={},
        role_states={},
        failure_counts={},
        fix_counts={},
        iteration=0,
    )


def _run_with_finished(ctx: RoleContext, finished: list) -> None:
    """Drive run_all_roles once with a stubbed check_active_roles result.

    Patches out the launch side too so no subprocesses spawn; the test
    only exercises the finished-role accounting loop.
    """
    with patch("orze_pro.engine.role_runner.check_active_roles",
               return_value=finished), \
         patch("orze_pro.engine.role_runner.run_role_step",
               return_value=None), \
         patch("orze_pro.engine.role_runner._maybe_bootstrap_professor"), \
         patch("orze_pro.engine.role_runner._maybe_inject_analyst"), \
         patch("orze_pro.engine.role_runner._maybe_inject_thinker"), \
         patch("orze_pro.engine.role_runner._maybe_launch_bot"):
        run_all_roles(ctx)


def test_timeout_does_not_increment_errors(tmp_path):
    """OUTCOME_TIMEOUT bumps consecutive_timeouts, not consecutive_errors."""
    cfg = {
        "ideas_file": str(tmp_path / "ideas.md"),
        "roles": {"engineer": {"mode": "claude", "skills": [],
                                "cooldown": 300}},
    }
    ctx = _ctx(cfg, tmp_path)

    _run_with_finished(ctx, [("engineer", OUTCOME_TIMEOUT)])

    state = ctx.role_states["engineer"]
    assert state.get("consecutive_errors", 0) == 0
    assert state["consecutive_timeouts"] == 1
    # Cooldown bumped by constant 2x on timeout
    assert state["cooldown_override"] == 300 * 2


def test_error_increments_errors(tmp_path):
    """OUTCOME_ERROR bumps consecutive_errors."""
    cfg = {
        "ideas_file": str(tmp_path / "ideas.md"),
        "roles": {"engineer": {"mode": "claude", "skills": [],
                                "cooldown": 300}},
    }
    ctx = _ctx(cfg, tmp_path)

    _run_with_finished(ctx, [("engineer", OUTCOME_ERROR)])

    state = ctx.role_states["engineer"]
    assert state["consecutive_errors"] == 1
    # Legacy alias stays in sync.
    assert state["consecutive_failures"] == 1
    assert state.get("consecutive_timeouts", 0) == 0


def test_soft_failure_increments_zero_output(tmp_path):
    """OUTCOME_SOFT_FAILURE bumps consecutive_zero_output, not errors."""
    cfg = {
        "ideas_file": str(tmp_path / "ideas.md"),
        "roles": {"research": {"mode": "claude", "skills": [],
                                "cooldown": 300}},
    }
    ctx = _ctx(cfg, tmp_path)

    _run_with_finished(ctx, [("research", OUTCOME_SOFT_FAILURE)])

    state = ctx.role_states["research"]
    assert state["consecutive_zero_output"] == 1
    assert state.get("consecutive_errors", 0) == 0
    assert state.get("consecutive_timeouts", 0) == 0
    assert "cooldown_override" not in state


def test_ok_clears_all_counters(tmp_path):
    """OUTCOME_OK resets every consecutive-* counter and clears cooldown_override."""
    cfg = {
        "ideas_file": str(tmp_path / "ideas.md"),
        "roles": {"engineer": {"mode": "claude", "skills": [],
                                "cooldown": 300}},
    }
    ctx = _ctx(cfg, tmp_path)
    ctx.role_states["engineer"] = {
        "cycles": 5, "last_run_time": 0.0,
        "consecutive_errors": 3, "consecutive_timeouts": 2,
        "consecutive_zero_output": 1, "cooldown_override": 1200,
    }

    _run_with_finished(ctx, [("engineer", OUTCOME_OK)])

    state = ctx.role_states["engineer"]
    assert state["consecutive_errors"] == 0
    assert state["consecutive_timeouts"] == 0
    assert state["consecutive_zero_output"] == 0
    assert "cooldown_override" not in state


def test_writes_ideas_file_true_when_sop_produces_ideas(tmp_path):
    """Role whose SOP declares produces: [{ideas_file}] → writes_ideas=True."""
    ideas_path = tmp_path / "ideas.md"
    sop_path = tmp_path / "skills" / "fake_research.skill.md"
    sop_path.parent.mkdir(parents=True, exist_ok=True)
    sop_path.write_text(
        "---\nid: fake_research\nproduces:\n  - '{ideas_file}'\n---\nBody.\n",
        encoding="utf-8",
    )

    cfg = {
        "ideas_file": str(ideas_path),
        "roles": {"research": {"mode": "claude",
                                "skills": [str(sop_path)]}},
    }
    # project_root = results_dir.parent, so put results_dir one level deep.
    (tmp_path / "results").mkdir()
    ctx = _ctx(cfg, tmp_path / "results")
    # Align _receipt_discover_outputs project_root with sop_path:
    # project_root = results_dir.parent = tmp_path, and sop_path is under
    # tmp_path/skills so is_absolute() short-circuits the project_root join.
    assert _role_writes_ideas(ctx, "research") is True


def test_writes_ideas_file_false_when_sop_produces_other(tmp_path):
    """Role whose SOPs produce GOAL.md / RESEARCH_RULES.md → writes_ideas=False."""
    sop_path = tmp_path / "skills" / "fake_engineer.skill.md"
    sop_path.parent.mkdir(parents=True, exist_ok=True)
    sop_path.write_text(
        "---\nid: fake_engineer\n"
        "produces:\n  - GOAL.md\n  - RESEARCH_RULES.md\n---\nBody.\n",
        encoding="utf-8",
    )

    cfg = {
        "ideas_file": str(tmp_path / "ideas.md"),
        "roles": {"engineer": {"mode": "claude",
                                "skills": [str(sop_path)]}},
    }
    (tmp_path / "results").mkdir()
    ctx = _ctx(cfg, tmp_path / "results")
    assert _role_writes_ideas(ctx, "engineer") is False


def test_writes_ideas_file_false_for_role_without_sops(tmp_path):
    """Role with no SOP metadata → writes_ideas=False (derived-from-SOPs policy)."""
    cfg = {
        "ideas_file": str(tmp_path / "ideas.md"),
        "roles": {"engineer": {"mode": "claude", "skills": []}},
    }
    (tmp_path / "results").mkdir()
    ctx = _ctx(cfg, tmp_path / "results")
    assert _role_writes_ideas(ctx, "engineer") is False
