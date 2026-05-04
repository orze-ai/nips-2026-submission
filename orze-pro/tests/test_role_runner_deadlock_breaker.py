"""Tests for the generalized auto-trigger-on-stall deadlock breaker.

Replaces the old engineer-specific behaviour at
``role_runner.run_role_step`` (was: ``stall_count >= 10 and role_name ==
"engineer"``). Now any role with ``triggered_by:`` set is auto-triggered
when its source stalls; opt out per-role with
``auto_trigger_on_stall: false``.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pytest

from orze_pro.engine.role_runner import RoleContext, run_role_step


def _ctx(cfg: dict, tmp_path: Path, role_states: dict | None = None,
         active_roles: dict | None = None) -> RoleContext:
    return RoleContext(
        cfg=cfg,
        results_dir=tmp_path,
        gpu_ids=[0],
        active_roles=active_roles if active_roles is not None else {},
        role_states=role_states if role_states is not None else {},
        failure_counts={},
        fix_counts={},
        iteration=0,
    )


def _base_cfg(tmp_path: Path, roles: dict) -> dict:
    return {
        "_orze_dir": str(tmp_path / ".orze"),
        "ideas_file": str(tmp_path / "ideas.md"),
        "goal_file": str(tmp_path / "GOAL.md"),
        "roles": roles,
    }


def _auto_triggered(role_name: str, role_cfg: dict, ctx: RoleContext,
                    caplog) -> bool:
    """True iff the deadlock breaker fired and logged its WARNING.

    Mode is set to ``script`` with no ``script`` field so run_role_step
    returns shortly after the trigger gate without spawning anything.
    The "Auto-triggering ..." WARNING is the load-bearing observable.
    """
    caplog.clear()
    with caplog.at_level(logging.WARNING,
                         logger="orze_pro.engine.role_runner"):
        run_role_step(role_name, role_cfg, ctx)
    return any("Auto-triggering" in rec.message for rec in caplog.records)


def test_engineer_default_fires_after_stall(tmp_path, caplog):
    """Default behavior preserved: engineer with triggered_by + stall>=10 fires."""
    role_cfg = {"mode": "script", "triggered_by": "fsm", "cooldown": 300}
    cfg = _base_cfg(tmp_path, {"engineer": role_cfg, "fsm": {}})
    ctx = _ctx(cfg, tmp_path,
               role_states={"fsm": {"consecutive_zero_output": 10}})
    assert _auto_triggered("engineer", role_cfg, ctx, caplog)


def test_engineer_below_threshold_does_not_fire(tmp_path, caplog):
    role_cfg = {"mode": "script", "triggered_by": "fsm", "cooldown": 300}
    cfg = _base_cfg(tmp_path, {"engineer": role_cfg, "fsm": {}})
    ctx = _ctx(cfg, tmp_path,
               role_states={"fsm": {"consecutive_zero_output": 9}})
    assert not _auto_triggered("engineer", role_cfg, ctx, caplog)


def test_data_analyst_default_fires_after_stall(tmp_path, caplog):
    """Generalization win: any triggered_by role now fires by default."""
    role_cfg = {"mode": "script", "triggered_by": "professor", "cooldown": 300}
    cfg = _base_cfg(tmp_path, {"data_analyst": role_cfg, "professor": {}})
    ctx = _ctx(cfg, tmp_path,
               role_states={"professor": {"consecutive_zero_output": 12}})
    assert _auto_triggered("data_analyst", role_cfg, ctx, caplog)


def test_triggered_role_with_auto_trigger_false_does_not_fire(tmp_path, caplog):
    """Opt-out: auto_trigger_on_stall=False prevents auto-fire even when stalled."""
    role_cfg = {
        "mode": "script",
        "triggered_by": "professor",
        "auto_trigger_on_stall": False,
        "cooldown": 300,
    }
    cfg = _base_cfg(tmp_path, {"data_analyst": role_cfg, "professor": {}})
    ctx = _ctx(cfg, tmp_path,
               role_states={"professor": {"consecutive_zero_output": 100}})
    assert not _auto_triggered("data_analyst", role_cfg, ctx, caplog)


def test_role_without_triggered_by_never_enters_branch(tmp_path, caplog):
    """No triggered_by => deadlock-breaker branch is unreachable.

    The new policy gates auto-trigger on `triggered_by` being set, so
    roles that nobody schedules can't be accidentally auto-fired.
    """
    role_cfg = {"mode": "script", "cooldown": 300}
    cfg = _base_cfg(tmp_path, {"research": role_cfg})
    ctx = _ctx(cfg, tmp_path)
    assert not _auto_triggered("research", role_cfg, ctx, caplog)


def test_custom_threshold_respected(tmp_path, caplog):
    """auto_trigger_stall_threshold overrides the default of 10."""
    role_cfg = {
        "mode": "script",
        "triggered_by": "professor",
        "auto_trigger_stall_threshold": 25,
        "cooldown": 300,
    }
    cfg = _base_cfg(tmp_path, {"data_analyst": role_cfg, "professor": {}})

    ctx_low = _ctx(cfg, tmp_path,
                   role_states={"professor": {"consecutive_zero_output": 20}})
    assert not _auto_triggered("data_analyst", role_cfg, ctx_low, caplog)

    ctx_hi = _ctx(cfg, tmp_path,
                  role_states={"professor": {"consecutive_zero_output": 25}})
    assert _auto_triggered("data_analyst", role_cfg, ctx_hi, caplog)
