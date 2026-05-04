# SOP-Derived Role Behavior — Refactor Plan

> **Status**: deferred. Immediate symptoms are fixed on main (see
> context below); this plan captures the deeper systematic refactor
> to unify role runtime behavior around SOP frontmatter as the single
> source of truth.

## Context

Three recent band-aid fixes landed on main:

| Commit | Repo | What it patched |
|---|---|---|
| `cfe8b77` | orze | Added `RoleProcess.writes_ideas_file` flag to skip soft-failure check for strategy roles |
| `eb2ae54` | orze-pro | Hardcoded `timeout: 1800` for data_analyst, `timeout: 1200` for engineer at auto-inject sites |
| `5dc3fcc` | orze-pro | `explicitly_triggered` bypasses cooldown + logs circuit-breaker state |

Each of these is a band-aid over the same underlying issue: **role
runtime behavior is hardcoded against role names instead of derived
from each role's SOP manifest**. Add a new role tomorrow and all
three problems return.

## Target design

SOPs are already the source of truth for what a role *does* (prompt
composition). Make them the source of truth for how the daemon *runs*
the role too:

```
role.timeout         = base + per_skill × len(skills)
                       (no per-role hardcoded number)

role.success_signal  = any(sop.produces modified during cycle)
                       (derived from receipts, not a name set)

role.backoff_policy  = only OUTCOME_ERROR triggers exponential
                       backoff; OUTCOME_TIMEOUT bumps cooldown once;
                       OUTCOME_SOFT_FAILURE tracked separately

role.triggered_by    = explicit trigger always wins over cooldown
                       (already landed in 5dc3fcc)
```

Every existing and future role auto-inherits correct behavior from
its skills list. Zero per-role hardcoding.

## Work items

### Already done (on main)

- [x] **orze**: `OUTCOME_OK / OUTCOME_TIMEOUT / OUTCOME_ERROR /
      OUTCOME_SOFT_FAILURE` enum in `orze.engine.roles`
- [x] **orze**: `check_active_roles()` returns `(name, outcome)` tuples
- [x] **orze**: `is_success(outcome)` back-compat helper
- [x] **orze**: tests for each outcome path (5 tests pass)

### Remaining (to land together)

- [ ] **orze-pro**: update `role_runner.py:1088` caller loop to
      unpack `(role_name, outcome)` and branch on outcome. Strategy-role
      detection comes from `rp.writes_ideas_file` (already set at
      launch site), not a hardcoded `strategy_roles` set.
- [ ] **orze-pro**: split failure counters by outcome
  - `consecutive_errors` (from `OUTCOME_ERROR`) → triggers exponential
    backoff (`cooldown_override = 300 * 2^(n-4)`)
  - `consecutive_timeouts` (from `OUTCOME_TIMEOUT`) → bumps cooldown
    by constant 2× (not exponential) and logs a hint to increase
    the role's timeout budget
  - `consecutive_zero_output` (from `OUTCOME_SOFT_FAILURE`) → already
    tracked; keep as-is
- [ ] **orze-pro**: delete hardcoded `timeout: 1800` (data_analyst)
      and `timeout: 1200` (engineer) at the two `_maybe_inject_*`
      sites. Replace with a single helper:
      ```python
      def _default_role_timeout(skills_count: int) -> int:
          return 600 + 240 * skills_count
      ```
      Applied at inject-time only when `timeout` not explicitly set.
      Project-level `orze.yaml` timeout still wins.
- [ ] **orze-pro**: at launch site (`RoleProcess(...)` construction),
      derive `writes_ideas_file` from SOP metadata instead of the
      hardcoded `_STRATEGY_ROLES` set:
      ```python
      declared = _receipt_discover_outputs(role_cfg, project_root,
                                           template_vars)
      ideas_file = ctx.cfg.get("ideas_file", "ideas.md")
      writes_ideas = any(ideas_file in outputs
                         for outputs in declared.values())
      ```
- [ ] **orze-pro**: delete the second hardcoded `strategy_roles` set
      in `role_runner.py:1122`. With outcome-based logic it becomes
      redundant — `OUTCOME_OK` already means "success by whatever
      criterion applies to this role".
- [ ] **orze-pro**: tests
  - Timeout outcome does NOT increment `consecutive_errors`
  - Error outcome DOES increment `consecutive_errors`
  - Auto-injected role with N skills gets timeout `600 + 240*N`
  - Role whose only `@sop:` produces `{ideas_file}` →
    `writes_ideas_file=True`
  - Role whose SOPs produce `GOAL.md, RESEARCH_RULES.md` →
    `writes_ideas_file=False`
- [ ] **Release**: bump orze 3.4.6 + orze-pro 0.7.2 together.
      orze-pro 0.7.2 requires orze>=3.4.6 (outcome tuple is a
      breaking return-signature change).

## Files touched

- `/workspace/orze/src/orze/engine/roles.py` (done)
- `/workspace/orze/src/orze/engine/process.py` (done — has `writes_ideas_file` field)
- `/workspace/orze/tests/test_roles_soft_failure.py` (done)
- `/workspace/orze-pro/src/orze_pro/engine/role_runner.py`
  - `_maybe_inject_analyst`: remove `timeout: 1800` / `timeout: 1200`
  - Add `_default_role_timeout(skills_count)` helper
  - `RoleProcess(...)` construction: use SOP-derived
    `writes_ideas_file`; delete the `_STRATEGY_ROLES` constant
  - `for role_name, outcome in finished:`: branch on outcome enum
  - Delete inner `strategy_roles = {...}` at line 1122
  - Split failure counters by outcome

## Non-goals

- Do NOT add new SOP frontmatter fields (e.g., `expected_duration`)
  in this refactor. The `600 + 240 * N` formula is sufficient; if
  a particular SOP needs a longer budget later, we can add a
  `duration_hint` field then. YAGNI for now.
- Do NOT redesign `consecutive_zero_output` tracking — the current
  per-role counter works.
- Do NOT touch the circuit-breaker reset logic — success already
  clears `cooldown_override` and `consecutive_failures` on line
  1089-1092.

## Validation plan

1. All existing orze tests pass (unchanged behavior for research roles
   appending to ideas.md)
2. All existing orze-pro tests pass
3. New tests in both repos pass (listed above)
4. Live validation on nexar_collision: after deploying orze 3.4.6 +
   orze-pro 0.7.2 + daemon restart, observe:
   - data_analyst cycles complete (no timeouts on 5-SOP workload)
   - Strategy-role soft-failure warnings stop
   - `_metric_patterns_cache.json` still works (regression)
   - Research / research_gemini still write `ideas.md`

## Why deferred

Refactor was blocked mid-session when the engineer's local Claude
client kept crashing (segfault in `libgcc_s` / SIGABRT from Node on
SSH disconnect). Root cause of crashes: large tool-call outputs +
possibly the 1.19GB `orze.log` file being scanned repeatedly. Log
was rotated; next attempt should:

1. Run the working claude session inside a detached tmux so SSH
   drops don't terminate it
2. Keep each turn's tool calls small and responses under ~30 lines
3. Edit `role_runner.py` with surgical `Edit` tool calls, not full
   `Write` rewrites (file is 1400+ lines)
