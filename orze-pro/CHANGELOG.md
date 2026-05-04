# Changelog — orze-pro

## 0.10.0 — silent-role-death fixes (round-1 + round-2)

Companion to orze 4.3.0. Same 2026-04 incident: 5+ days of identical
35-byte stub outputs from claude-mode roles, masked by stale
`cooldown_override` and a global stall timer that didn't account for
roles with 6–10 skills.

### Added
- **Repeated-stub detector**
  (`src/orze_pro/engine/role_runner.py`) — after each role cycle,
  inspects the last 5 files in `.orze/logs/<role>/`. If all are
  byte-identical AND each is <=200 bytes, treats as a structural
  failure (`degraded_repeated_stub`) and emits a
  `role_circuit_breaker` notification regardless of subprocess rc.
  Gated by `role_state["_repeated_stub_alerted_at"]` so the alert
  fires once per streak; cleared automatically when output diverges.
- **Auto-derived per-role timeout**
  (`src/orze_pro/engine/role_runner.py:_resolve_role_timeout`) —
  when `timeout:` is not set explicitly,
  `max(300, 60 * len(skills))` for `mode: claude`. Logs the
  derivation at INFO once per role at registration. Explicit
  `timeout:` in `orze.yaml` always wins.
- **Auto-derived per-role `stall_minutes`**
  (`src/orze_pro/engine/role_runner.py:_resolve_role_stall_minutes`)
  — `max(5, 2 * len(skills))` for `mode: claude` when not set
  explicitly. Plumbed through `RoleProcess.stall_minutes_override`
  (added in orze) so `check_active_roles` uses the per-role timer
  instead of the global `role_stall_minutes`.
- **Per-role `stall_warmup_seconds`**
  (`src/orze_pro/engine/role_runner.py`) — 60s default,
  configurable per role. Plumbed into `RoleProcess`. The stall timer
  in orze `_is_role_stalled` doesn't begin counting until first
  stdout byte received OR warmup elapses, whichever is sooner.

### Changed
- **`cooldown_override` capped at 24h**
  (`src/orze_pro/engine/role_runner.py`) — new
  `MAX_COOLDOWN_OVERRIDE_S=86400` clamps every write site
  (timeout, rate-limit, error-path `2^(err-4)` doubling). Also
  clamps on first read of `role_state` per cycle via
  `_clamp_cooldown_override()`, defanging historical bad values
  (5.1×10⁹⁹s observed in production) loaded from
  `.orze_state_<host>.json` written by pre-cap builds. One-time
  WARNING log per role on clamp. `clamp_all_cooldowns()` exported
  for orchestrator-side bulk-clamp.

## 0.9.4 — validator field-path traversal fix + evolution pause fix

### Fixed
- **Validator dot-path traversal** — `_check_condition` used `dict.get(field)` which returned `None` for dot-separated paths like `data.competition_id`. Now traverses nested dicts correctly via `_resolve_dotpath`. Previously, any validator referencing a nested field would silently fail and could incorrectly block ideas.
- **Validator None=vacuous for all operators** — When a field is absent from the idea config, all operators (including `equals` and `not_equals`) now return True (vacuously satisfied). Previously, `equals` returned False and `not_equals` returned True on absent fields, causing false rejections.
- **Evolution FSM no longer re-creates pause every tick** — Removed `maintain: [pause_research]` from the `paused` state in `evolution.yaml`. The pause is still set on the transition to `paused`, but is no longer re-asserted every FSM cycle (~30s), which was blocking all research even when cleared by external watchdogs.

## 0.8.3 — role logging, stray sweeper, and intervention detection

### Changed
- **Role logs moved to `.orze/logs/<role>/`** — Per-role log directories under `.orze/` instead of `results/_research_logs/`
- **Stray file sweeper integration** — Role-generated files at project root are automatically quarantined to `orze_results/stray/<role>/cycle_NNN/` or `orze_results/methods/<role>/cycle_NNN/` for `.py`/`.sh` files

### Added
- Intervention detection hook in role post-completion — Scans logs for blocked patterns (HF gated, missing keys, OOM, etc.) and fires `needs_intervention` notification with 6h cooldown
- Root pollution notification — Alerts when roles write files to project root (with evidence of which files)

### Fixed
- Log paths now use `orze_path(cfg, "logs", role_name)` for consistent `.orze/` layout

## 0.8.2 — ANON org migration

### Changed
- **GitHub org migrated** from `ANON/orze` to `ANON/orze` — README cross-link to the orze repo
- **Paperdog Docker image migrated** from `ANON/paperdog:latest` to `ANON/paperdog:latest` — `professor_paper_lake.skill.md` (container image + run command)

### Fixed
- **Repo hygiene** — untrack `__pycache__/*.pyc` files that were previously committed despite being covered by `.gitignore`

## 0.8.1 — FSM trigger path fix + Python 3.9

### Fixed
- **Trigger file path mismatch** — FSM plugins wrote triggers to `results_dir/` but role_runner read from `.orze/triggers/`; all FSM-triggered roles (code_evolution, engineer, meta_research) never fired
- **Python 3.9 crash in all 4 FSM plugins** — `str | None` union syntax requires 3.10+; added `from __future__ import annotations`

## 0.8.0 — Unified install UX

### Changed
- **One-line install** — `ORZE_PRO_KEY=... curl -sL https://ANON.example/install | bash`
- **README rewritten** — single install path, removed step-by-step and alternative activation noise
- **Requires orze >= 4.0.0**

## 0.7.13

### Fixed
- **`discover_skills()` TypeError on string input** — coerces `project_root` to `Path` at function entry.

## 0.7.11

### Fixed

- **`call_gemini` no longer returns 2-char junk when thinking exhausts
  `maxOutputTokens`.** Gemini 2.5 Pro charges internal reasoning tokens
  against the output budget. With `web_search=True` (the default for
  research and the FSM idea_verifier), the search planner regularly
  burned through the 8192-token cap and returned a `finishReason:
  MAX_TOKENS` response with a few whitespace chars of real output.
  Orze-pro was logging that as `returned 2 chars` and the caller
  treated it as a successful answer — research soft-failed every
  cycle, accumulating ERROR-level `consecutive soft failures` warnings
  in `orze.log` even though the role was exiting 0.

  Fix: raise the default `maxOutputTokens` to 32768 so thinking has
  real headroom, filter thought parts explicitly (`thought: true`),
  and detect `MAX_TOKENS` + `<32` chars of stripped text as a miss so
  the fallback chain gets a chance at the next model.

  Verified against a 118K-char research prompt with `web_search=True`:
  three consecutive calls now return 498 / 1152 / 2204 chars of parseable
  ideas instead of the previous ~80% 2-char failure rate.

## 0.7.10

### Changed

- **`--dangerously-skip-permissions` is now the default for `mode:
  claude` roles.** Auto-injected roles (professor, data_analyst,
  engineer, thinker) run headless under the orze daemon — permission
  prompts in `claude -p` have no stdin to answer, so the per-command
  permission layer was stalling roles instead of protecting anything.
  The real sandbox — `--allowedTools`, the orze.yaml `mcp_servers`
  list, the role's composed SOP skills — is unchanged.

  New role-config key: `dangerously_skip_permissions` (default `true`).
  Projects that need stricter sandboxing opt out per-role:
  ```yaml
  roles:
    professor:
      dangerously_skip_permissions: false
  ```

  This closes the last out-of-the-box gap for the paperdog paper-lake
  SOP shipped in 0.7.9 — fresh projects no longer need to add
  `Bash(docker ps *)` / `Bash(curl http://localhost:8000/*)` allow
  rules to `.claude/settings.json` before the professor can self-heal
  the container. Tests:
  `test_skip_permissions_on_by_default`,
  `test_skip_permissions_opt_out`,
  `test_skip_permissions_opt_in_is_idempotent`.

## 0.7.9

### Added

- **`professor_paper_lake` bundled SOP — paperdog integration
  out-of-the-box.** Auto-injected into every professor role via
  `_PROFESSOR_SKILLS`, slotted right after `professor_base` so the
  paper lake is the *first* BFS pass before WebSearch/cross-domain
  queries. The SOP defines the full lifecycle: self-heal the local
  container (`ANON/paperdog:latest`, name `paperdog`) at cycle
  start, bootstrap from the `GET /` manifest, pick endpoints from
  `endpoints[]` rather than guessing, verify every cited arxiv
  reference, write findings to `GOAL.md` under `Prior Art & Known
  Solutions`, and log each call to `results/_paper_lake_usage.log`.

  Successor to the removed Paper Lantern auto-injection (v0.7.8) —
  same "one-line opt-in, zero-line default" ergonomics, but backed
  by a self-hosted unlimited corpus (~400K arxiv CS papers) instead
  of a shared rate-capped key.

  Projects that want a different paper-lake backend (or none) can
  override the skill list in their `orze.yaml` under
  `roles.professor.skills` and drop `@sop:professor_paper_lake`.

## 0.7.8

### Removed

- **Paper Lantern MCP integration is gone.** `orze_pro.paperlantern`
  (the bundled API key + URL) and the
  `_default_professor_mcp_servers()` / `_merge_professor_mcp_defaults()`
  machinery that auto-injected it into every professor role are
  deleted. The shared API key was rate-capped (~20 queries/month) and
  the service has been superseded by project-level paper-lake
  backends (e.g., self-hosted paperdog). Projects that want a specific
  MCP server on the professor now declare it explicitly in
  `orze.yaml` under `roles.professor.mcp_servers` — same pre-existing
  hook, no auto-merge. `mcp_servers: {}` and unset both mean "no MCP
  servers" now; before, unset meant "Paper Lantern gets injected."

  Bundled `professor_web_search` SOP no longer has a "Step 2b: Paper
  Lantern" section. The web-search flow is now: WebSearch (≥3
  queries) → WebFetch promising URLs → write findings to GOAL.md.

## 0.7.7

### Added

- **Role stall detection — threaded through.** Both `check_active_roles`
  call sites (non-blocking poll in `run_all_roles`; wait-to-finish
  poll in `run_role_once`) now read `role_stall_minutes` from
  `orze.yaml` and pass it through. Default `5` when the key is
  omitted — well under the typical 20-min role timeout. Set to `0`
  to disable. Requires orze ≥ 3.4.11 for the underlying kill logic.

- **`ideas.md` mtime snapshot at role launch.** New
  `RoleProcess.ideas_md_mtime_pre` is captured alongside
  `ideas_pre_size` / `ideas_pre_count`, populated from the same
  `stat()` call, and fed into orze 3.4.11's cross-daemon
  `_ideas_were_modified` fallback. Silently leaves the field at
  `0.0` when `ideas.md` doesn't exist at launch — the consumer
  treats `0.0` as "no snapshot" and skips the mtime check.

### Fixed

- **Claude-backend CLI discovery uses `sys.prefix`.** `role_runner`
  previously located `orze-claude` via
  `Path(sys.executable).parent / "orze-claude"`, which resolves
  symlinks. In environments where `python` is a symlink into a
  non-bin directory, the binary wasn't found. Switched to
  `sys.prefix + "/bin/orze-claude"` so the lookup tracks the active
  venv regardless of symlink topology.

- **`claude_bin` defaults to `orze-claude`.** Subscription-first
  with API-key fallback is now the default for `mode: claude` roles
  — matches the CLI most projects already installed. Projects that
  need the raw Claude Code CLI can still set `claude_bin: claude`
  in `orze.yaml`.

- **`OUTCOME_RATE_LIMITED` no longer trips the circuit breaker.**
  A rate-limit exit (identified by orze's `_is_rate_limit_exit`
  signatures) is a transient provider-side event, not a role
  misconfiguration. `run_all_roles` now short-circuits the
  consecutive-failure accounting on this outcome so one rate-limit
  hit doesn't cascade the whole role into backoff.

## 0.7.6

### Fixed

- **`role_runner` script-mode now prepends orze's site-packages to
  `PYTHONPATH`.** The built-in FSM runner (scaffolded by `orze --init`)
  does `from orze.fsm.runner import main`, but script-mode roles are
  spawned via `cfg["python"]` which may point at a project venv that
  doesn't have `orze` installed. Every FSM cycle failed with
  `ModuleNotFoundError: No module named 'orze'`. The PYTHONPATH
  injection that previously covered only `mode: research` now also
  covers `mode: script`, and additionally injects orze's own
  site-packages dir so `import orze` works in any subprocess regardless
  of which venv runs it.

- **`call_gemini` defaults to a generally-available model.** The
  previous default `gemini-3.1-pro-preview` and its fallback chain were
  all preview models that return empty responses on GA API keys. Any
  caller omitting the `model` arg (notably the FSM idea_verifier's
  `professor_review` action) silently got an empty response, skipped
  the review, and never shipped a verdict. New fallback chain:
  `caller_model → gemini-2.5-pro → gemini-1.5-pro-latest →
  gemini-3-pro-preview → gemini-3.1-pro-preview`.

## 0.7.5

### Fixed

- **`validate_idea` default-allows unknown keys on `--config`-driven
  scripts.** A `train.py` that takes `--config <yaml>` declares "the
  project owns the schema." Previously `validate_idea` still scanned
  the source for `cfg.get("k")` patterns and rejected any idea whose
  YAML key didn't appear in that set. Helper-function variable renames,
  cross-module delegation, or dict-literal keys missed by the regex
  caused EVERY idea to be marked `status='skipped'` — and the engineer
  auto-fix loop couldn't repair it because only `train.py` was
  editable, never the framework-side validator. New policy:
  * `--config`-using scripts: trust the YAML; log unknown keys at
    DEBUG for diagnosability, do NOT block.
  * Scripts with no `--config` (every knob is a CLI flag): keep the
    strict rejection — it's well-defined there.

## 0.7.4

### Fixed

- **Build: pin `setuptools>=77.0`.** PEP 639 SPDX license strings
  (`license = "LicenseRef-Proprietary"`) need setuptools 77+. On hosts
  with Ubuntu-22.04-default `setuptools 59.6.0`, older setuptools
  silently dropped the entire `[project]` block and built a stub wheel
  named `UNKNOWN-0.0.0` — onboarding users saw no error, just a
  mystery package on `pip list`. The pin forces build isolation to
  upgrade setuptools.

- **`sops.validate_idea` recognizes more config-var names.** The
  AST walker that decides whether a key is a real config lookup only
  matched `cfg` / `config`; scripts that renamed the merged dict
  inside helpers (e.g. `hp = merge(base, idea); hp.get("lr")`) were
  producing spurious "unrecognized args" rejections. The regex now
  also matches `hp`, `hparams`, `params`, `args`, `opts`, `options`,
  `kwargs`, `settings`, `conf`.

- **`orze_pro.__version__` now reads from `importlib.metadata` instead
  of a hardcoded string.** Prevents the same drift class that caused
  the orze 3.4.6 → 3.4.7 auto-upgrade loop: a bump in `pyproject.toml`
  that forgot `__init__.py` would otherwise leave runtime reporting
  the old version.

## 0.7.3

**Requires `orze >= 3.4.6`.**

### Fixed

- **Research / code_evolution agents now resolve `idea_lake.db` from
  `results_dir` (the canonical location).** Previously they looked
  next to `ideas.md`, which produced an empty 0-byte stub at the
  project root and logged `Lake query failed: no such table: ideas`
  on every research cycle.

- **Historical-metric queries guard against malformed `eval_metrics`.**
  All `json_extract(eval_metrics, …)` reads in `research_context.py`
  and `meta_research.py` now pre-filter with `json_valid(eval_metrics)`,
  so a single bad row can no longer raise `malformed JSON` and abort
  context loading.

- **Warning-severity Tier 2 validators no longer spam the log.**
  `filter_queued_ideas` re-validates every queued idea on each
  scheduler poll (~15s); warning-severity hits for the same
  `(idea_id, validator, field)` used to log every pass. A seen-set
  now dedupes per daemon lifetime (error-severity behavior is
  unchanged). Backwards-compatible: callers that don't pass
  `idea_id` still log every time.

- **`orze_pro.agents.*` cycles now `exit 0` on graceful no-op paths.**
  Cycles that legitimately produce no ideas (e.g., cooldown, nothing
  to steer) no longer surface as `OUTCOME_ERROR` / red alerts.

## 0.7.2

**Requires `orze >= 3.4.6`.**

### Changed

- **SOP-derived role runtime behavior.** Role timeout and
  ideas-file-writer flags are now derived from the SOP composition
  rather than from role-name allowlists. `_default_role_timeout(N)
  = 600 + 240 * N` gives roles enough budget to work through each
  skill plus setup/teardown. Explicit `timeout:` and `cooldown:` in
  `orze.yaml` still win. `triggered_by` now bypasses role cooldown.

- **Legacy `bug_fixer` role retired.** Daemon renamed to `watchdog`
  (`orze_pro.agents.watchdog`); SOPs that used to trigger `bug_fixer`
  now target `engineer`. Projects carrying a `roles.bug_fixer:` entry
  receive a one-time migration notice; the pipeline otherwise keeps
  running unchanged.

## 0.7.1

**Requires `orze >= 3.4.2`.**

### Added

- **`orze_pro.agents.pattern_inference.infer_metric_patterns`**:
  LLM-backed fallback for the metric harvester. When regex defaults
  miss, this module calls the Claude CLI (Haiku by default) with a
  head+tail sample of `train_output.log`, asks for Python regex
  patterns that capture the configured primary metric, validates the
  returned JSON (each pattern must compile and have at least one
  capture group), and returns the list. The orze harvester caches
  results keyed by `(train_script, mtime)` so inference runs at most
  once per script per edit.

  Auto-wired by orze's orchestrator when orze-pro is importable.
  Customize the model via `metric_harvest.inference_model` (default
  `"haiku"`) or disable via `metric_harvest.llm_fallback: false` in
  `orze.yaml`.

## 0.7.0

**Requires `orze >= 3.4.0`.**

### Breaking

- **`rules_file:` no longer supported** on any role config. Every
  `mode: claude` role auto-injected by orze-pro
  (`_maybe_bootstrap_professor`, `_maybe_inject_analyst`,
  `_maybe_inject_thinker`) now emits a `skills:` list pointing at
  bundled static SOPs under `orze_pro/sops/`. Projects that still
  have `rules_file:` in their `orze.yaml` must migrate — see the
  orze 3.4.0 changelog.

- **`orze_pro/prompts/*_RULES.md` directory deleted.** The six
  monolithic role prompts (`PROFESSOR_RULES.md`, `THINKER_RULES.md`,
  `DATA_ANALYST_RULES.md`, `ENGINEER_RULES.md`, `BUG_FIXER_RULES.md`,
  `CODE_EVOLUTION_RULES.md`) have been split into 29 composable
  `.skill.md` files under `orze_pro/sops/` and ship via
  `package-data`. There is no `prompts/` dir in the wheel anymore.

- **`orze_pro.agents.professor_bootstrap` module deleted.** Dead after
  the static-SOP refactor. Three generic helpers it hosted
  (`detect_gpu_info`, `_load_env_file` → `load_env_file`,
  `_call_llm_for_bootstrap` → `call_first_available_llm`) are now in
  `orze_pro.agents._llm_utils`. Callers (task_splitter) updated.

### Added

- **29 bundled static SOPs** at `orze_pro/sops/*.skill.md`, one per
  focused reasoning unit (role + job / phase). Every SOP declares
  frontmatter: `id`, `name`, `role`, `order`, `produces`,
  `consumed_by`, `requires`, `trigger`. Roles:
  - professor (9): base, web_search, cross_domain_query, idea_review,
    diversity_enforcement, gap_closure, strategy_review,
    regression_detection, steering.
  - thinker (9): synthesis, base, phase_a–f, axiom_removal.
  - data_analyst (5): base, error_analysis, visualization, insights,
    anomaly_hypotheses.
  - engineer (3): base, implement, fix_bugs.
  - bug_fixer, code_evolution, research (1 each).

- **SOP registry** at `orze_pro.skills.registry`:
  - `discover_skills(project_root, include_bundled=True)` returns
    `SkillMetadata` for both bundled (`tier: static`) and project
    (`tier: dynamic`) SOPs. Dynamic overrides static on id clash.
  - `validate_wiring(skills)` reports dangling `requires`, orphan
    `produces` (no consumer), and missing `overrides` targets.

- **Bundled loader** at `orze_pro.skills.bundled`: resolves
  `@sop:<name>` refs to SOPs shipped in the wheel.

- **Execution receipts** wired into `role_runner`:
  - Before each role launches, capture declared output mtimes.
  - After `check_active_roles` reports completion, write
    `results/_receipts/<role>_cycle<NNN>.json` with which declared
    SOPs showed evidence of execution (outputs changed).

- **Skill-level trigger gates populated per cycle** in
  `run_role_step`. `research_cycles`, `last_activation_cycle`,
  `plateau_patience` are injected into `role_cfg['_trigger_context']`
  before `compose_skills` runs, so `periodic_research_cycles(N)` and
  `on_plateau(N)` evaluate against real state instead of always
  defaulting to False.

- **`orze_pro.cli_sop`**: the handler for `orze sop list|check|status`.
  Aggregates skills (via registry), methods, validators, portfolios.

### Changed

- Auto-injected role configs in `role_runner.py` now reference module-
  level skill manifests (`_PROFESSOR_SKILLS`, `_DATA_ANALYST_SKILLS`,
  `_ENGINEER_SKILLS`, `_THINKER_SKILLS`). No more file-copying from
  the bundle into the project.
- `fsm/plugins/orze_guards.py:role_unhealthy` — checks project-local
  entries in each role's `skills:` list instead of verifying
  `rules_file` paths.
- `fsm/plugins/idea_verifier.py:_load_professor_prompt` — composes the
  bundled professor SOPs via `compose_skills` so the verifier reasons
  over the exact prompt the live professor role sees.
- `agents/meta_research.py` — locates RESEARCH_RULES.md by project
  path (+ fallback: first `.md` skill found), not via `rules_file:`
  lookup.
- `agents/task_splitter.py` — per-subtask `orze.yaml` template emits
  `skills:` lists; `create_task_folders` no longer invokes
  `bootstrap_professor`.

### Removed

- `orze_pro/sops/_smoke.skill.md` test fixture. Tests exercise the
  bundled discovery path against `thinker_base` instead.
