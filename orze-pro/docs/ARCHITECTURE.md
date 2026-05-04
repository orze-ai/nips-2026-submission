# Orze-Pro — Architecture & Design Decisions

## Role System

### Single-Instance Guarantee
Each role (research, professor, code_evolution, bug_fixer, FSM) runs on at most one machine at a time, enforced by filesystem lock directories (`_{role_name}_lock/`). This is by design — roles read shared state and make decisions that must be serialized.

### Why Not Distributed Roles?
- The research agent generates ideas based on the full leaderboard — running two would produce duplicate ideas
- The professor edits configs — concurrent edits corrupt files
- The FSM tracks state machines — split-brain would cause contradictory transitions
- At current workloads (1 idea per 30s), a single research agent saturates the idea queue for ~50 GPUs

### Role Dependencies
```
GOAL.md → Professor (strategy) → Research Agent (ideas) → Training (GPU)
                                                              ↓
                                    Bug Fixer ← failures ← results/
                                    Code Evolution ← plateau ← FSM
```

## Coordination Layer

### Filesystem as Database
Orze uses the filesystem as its coordination layer:

| Abstraction | Implementation | Atomicity |
|-------------|---------------|-----------|
| Experiment claim | `mkdir(results/{id}/)` | POSIX atomic |
| Role lock | `mkdir(_{role}_lock/)` | POSIX atomic |
| Status broadcast | `atomic_write(status.json)` | write-to-temp + rename |
| Idea queue | `idea_lake.db` (SQLite) | SQLite transactions |
| Idea inbox | `ideas.md` (append + consume) | File lock with timeout |
| Config | YAML files | No locking (rare writes) |

### Why Filesystem Over a Database?
1. **Zero infrastructure** — no Redis/Postgres to install, configure, monitor
2. **Works on any cluster** — NFS, Lustre, EFS all support POSIX semantics
3. **Debuggable** — `cat results/idea-001/metrics.json` beats `SELECT * FROM experiments`
4. **Resilient** — filesystem survives process crashes; no connection pools to manage
5. **Write frequency is low** — 1 idea per 30s, 1 experiment per hours. No need for a database.

### Trade-offs Accepted
- NFS attribute cache can cause 3-60s stale reads (acceptable for experiment coordination)
- SQLite on NFS is not bulletproof (mitigated by DELETE journal mode + retry loop)
- No real-time event system (polling at 30s intervals is sufficient)
- Config reads are not locked (professor edits are rare, impact is 1 bad experiment)

## Role Composition via SOPs (v0.7.0+)

### Problem
Monolithic role prompts (the old `prompts/*_RULES.md` files) were a
single unit: 346 lines for the professor, 177 for the thinker, etc.
Adding a new behavior meant editing a large file; disabling one
meant deleting a section; projects with task-specific needs had to
fork or LLM-bootstrap their own copy.

### Solution
Role behavior is decomposed into **static SOPs** — one `.skill.md` per
focused reasoning unit — bundled in the wheel at `orze_pro/sops/`.
Each SOP declares frontmatter metadata: `id`, `role`, `order`,
`produces`, `consumed_by`, `requires`, `trigger`.

Roles compose a prompt from a list of SOP references in `orze.yaml`:

```yaml
roles:
  professor:
    mode: claude
    skills:
      - "@sop:professor_base"
      - "@sop:professor_web_search"
      - "@sop:professor_idea_review"
      # ...
```

Project-specific behavior is added as **dynamic SOPs** under
`<project>/skills/*.skill.md`. A dynamic SOP with the same `id` as
a bundled one overrides it; otherwise both compose together ordered
by frontmatter `order`.

### Task-specific benchmarks, thresholds, pitfalls
These used to require LLM-generated PROFESSOR_RULES.md content. They
now live in `RESEARCH_RULES.md` (project-owned, free-form) referenced
by the research role's skills list, or in a project-authored dynamic
professor SOP.

### Observability
- `orze sop list` — enumerate all registered SOPs (skill / method /
  validator / portfolio) with tier and role.
- `orze sop check` — validate wiring (dangling requires, orphan
  produces, missing overrides).
- `orze sop status` — per-SOP execution evidence from receipts
  recorded in `results/_receipts/`.

## Task Splitter (v0.3.6+)

### Problem
Users write a single GOAL.md with multiple tasks (e.g., "detect, track, and segment"). Each task needs its own orze instance with separate metrics, leaderboard, research agent, and professor.

### Solution
LLM-based task detection:
1. Read GOAL.md
2. Call LLM to identify distinct tasks, metrics, dependencies, compute weights
3. Propose GPU allocation proportional to compute weight
4. Prompt user for confirmation (Y/n/edit)
5. Create per-task subfolders with all scaffold files
6. Generate start_all.sh / stop_all.sh helper scripts

### Why LLM for Detection?
Regex-based parsing is too brittle. Real goals look like:
> "We need to detect objects in 2D and 3D, track them across frames, and segment the road surface"

That's 4 tasks in one sentence. The LLM understands:
- Task boundaries (detection vs tracking vs segmentation)
- Appropriate metrics (NDS vs mAP vs AMOTA vs mIoU)
- Dependencies (tracking depends on detection)
- Relative compute requirements

### What's NOT LLM-Generated
- `train.py` — stub template with orze contract, user implements training logic
- `orze.yaml` — deterministic template from task metadata
- GPU allocation — heuristic proportional to compute_weight

## FSM Primary Metric (v0.3.2+)

### Problem
FSM guards were hardcoded to read `"accuracy"` from metrics.json. Projects using NDS, mAP, AMOTA, mIoU would never detect plateaus.

### Fix
`_get_primary_metric_name()` reads `report.primary_metric` from orze.yaml at runtime. Falls back to `"accuracy"` for backwards compatibility.

## Multi-Machine Design

See [DISTRIBUTED.md](../docs/DISTRIBUTED.md) in the orze repo for the full multi-machine architecture, scaling limits, and known limitations.

### TL;DR
- 1 commander + unlimited workers on shared filesystem
- Atomic mkdir for experiment claiming (never double-claims)
- Filesystem locks for role serialization
- Reliable up to ~20 machines, likely works up to ~50
- Not designed for 100+ nodes or multi-orchestrator setups
