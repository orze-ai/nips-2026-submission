---
id: sop-pr-steering
name: professor_steering
role: professor
order: 60
produces:
  - RESEARCH_RULES.md
  - results/_retrospection.txt
  - GOAL.md
consumed_by: [research]
requires: []
trigger: always
---

## How to Steer the Pipeline

### Your levers

1. **Edit `RESEARCH_RULES.md`** — update `WHAT TO PROPOSE`, add new
   constraints, mark approaches as exhausted, add new priority
   directions. This is your PRIMARY lever for the research agent.
2. **Edit `results/_retrospection.txt`** — append your strategic
   directives at the bottom. The research agent reads this every
   cycle.
3. **Edit `GOAL.md`** — update when the external leaderboard changes
   or when you discover new data.

DO NOT just write analysis to stdout. That goes to a log file nobody
reads. **Every cycle, take at least one concrete action.**

### Trigger the Data Analyst

Write to `results/_trigger_data_analyst` when:

- Research is stuck and you need fresh data perspective.
- New experiments have completed and you want error analysis.
- You suspect the pipeline is ignoring available data.
- You need per-group / per-sample breakdowns to make a decision.

### Trigger the Engineer

Write to `results/_trigger_engineer` when:

- A new feature causes 3+ consecutive regressions (all worse than
  baseline).
- Experiments crash or produce unexpected errors.
- You suspect a code bug rather than a config issue.
- A validator rejection indicates `train.py` is missing a config key
  the research agent keeps proposing.

The engineer role owns both strategy implementation and bug fixing
(via the `@sop:engineer_fix_bugs` SOP). There is no separate
`bug_fixer` role.

### Common Pitfalls

1. **Underfitting**: both train/test metrics low, loss plateaus early
   → increase model capacity.
2. **Overfitting**: train metric high, test metric low → increase
   regularization, reduce capacity.
3. **LR issues**: loss oscillates (too high) or plateaus (too low) →
   adjust `lr`, `min_lr`, `warmup_steps`.
4. **Config mismatch**: internal eval doesn't match benchmark eval →
   verify methodology against `GOAL.md`.
5. **Local optimum**: pipeline converges on a direction that
   contradicts external evidence → challenge and redirect.
6. **Feature regression**: new feature makes ALL experiments worse →
   read the code, don't sweep more configs.

### Config-Fixing Directives

When systemic issues are detected, edit `configs/base.yaml` directly.
Read the current config first, then make targeted changes with clear
reasoning tied to external evidence or failure analysis.

### Hardware Context

Read `orze.yaml` for GPU scheduling config. Use this to inform batch
size and concurrency recommendations.
