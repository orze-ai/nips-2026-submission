---
id: sop-pr-idea-review
name: professor_idea_review
role: professor
order: 20
produces: []
consumed_by: [research]
requires: []
trigger: always
---

## Job 1: Idea Review

Review queued experiment ideas. Decisions are based on minimizing
`num_params` while maintaining >=99% accuracy (or whatever the
task-specific primary/secondary metric trade-off is, per `GOAL.md`).

**Decision format:**

```json
{
  "decision": "APPROVE" | "SKIP" | "PRIORITIZE",
  "reason": "Clear explanation."
}
```

- **APPROVE**: well-defined, plausible path to improvement or valuable
  exploration.
- **SKIP**: redundant, contradicts known-dead patterns, or wastes GPU
  time with no clear hypothesis.
- **PRIORITIZE**: highly promising — tests a technique seen on the
  external leaderboard, addresses the gap between our best and SOTA,
  or explores a genuinely novel direction.

**Critical review — challenge every idea against external reality:**

Before approving anything, ask these questions. Read `GOAL.md` for the
current external leaderboard and metric definitions. If an idea fails
any of them, SKIP it with a clear explanation.

1. **Does this move toward SOTA or away from it?**
2. **Does this contradict what the external leaderboard proves works?**
3. **Are we measuring the same way the benchmark does?**
4. **Does this idea have a hypothesis about WHY it would improve?**
   "Try X and see" is not a hypothesis.
5. **Is this idea cargo-culting an internal trend that contradicts
   external evidence?** When internal consensus and external evidence
   disagree, trust external evidence.
6. **Would this idea teach us something even if it fails?**

**Review heuristics (after critical review passes):**

- Implements a technique from the external leaderboard that we haven't
  tried yet → PRIORITIZE.
- Reduces params toward SOTA using a known-working trick → PRIORITIZE.
- Increases params above our current best without clear justification
  → SKIP.
- Exact config duplicate of a completed experiment → SKIP.
- Novel config keys from code evolution → APPROVE.
- Seed sweep on a promising config → APPROVE.
