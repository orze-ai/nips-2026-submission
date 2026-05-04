---
id: sop-pr-cross-domain
name: cross_domain_structural_query
role: professor
order: 15
produces:
  - GOAL.md
  - results/_cross_domain_log.md
consumed_by: [research, thinker]
requires: []
trigger: always
---

# Professor: Cross-Domain Structural Query (MANDATORY)

**Trigger reason:** {trigger_reason}

Every cycle, issue **one** WebSearch query against a field *outside* the
task domain that shares the same structural problem. This prevents
the pipeline from only ever searching within its own niche and missing
techniques that map in from other disciplines.

## The rule

Pick the analogy from **problem structure**, not topic:

| If the task has this structure                             | Search these fields                                    |
|------------------------------------------------------------|--------------------------------------------------------|
| rare event preceded by increasing signal                   | earthquake early warning, sepsis prediction, financial-crash forecasting |
| sparse labels + rich temporal dynamics                     | survival analysis, hazard models, reliability engineering |
| ordinal targets / time-to-event                            | ordinal regression in medical imaging, credit risk     |
| multi-view with one dominant view                          | audio–visual fusion, sensor fusion in robotics         |
| small labeled set + large unlabeled pool                   | semi-supervised learning in genomics, weak supervision in NLP |

Rotate queries across cycles — append to `results/_cross_domain_log.md`
so the next cycle doesn't repeat. If the log already has a given analogy
this month, skip it and pick a fresh one.

## Write-up

When the query returns something concrete (a method, a loss formulation,
a training trick), add a bullet to `GOAL.md` under the section:

```
## Prior Art — Cross-Domain Analogies
- [YYYY-MM-DD] <analogy field>: <technique>, <source URL>. Why it maps:
  <one sentence on the structural correspondence to our task>.
```

If the query returns nothing useful, **record the negative result** in
`results/_cross_domain_log.md` so later cycles don't re-query the same
dead end.

## Non-goals

- Do NOT replace the within-domain Job-0 web search. This SOP adds ONE
  query on top of it.
- Do NOT propose experiments here. You're an external-knowledge scout,
  not an idea generator.
