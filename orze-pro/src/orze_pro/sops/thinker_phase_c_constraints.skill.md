---
id: sop-th-phase-c
name: thinker_phase_c_constraints
role: thinker
order: 50
produces: []
consumed_by: [thinker]
requires: []
trigger: always
---

## Phase C: Constraint-Based Invention

Given what you learned in Phases A and B, specify the **constraints a
good solution must satisfy**.

Example constraints (adapt to the specific problem):

- Must capture temporal evolution, not just frame-level patterns
- Must handle class imbalance without artificial reweighting
- Must provide dense supervision even when labels are sparse
- Must regularize against overfitting on small datasets
- Must allow the backbone to learn task-specific features (end-to-end)

Now **derive a method that satisfies ALL constraints simultaneously.**

Don't search for an existing method — CONSTRUCT one from the
constraints. If it happens to match an existing technique, note that,
but the reasoning should flow from constraints to solution, not from
memory to solution.

The constructed method is a candidate for Phase E's ranked proposals.
