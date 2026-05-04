---
id: sop-th-phase-a
name: thinker_phase_a_reformulation
role: thinker
order: 20
produces: []
consumed_by: [thinker]
requires: []
trigger: always
---

## Phase A: Problem Reformulation

**This is the most important phase.** The current approach embeds
assumptions about how to formulate the problem. Those assumptions may
be wrong.

Write out **5 fundamentally different formulations** of the task.

For each formulation:

1. **Name it** (e.g. "temporal risk regression", "survival analysis",
   "anomaly detection").
2. **Define the loss function** that naturally follows from this
   formulation.
3. **Describe what the model learns** under this formulation vs the
   current approach.
4. **Identify what information** this formulation captures that the
   current one misses.

Example: if the current approach is binary classification
("collision or not"), alternatives might include:

- Regression on time-to-event (continuous target, naturally temporal)
- Survival analysis (hazard function, increasing risk over time)
- Sequential decision process (each frame is a decision point)
- Contrastive learning (distinguish increasing-risk sequences from
  decreasing-risk)
- Self-supervised prediction (predict future frames, detect anomalous
  futures)

**Pick the 2 most promising formulations** and explain why they better
match the problem's structure than the current approach. These two are
the seeds Phase E will turn into concrete proposals.
