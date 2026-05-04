---
id: sop-da-anomaly
name: anomaly_driven_hypotheses
role: data_analyst
order: 50
produces:
  - results/_failure_hypotheses.md
consumed_by: [research, thinker]
requires: [sop-da-error-analysis]
trigger: periodic_research_cycles(5)
---

# Data Analyst: Failure-Driven Hypotheses

**Trigger reason:** {trigger_reason}

After the main error analysis completes, look at the **worst 1% of samples
by per-sample loss** from the most recent strong model. The goal is not
another summary — the goal is to convert failure structure into testable
hypotheses the research agent can act on.

## How to produce hypotheses

1. **Cluster the worst samples by any available metadata** — time-to-event,
   time-of-day, scene type, subject count, whatever is in the data. Prefer
   clusters the current loss function cannot detect.

2. **For each cluster (2–5 total, no more), write exactly this block**:

   ```
   ## Cluster <N>: <short name>
   Size: <count> samples (<pct>% of failures)
   Pattern: <what the samples share — be specific, avoid "some", "many">
   Hypothesis: The model fails on <X> because <Y>.
   Fix: <one concrete action a research agent could propose>
   ```

3. **Prioritize structural failures over noise.** A cluster of label
   errors is useful once (propose relabeling) but not every cycle.
   Clusters where the model can't represent the pattern (e.g. "needs
   temporal context we don't feed it") are the valuable ones.

## Stop rules

- **Do not write more than 5 clusters.** Generating a long list of weak
  hypotheses crowds out the strong ones in the research agent's context.
- **Do not propose hyperparameter tweaks here.** This SOP produces
  *structural* hypotheses. "Try lr=5e-6" is not a failure hypothesis —
  it's an experiment, and the research agent generates those.

## Output

Write the hypotheses to `results/_failure_hypotheses.md`. The research
agent reads this file every cycle and MUST address the top cluster
before generating free-form ideas.
