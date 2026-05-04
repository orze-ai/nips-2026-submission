---
id: sop-da-visualization
name: data_analyst_visualization
role: data_analyst
order: 30
produces:
  - results/_analysis
consumed_by: [professor]
requires: []
trigger: always
---

## Job 3: Visualization

Generate analysis artifacts the professor can act on:

1. **Score distributions** — histogram of model confidence for correct
   vs incorrect predictions.
2. **Per-group performance** — bar chart of metrics by any available
   grouping variable.
3. **Error breakdown** — frequency of different error types.
4. **Confusion patterns** — what the model confuses (false positives vs
   false negatives).

Save plots to `results/_analysis/` as PNG files. Use matplotlib — it's
always available.
