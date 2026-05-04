---
id: sop-da-insights
name: data_analyst_insights
role: data_analyst
order: 40
produces:
  - results/_analyst_insights.md
consumed_by: [professor, research]
requires: []
trigger: always
---

## Job 4: Actionable Insights

After completing audit + error analysis + visualization, write a
synthesized summary to `results/_analyst_insights.md`.

1. **What data is unused** — columns, metadata, annotations the
   pipeline ignores.
2. **Where the model fails** — specific segments, conditions, or sample
   types.
3. **What the errors suggest** — patterns that point to specific fixes
   (e.g. "model fails on short TTC samples" suggests the temporal
   window matters).
4. **Priority recommendations** — ranked list of what to try next,
   grounded in data.

## Coordination

The professor reads this file every cycle. Write clearly and concisely.
**Lead with the most actionable finding.** If you discover unused data
that could improve the model, say so explicitly at the top.
