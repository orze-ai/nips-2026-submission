---
id: sop-pr-strategy
name: professor_strategy_review
role: professor
order: 40
produces: []
consumed_by: []
requires: []
trigger: always
---

## Job 2: Strategy Review

Continuously monitor research strategy against the **external
competitive landscape**.

### Key questions every cycle

1. **Where are we vs SOTA?** Read `GOAL.md` for the external
   leaderboard. Compare our best against top entries on both primary
   and secondary metrics.
2. **What techniques does SOTA use that we haven't implemented?** List
   specific techniques from external top entries that our pipeline
   hasn't explored.
3. **Is the research agent exploring the right region of design
   space?** Compare our experiment distribution against where external
   winners cluster. If there's a mismatch, redirect via
   `RESEARCH_RULES.md`.
4. **Are we wasting GPU time on dead directions?** If no external top
   entry uses technique X but we keep generating ideas with it, cut it.
5. **Is our internal best actually competitive externally?** Does our
   evaluation methodology match the benchmark's? A result that looks
   great internally but uses different eval criteria is not a real
   result.
6. **What's the failure rate telling us?** If most experiments fail,
   is that expected for this domain, or is there a systematic bug?

### Status thresholds

Derive thresholds from `GOAL.md` each cycle:

- **CRITICAL**: our best is far from competitive on primary metric, or
  the pipeline is producing mostly failures with no improving trend.
- **WARN**: our best is competitive but not improving, stuck on a
  plateau, or internal trends contradict external evidence.
- **OK**: our best is competitive and improving, or we've
  matched/beaten SOTA.

### Output format

```json
{
  "status": "OK" | "WARN" | "CRITICAL",
  "external_sota": "Summary of top external result",
  "our_best": "Summary of our top result",
  "gap_analysis": "Specific techniques and metrics where SOTA leads",
  "action": {
    "type": "NONE" | "EDIT_CONFIG" | "EDIT_RULES" | "TRIGGER_CODE_EVOLUTION",
    "details": "Specific instructions"
  }
}
```
