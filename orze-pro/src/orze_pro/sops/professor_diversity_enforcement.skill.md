---
id: sop-pr-diversity
name: professor_diversity_enforcement
role: professor
order: 25
produces:
  - RESEARCH_RULES.md
consumed_by: [research]
requires: []
trigger: always
---

## Job 1b: Diversity Enforcement (EVERY CYCLE)

**The #1 failure mode of automated research is collapsing into
micro-sweeps of the champion config.** You MUST actively prevent this.

Every cycle, check the experiment distribution in the leaderboard:

1. **Count unique backbones / architectures** in the last 50
   experiments. If >80% use the same backbone, the search has
   collapsed. Add to `RESEARCH_RULES.md`: "At least 1 in 5 ideas must
   use a different backbone / architecture."
2. **Count unique technique categories** (data augmentation, loss
   function, model architecture, data curation, regularization,
   training schedule). If >80% of recent ideas are in the same
   category, force diversity.
3. **Check for champion-grinding** — if the last 10+ ideas are all
   minor variants of the current best (same backbone, same model, just
   tweaking one hyperparameter), SKIP future variants and redirect
   toward unexplored directions.

**Enforce in `RESEARCH_RULES.md` using an explicit diversity budget:**

```
## Diversity Budget (MANDATORY)
Each batch of 5 ideas MUST include:
- At least 1 idea exploring a different backbone or architecture
- At least 1 idea testing a technique NOT used in the current champion
- At most 2 ideas that are minor variants of the current champion
```

**Why this matters:** a pipeline that runs 2,000 experiments on one
backbone with one model type has explored 1% of the design space 2,000
times. Broad exploration finds the right region; deep exploitation
optimizes within it. You need BOTH.

When the search collapses:

- **WebSearch** for alternative techniques in the domain.
- Add unexplored directions to `RESEARCH_RULES.md` as MANDATORY.
- Rotate the "WHAT TO PROPOSE" priorities — don't let one direction
  monopolize.
