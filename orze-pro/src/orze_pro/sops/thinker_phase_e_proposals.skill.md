---
id: sop-th-phase-e
name: thinker_phase_e_proposals
role: thinker
order: 70
produces:
  - results/_paradigm_report.txt
  - results/_retrospection.txt
consumed_by: [professor]
requires: []
trigger: always
---

## Phase E: Concrete Proposals

Produce **exactly 3 paradigm-shift proposals**. Each must be
**fundamentally different** from the current approach AND from each
other.

For each proposal:

### Proposal N: [Name]

- **Hypothesis:** One sentence stating what you believe and why.
- **Formulation:** How the task is framed (classification? regression?
  temporal process?).
- **Loss function:** Mathematical description or pseudocode. Be specific
  — not "temporal loss" but the actual formula.
- **Training procedure:** What changes from current setup? Batch
  structure, sampling, epochs, optimizer.
- **What the model learns differently:** Under this proposal, what does
  the model learn that it can't learn under the current approach?
- **Expected outcome:** What metric improvement do you expect and why?
- **Implementation complexity:** easy (loss-function change only),
  medium (new dataset/training loop), hard (new model architecture).

**Rank proposals by: expected impact × (1 / implementation complexity).**

## Output

Write your complete analysis to `{results_dir}/_paradigm_report.txt`
using this format:

```
## Paradigm Report — {timestamp}
Trigger: {trigger_reason}

### Problem Reformulations
[Phase A output]

### Root Cause
[Phase B one-sentence synthesis — this is the CRITICAL line]

### Constraints
[Phase C output]

### Cross-Domain Analogies
[Phase D output]

### Proposal 1: [Name]
[Full proposal]

### Proposal 2: [Name]
[Full proposal]

### Proposal 3: [Name]
[Full proposal]

### Recommended Action
[Which proposal to try first and why]
```

**Also append a summary** (the "Recommended Action" section + top
proposal) to `{results_dir}/_retrospection.txt` so the research agent
sees it immediately. Prefix with
`\n\n## PARADIGM SHIFT — Thinker Report ({timestamp})\n` so it stands
out in the retrospection feed.
