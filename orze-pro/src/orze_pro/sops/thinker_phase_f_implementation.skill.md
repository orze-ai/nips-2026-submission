---
id: sop-th-phase-f
name: thinker_phase_f_implementation
role: thinker
order: 80
produces:
  - RESEARCH_RULES.md
consumed_by: [professor]
requires: []
trigger: always
---

## Phase F: Implementation (DO THIS BEFORE WRITING THE PARADIGM REPORT)

**Write code FIRST, then write the report. If you run out of time, the
code matters more than the report.**

Reports alone don't change anything — the professor can't implement
code changes, and the research agent only tweaks configs of existing
scripts. If you don't write the code, your paradigm shift will be
ignored and the system will fall back to the old approach.

**After Phase B (Root Cause), immediately write code for the most
obvious fix.** Don't wait until you've completed all analysis phases.
A working training script is worth more than five beautifully written
proposals.

For your best proposal:

1. **Read the existing training script** (e.g. `train.py`,
   `train_e2e.py`) to understand codebase patterns, imports, data
   loading, and evaluation.
2. **Write a new training script** implementing your proposed
   formulation. Name it descriptively. Reuse existing infrastructure —
   only change the loss function and training loop. Keep it MINIMAL
   (under 200 lines of new code).
3. **Verify it imports correctly**: run
   `python3 -c "import your_new_script"` via Bash.
4. **IMMEDIATELY append launch instructions to `RESEARCH_RULES.md`**
   under a `## THINKER PROPOSALS` section. Include the exact bash
   command to launch it. This is how the professor discovers your
   work — if you skip this, your code will be ignored.

Only after the code is written and `RESEARCH_RULES.md` is updated
should you return to finish Phases C, D, E and write the paradigm
report. If you run out of time, the code + `RESEARCH_RULES.md` update
are what matter; the report is optional.
