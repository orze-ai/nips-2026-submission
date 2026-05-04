---
id: sop-th-phase-b
name: thinker_phase_b_root_cause
role: thinker
order: 30
produces: []
consumed_by: [thinker]
requires: []
trigger: always
---

## Phase B: Root Cause Analysis

Look at the **pattern** of failures, not individual results.

Answer these questions:

1. **What does the model learn vs what does it NEED to learn?**
   (e.g. does it learn appearance patterns when it needs to learn
   temporal dynamics?)
2. **Why does it overfit?** Not "learning rate too high" — deeper:
   what structural property of the training setup causes overfitting?
3. **What assumption in the current approach contradicts the data?**
   (e.g. treating temporally correlated frames as independent samples)
4. **If the current approach were correct, what would we expect to see?
   Do we see it?** (e.g. if frame-level binary labels were sufficient,
   we'd expect later epochs to improve. If they don't, the labels are
   the problem.)

**Synthesize into one sentence:**

> The current approach fails because ___.

This sentence is the anchor for Phase C and Phase E. If you can't
write it crisply, your root-cause analysis is still incomplete — keep
digging before moving on.
