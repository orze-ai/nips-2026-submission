---
id: sop-pr-regression
name: professor_regression_detection
role: professor
order: 50
produces:
  - results/_retrospection.txt
consumed_by: [research, engineer]
requires: []
trigger: always
---

## Regression Detection — Code Inspection Protocol

**Do not treat the training script as a black box.** When experiments
regress, the bug may be in the code, not the config.

Every cycle, check whether recent experiments regressed vs the
baseline:

1. Read the leaderboard. If the last N experiments using a new feature
   ALL score worse than the best model without that feature, the
   feature implementation is likely buggy.
2. **After 3+ consecutive regressions from a new feature: STOP
   generating config variants. Read the training script (`train.py`)
   and inspect the feature's implementation.** Look for logic errors:
   wrong indexing, operations applied in wrong order, data thrown
   away by downstream steps.
3. If you find a bug, fix it directly in `train.py`. Then update
   `_retrospection.txt` to note the fix and instruct the research
   agent to re-run the experiments.
4. If you cannot identify the bug, write a detailed diagnosis to
   `results/_retrospection.txt` explaining the regression pattern and
   what you checked.

**The most common pattern**: a new augmentation or data feature is
implemented correctly in isolation, but a downstream step
(truncation, padding, normalization) undoes its effect. Always trace
the data flow end-to-end: load → augment → truncate/pad → model.
