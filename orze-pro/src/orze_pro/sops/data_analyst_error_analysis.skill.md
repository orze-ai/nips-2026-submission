---
id: sop-da-error-analysis
name: data_analyst_error_analysis
role: data_analyst
order: 20
produces:
  - results/_error_analysis.md
consumed_by: [professor, thinker]
requires: []
trigger: always
---

## Job 2: Error Analysis

After experiments complete, analyze what the model gets wrong.

1. **Load test predictions** — find `test_predictions.npz` from the
   best model(s) in `results/`.
2. **Load ground truth** — find and read the test labels (from
   `solution.csv`, test-set .pt files, or wherever labels live).
3. **Identify errors** — find samples where the model is most wrong
   (high-confidence incorrect predictions).
4. **Categorize errors** — group failures by available metadata:
   - If group/category columns exist in the data, compute per-group
     metrics.
   - If temporal metadata exists, analyze whether errors correlate
     with timing.
   - If the data has multiple splits or conditions, compare performance
     across them.
5. **Write findings to `results/_error_analysis.md`** — a structured
   report with:
   - Overall metrics breakdown
   - Per-group / per-category performance
   - Worst-performing segments with example IDs
   - Patterns in the failures
