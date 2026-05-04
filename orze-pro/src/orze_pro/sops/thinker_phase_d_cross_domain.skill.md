---
id: sop-th-phase-d
name: thinker_phase_d_cross_domain
role: thinker
order: 60
produces: []
consumed_by: [thinker]
requires: []
trigger: always
---

## Phase D: Cross-Domain Analogies

**What other fields predict events approaching a critical point?**

For each analogy:

1. Name the field and problem (e.g. "earthquake prediction", "medical
   deterioration", "financial crash forecasting").
2. How do they formulate it? What loss function or model structure do
   they use?
3. What specific technique could transfer to our problem?

Look for analogies where:

- The event is rare but preceded by increasing signals
- Temporal structure matters more than point-in-time features
- Labels are sparse but temporal dynamics are rich
- Small datasets are the norm (not millions of examples)

**Do not rely on memory of pop-ML analogies.** If you haven't
reconstructed the formulation from first principles, the analogy is
superficial and will yield a shallow proposal in Phase E.
