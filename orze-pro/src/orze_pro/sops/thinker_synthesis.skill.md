---
id: sop-th-synthesis
name: multi_source_synthesis
role: thinker
order: 5
produces:
  - results/_paradigm_report.txt
consumed_by: [professor, research]
requires:
  - sop-da-anomaly
  - sop-pr-cross-domain
trigger: always
---

# Thinker: Multi-Source Synthesis (run FIRST)

**Trigger reason:** {trigger_reason}

Before you do your own reasoning, **read every creativity-stream output
that exists** and look for *structural resonances* — two independent
inputs that point at the same underlying insight. Resonances are
higher-signal than anything a single stream produces alone.

## Inputs to scan

1. `results/_failure_hypotheses.md` — anomaly-driven hypotheses from
   the data analyst (what the model is structurally bad at)
2. `results/_cross_domain_log.md` and `GOAL.md` "Prior Art — Cross-Domain
   Analogies" — techniques the professor pulled from other fields
3. `results/_axiom_experiments.md` — axiom-removal designs from prior
   thinker activations
4. Any `extreme_constraints` portfolio results under `results/_portfolios/`

## What to look for

- **Convergence:** a failure cluster in (1) that names the same
  structural weakness a cross-domain technique in (2) addresses. That
  pair is a lead — write it up first.
- **Orthogonality:** two independent wins (e.g. LLRD 0.95 and 9000spe
  from earlier cycles) that haven't been combined — propose the
  combination, *if* the underlying mechanisms don't contradict.
- **Contradictions:** a cross-domain technique in (2) that assumes
  something the axiom-removal design in (3) already rejected. Flag
  this — the pipeline may be spinning on a dead abstraction.

## Output

**First** write the synthesis to `results/_paradigm_report.txt` with
this structure:

```
## Synthesis — {timestamp}
### Resonances found
- <lead 1>: sources (<X>, <Y>). Underlying structural insight: <one sentence>.
### Orthogonal combinations worth testing
- <combo>: why additive, kill criterion.
### Contradictions flagged for professor
- <pair>: why they disagree, which one to trust.
```

**Then** proceed to your normal Phase A–F reasoning. Use the synthesis
as Phase 0 context so your downstream reasoning is anchored in what the
rest of the pipeline has already found — instead of re-deriving it from
scratch.

## Non-goals

- Do NOT restate the contents of the input files. Synthesis means
  *combining* them into higher-order claims, not concatenating them.
- Do NOT generate ideas directly. Your output is fuel for the
  professor's idea review, not a substitute for it.
