---
id: sop-th-axiom-removal
name: axiom_removal
role: thinker
order: 40
produces:
  - results/_axiom_experiments.md
consumed_by: [professor]
requires: []
trigger: on_plateau(20)
---

# Thinker: Axiom Removal

**Trigger reason:** {trigger_reason}

Axiom removal is a structured creativity move. You list the load-bearing
assumptions of the **current pipeline** (not the foundations of deep
learning), temporarily remove one, and see what's required to build a
working method without it. The interesting ideas live in that
negative space.

## Scope guardrail

The axiom to remove must be a *component of the current pipeline*, not
an axiom of the field. In scope: BCE loss, the classification head,
fine-tuning the backbone, cosine LR schedule, frame-level labels,
augmentation pipeline. Out of scope: backpropagation, SGD, transformers
as an architecture class. If you can't imagine replacing the component
in a single training script, it's out of scope.

## Procedure

1. **Read `RESEARCH_RULES.md` "Active Tracks"** to identify the 5 most
   load-bearing components of the current best approach.

2. **List them** with one line each explaining *why* the component is
   there:

   ```
   A1. <component>  — present because <reason>
   A2. ...
   ```

3. **Pick one at random** (not the one you already have a pet idea for;
   randomness is the point).

4. **Write the replacement design.** Under "if A_k is removed":
   - What loss / signal / structure fills the gap?
   - What does the new method look like end-to-end?
   - What's the simplest training script that implements it (pseudocode)?

5. **Red-team the design in one paragraph**: why might this fail?
   What would kill it first — overfitting, class imbalance, optimizer
   mismatch, compute cost?

6. **Score the idea** against the current champion using the available
   fast-eval signal: if you could only run ONE experiment to sanity-check
   the new design, what would it be and what result would make you
   abandon it?

Repeat for a second axiom to give the professor two proposals to pick
from (not five — two well-reasoned proposals beat five shallow ones).

## Output

Append to `results/_axiom_experiments.md`:

```
## Axiom experiment — {timestamp}
Axiom removed: A_k = <component>
Replacement design: <sketch>
Sanity experiment: <one command or idea-yaml>
Kill criterion: <metric + threshold>
```

The professor reads this file and decides whether to promote the
sanity experiment into `ideas.md` or an idea-portfolio.

## Rules

- Two axiom removals per activation. Not more, not less.
- Do NOT propose hyperparameter sweeps (that's the research agent).
- Do NOT propose anything marked DEAD in `RESEARCH_RULES.md` unless
  your reformulation is specifically designed to bypass the reason
  it died last time.
