---
id: sop-eng-implement
name: engineer_implement
role: engineer
order: 20
produces: []
consumed_by: []
requires: []
trigger: always
---

## Duty 1: Implement (when trigger says "IMPLEMENTATION TASK")

A portfolio tried to generate experiments but the training script
doesn't support the required config. You must port the method.

### Step 1: Read the task

1. The trigger reason tells you WHAT to implement and WHERE.
2. Read the method spec: `results/_methods/<method>.yaml` — loss
   formulas, hyperparameters, source path.
3. Read the source code referenced in the method spec — understand the
   exact implementation.

### Step 2: Read the target training script

1. Read the `train_*.py` file that needs the feature.
2. Find where the loss function is defined.
3. Find the argparse section.
4. Understand existing code patterns.

### Step 3: Implement

1. Add the new loss function following the pattern in the source code.
2. Add argparse flags for the new config keys.
3. Ensure the script still accepts orze standard args: `--idea-id`,
   `--results-dir`, `--ideas-md`, `--config`.
4. Follow the existing code style.

### Step 4: Verify

1. Run `python3 train_script.py --help` — new flags should appear.
2. Syntax check:
   `python3 -c "import ast; ast.parse(open('train_script.py').read())"`
3. The portfolio will auto-generate ideas on the next orze iteration.


## Duty 3: Queue prescribed work (when trigger has PRIORITY sections)

The trigger file (typically `_trigger_engineer*` or its forwarded copy)
contains one or more `### PRIORITY <N>` blocks. Each block prescribes
concrete experiment work — recipe, target competition, hyperparameters,
expected outcome. Your job is to translate those blocks into runnable idea
entries appended to `ideas.md`. **Generic rule: this works for any future
prof-style trigger with PRIORITY sections — do not special-case any one
cycle.**

This duty is the ONLY case where the engineer is permitted to write to
`ideas.md`. The Duty 2 prohibition ("DO NOT modify ideas.md") does not
apply when PRIORITY blocks are present.

### Step 1: Parse PRIORITY blocks

1. Read the trigger file pointed to by `{trigger_reason}` (or the most
   recent unresolved `_trigger_engineer*` if the reason is opaque).
2. Find every section whose heading matches `PRIORITY <N>` (case
   insensitive; matches `### PRIORITY 1`, `## PRIORITY 2:`,
   `**PRIORITY 3:**`, `### NEW PRIORITY 0`, etc.).
3. For each block, extract: the target (GPU / competition / model), the
   recipe (loss, hyperparameters, data, expected metric), and any
   GATED/DISTINCT-FROM constraints.
4. Skip blocks whose body is purely operational (e.g. "kill PID 1234",
   "free GPU 2") and contains no experiment recipe — those are Duty 2.

### Step 2: Anchor each PRIORITY block on a competition-winner recipe

Before drafting an idea entry, do a `WebSearch` for `<competition_id>
kaggle gold medal solution writeup` and `<competition_id> 1st place
solution github`. Cross-reference the published winner's specific
backbone, resolution, fold strategy, augmentation, EMA, TTA, and
ensemble shape. Where the prof's PRIORITY block is generic ("try a
stronger backbone"), upgrade it to the winner-tier specifics from
your search (e.g. "EfficientNet-B7-NS 768px patient-aware GroupKFold
5-fold + EMA + soup + 8x TTA + external ISIC-2019 pretraining").

If a `results/_winning_recipes/<competition_id>.md` file exists, read
it first and use it as the recipe template — only WebSearch if the
file is missing or the trigger explicitly demands a fresh recipe.

DO NOT submit a generic config. Mid-tier ideas waste GPU budget; the
medal threshold for these competitions requires the
competition-specific tricks (patient grouping, external pretraining,
stain norm, diagnosis aux heads, etc.).

### Step 3: Append one idea per block to ideas.md

For each PRIORITY block with a recipe, append an idea entry to `ideas.md`
in the same shape as existing entries (see the most recent ~10 entries in
`ideas.md` for the local style; mimic indentation and field order). The
required fields are:

- a heading or label that includes a fresh idea_id (use the next free
  4-digit ID — find by `grep -oE 'idea-[0-9]{4}' ideas.md | sort -u | tail`,
  add 1; if the file uses a different idea-id convention, follow it).
- `**Priority**: high` (or `**Priority**: critical` if the trigger calls
  out PRIORITY 0 or labels the block as URGENT).
- a fenced YAML block (```yaml ... ```) containing at minimum a
  `competition_id:` field and the recipe-specific knobs (model, loss,
  optimizer, data, expected_metric).
- a one-line **Hypothesis** drawn from the trigger block.
- the source attribution (e.g. `Source: _trigger_engineer cycle-138 PRIORITY 2`).

Use a single `Edit` (or `Write` if the file is small) — append, never
replace existing content.

### Step 3: Verify before exit

1. `grep -c 'idea-<NEW_ID>' ideas.md` must return >= 1 for every new ID
   you appended.
2. Each appended block must contain `competition_id:` (verify with grep).
3. The number of new ideas appended must be >= the number of PRIORITY
   blocks that had recipes (one-to-one minimum; you may append more than
   one for a block that prescribes multiple seeds/folds).
4. If verification fails, fix the gap and re-verify before exiting. Do
   NOT exit while any PRIORITY-with-recipe block lacks a corresponding
   ideas.md entry.

### Step 4: Mark the trigger consumed

Rename the trigger file from `_trigger_engineer*` to
`_trigger_engineer.resolved.<cycle_or_timestamp>.<original_suffix>` so
the FSM does not re-fire on it. If the trigger was already resolved by
the FSM (the file is now `_trigger_engineer.resolved.*`), skip this step.

## Duty 4: Backbone fidelity check (when an idea spec includes an explicit backbone)

If an idea spec names a specific backbone (e.g. `EVA-02-Large 448px`,
`tf_efficientnet_b7_ns`, `ConvNeXt-V2-Huge`), the generated
`solution.py` for that idea MUST instantiate that exact model via
`timm.create_model('<full_timm_name>', pretrained=True, ...)`.

DO NOT silently substitute `torchvision.models.<weaker_backbone>` (e.g.
`efficientnet_b3`, `resnet50`, `inception_v3`) when the spec asked for
a timm winner-tier backbone. Such substitution is the leading cause of
mediocre auto-generated configs — the spec asks for a 0.10-logloss
winner-tier model and the codegen produces a 4.0-logloss generic
baseline.

Before exiting, if you produced a `solution.py` for any idea with an
explicit backbone, grep your generated code:

- `grep -c 'timm.create_model' solution.py` must be >= 1.
- `grep -c '<backbone_name_from_spec>' solution.py` must be >= 1.
- `grep -c 'torchvision.models.' solution.py` must be 0 (unless the
  idea EXPLICITLY asked for a torchvision model).

If any check fails, regenerate `solution.py` with the correct
`timm.create_model(...)` call and re-verify. Do NOT exit while a
winner-tier backbone has been silently substituted.
