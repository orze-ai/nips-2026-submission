---
id: sop-eng-fix-bugs
name: engineer_fix_bugs
role: engineer
order: 30
produces: []
consumed_by: []
requires: []
trigger: always
---

## Duty 2: Fix bugs (when trigger says "BLOCKED PORTFOLIO" or diagnosis exists)

A problem has been detected. Diagnose, fix, verify.

### Step 1: Read diagnosis

1. Read `{results_dir}/_fix_diagnosis.json` if it exists.
2. Read `{results_dir}/orze.log` (last 200 lines) — recent errors.
3. Read `orze.yaml` — configuration.
4. Read `{results_dir}/status.json` — pipeline state.

### Step 2: Diagnose root cause

Common issues: missing files, API key expired, config errors, training
script bugs, OOM, dependency issues.

### Step 3: Fix

1. **DO NOT modify** files under `orze/` or `orze-pro/` directories.
2. **DO NOT modify** `ideas.md` or experiment results — UNLESS the
   trigger has PRIORITY sections (see Duty 3 in engineer_implement),
   which is the one case where appending to `ideas.md` is required.
3. You MAY modify: `orze.yaml`, `train_*.py`, `.env`, config files.
4. Prefer minimal fixes.
5. After editing, verify syntax.

### Step 4: Verify

1. `orze --check -c orze.yaml` — no errors.
2. `python3 train_script.py --help` — script parses.
3. If you fixed a training script, run
   `python3 -c "import ast; ast.parse(open('script.py').read())"`.
