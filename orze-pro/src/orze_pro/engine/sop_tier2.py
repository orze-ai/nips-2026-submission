"""Tier 2 SOP engine — professor-generated YAML specs enforced by the engine.

CALLING SPEC:
    load_method_specs(results_dir) -> Dict[str, dict]
        Load all method specs from results/_methods/*.yaml.

    load_validators(results_dir) -> List[dict]
        Load all validator rules from results/_validators/*.yaml.

    validate_idea_tier2(idea_config, validators) -> (bool, str)
        Check idea against professor-written validators.

    load_portfolios(results_dir) -> List[dict]
        Load experiment portfolios from results/_portfolios/*.yaml.

    inject_method_context(methods) -> str
        Format method specs as markdown for research agent context.

    build_sop_instructions(results_dir) -> str
        Build prompt section teaching professor to write YAML specs.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

logger = logging.getLogger("orze")


def _triggers_dir(fallback: Path) -> Path:
    """Resolve triggers directory from ORZE_DIR env, falling back to results_dir."""
    orze_dir = os.environ.get("ORZE_DIR")
    if orze_dir:
        d = Path(orze_dir) / "triggers"
        d.mkdir(parents=True, exist_ok=True)
        return d
    return fallback

# Cache: {(dir_path, max_mtime): loaded_data}
_cache: dict = {}

# Dedupe set: warning-severity validator hits we've already logged.
# Key: (idea_id, validator_name, field). Cleared on daemon restart.
# Prevents log spam when an idea stays queued across many scheduler
# iterations and warning-severity validators re-fire every pass.
_warned_validations: set = set()


def _load_yaml_dir(dirpath: Path) -> List[dict]:
    """Load all YAML files from a directory. Cached by max mtime."""
    if not dirpath.exists():
        return []
    files = list(dirpath.glob("*.yaml")) + list(dirpath.glob("*.yml"))
    if not files:
        return []
    max_mtime = max(f.stat().st_mtime for f in files)
    cache_key = (str(dirpath), max_mtime)
    if cache_key in _cache:
        return _cache[cache_key]

    results = []
    for f in sorted(files):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data["_file"] = str(f)
                results.append(data)
        except (yaml.YAMLError, OSError) as e:
            logger.warning("Failed to load %s: %s", f, e)
    _cache[cache_key] = results
    return results


# ---------------------------------------------------------------------------
# Method specs
# ---------------------------------------------------------------------------

def load_method_specs(results_dir: Path) -> Dict[str, dict]:
    """Load all method specs from results/_methods/*.yaml."""
    specs = _load_yaml_dir(results_dir / "_methods")
    return {s.get("name", s["_file"]): s for s in specs}


def inject_method_context(methods: Dict[str, dict]) -> str:
    """Format method specs as markdown for research agent context."""
    if not methods:
        return ""
    lines = ["## Proven Methods (from _methods/*.yaml)\n"]
    lines.append("These methods have been analyzed from source code. "
                 "Use their loss functions and training recipes.\n")
    for name, spec in methods.items():
        lines.append(f"### {name}")
        if spec.get("proven_score"):
            lines.append(f"- **Proven score**: {spec['proven_score']}")
        if spec.get("source"):
            lines.append(f"- **Source**: `{spec['source']}`")
        if spec.get("loss_components"):
            lines.append("- **Loss components**:")
            for k, v in spec["loss_components"].items():
                lines.append(f"  - {k}: `{v}`")
        if spec.get("hyperparameters"):
            lines.append("- **Hyperparameters**:")
            for k, v in spec["hyperparameters"].items():
                lines.append(f"  - {k}: {v}")
        if spec.get("config_mapping"):
            lines.append("- **Config mapping** (use in idea YAML):")
            for script, overrides in spec["config_mapping"].items():
                lines.append(f"  - `{script}`: `{overrides}`")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------

def load_validators(results_dir: Path) -> List[dict]:
    """Load all validator rules from results/_validators/*.yaml."""
    return _load_yaml_dir(results_dir / "_validators")


def _resolve_dotpath(config: dict, path: str):
    """Traverse a dot-separated field path into nested dicts.

    Returns the value if found, or _MISSING sentinel if any segment
    is absent.  E.g. _resolve_dotpath({"data": {"competition_id": "X"}},
    "data.competition_id") -> "X".
    """
    parts = path.split(".")
    cur = config
    for part in parts:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


_MISSING = object()


def _check_condition(idea_config: dict, field: str, op: str,
                     rule: dict) -> bool:
    """Evaluate a single field/operator/value condition against idea_config.

    Supports dot-separated field paths (e.g. ``data.competition_id``).
    If the field is absent, the rule is vacuously satisfied (True)
    for all operators except 'exists'. This prevents false rejections
    when ideas don't set optional fields.
    """
    value = _resolve_dotpath(idea_config, field)
    absent = value is _MISSING
    if absent:
        value = None
    # Absent field → rule doesn't apply (except exists/not_exists)
    if absent and op not in ("exists", "not_exists"):
        return True
    if op == "equals":
        return value == rule.get("value")
    elif op == "not_equals":
        return value != rule.get("value")
    elif op == "in":
        return value in rule.get("values", [])
    elif op == "not_in":
        return value not in rule.get("values", [])
    elif op == "exists":
        return not absent
    elif op == "not_exists":
        return absent
    elif op in ("gt", "gte", "lt", "lte", "less_than_or_equal"):
        try:
            fval = float(value)
            rval = float(rule.get("value", 0))
        except (TypeError, ValueError):
            return True  # can't compare → pass
        if op == "gt":
            return fval > rval
        elif op == "gte":
            return fval >= rval
        elif op == "lt":
            return fval < rval
        else:  # lte, less_than_or_equal
            return fval <= rval
    return True


def validate_idea_tier2(idea_config: dict,
                        validators: List[dict],
                        idea_id: Optional[str] = None) -> Tuple[bool, str]:
    """Check idea against professor-written validators.

    Returns (is_valid, error_message). Only 'error' severity blocks;
    'warning' severity just logs. When ``idea_id`` is provided, warning-severity
    hits are deduplicated per (idea_id, validator, field) so each match only
    logs once per daemon lifetime (prevents spam when an idea stays queued
    across many scheduler iterations).

    Supports conditional rules via ``then_require``:
        - field: share_ln_f
          operator: equals
          value: true
          then_require:
            field: qk_norm_type
            operator: equals
            value: anti_quarter

    When ``then_require`` is present, the top-level condition acts as a
    guard: if it does NOT match, the rule is skipped (vacuously true).
    Only when the guard matches is the ``then_require`` sub-condition
    checked.
    """
    for v in validators:
        severity = v.get("severity", "error")
        for rule in v.get("rules", []):
            field = rule.get("field")
            if not field:
                continue
            op = rule.get("operator", "equals")

            then_require = rule.get("then_require")
            if then_require:
                # Conditional rule: IF guard matches THEN sub-condition must hold
                guard_matches = _check_condition(idea_config, field, op, rule)
                if not guard_matches:
                    # Guard doesn't match — rule is vacuously satisfied
                    continue
                # Guard matched — check the sub-condition
                sub_field = then_require.get("field")
                sub_op = then_require.get("operator", "equals")
                passed = _check_condition(idea_config, sub_field, sub_op,
                                          then_require)
                if not passed:
                    sub_value = idea_config.get(sub_field)
                    msg = (f"Validator '{v.get('name', '?')}': "
                           f"when {field}={idea_config.get(field)!r}, "
                           f"field '{sub_field}' {sub_op} "
                           f"{then_require.get('value', then_require.get('values', '?'))} "
                           f"failed (got {sub_value!r})")
                    if severity == "error":
                        return False, msg
                    else:
                        key = (idea_id, v.get("name", "?"), sub_field)
                        if idea_id is None or key not in _warned_validations:
                            if idea_id is not None:
                                _warned_validations.add(key)
                            logger.warning("Validator warning: %s", msg)
            else:
                # Simple unconditional rule
                passed = _check_condition(idea_config, field, op, rule)
                if not passed:
                    value = idea_config.get(field)
                    msg = (f"Validator '{v.get('name', '?')}': "
                           f"field '{field}' {op} "
                           f"{rule.get('value', rule.get('values', '?'))} "
                           f"failed (got {value!r})")
                    if severity == "error":
                        return False, msg
                    else:
                        key = (idea_id, v.get("name", "?"), field)
                        if idea_id is None or key not in _warned_validations:
                            if idea_id is not None:
                                _warned_validations.add(key)
                            logger.warning("Validator warning: %s", msg)

    return True, ""


# ---------------------------------------------------------------------------
# Portfolios
# ---------------------------------------------------------------------------

def load_portfolios(results_dir: Path) -> List[dict]:
    """Load experiment portfolios from results/_portfolios/*.yaml."""
    return _load_yaml_dir(results_dir / "_portfolios")


def validate_portfolio_backbone(backbone: dict, grid: dict,
                                method: dict, python: str = "") -> Tuple[bool, str]:
    """Validate that a portfolio backbone's train_script:
    1. Follows the orze contract (accepts --idea-id, --results-dir, --config)
    2. Accepts the method's required config keys

    Returns (is_valid, error_message).
    """
    import sys
    if not python:
        python = sys.executable

    train_script = backbone.get("train_script", "")
    if not train_script:
        return False, "backbone missing train_script"

    try:
        from orze_pro.engine.sops import validate_idea, _get_valid_args
    except Exception as e:
        logger.warning("Portfolio validation error: %s", e)
        return True, ""

    # Check 1: orze contract — script must accept --idea-id and --results-dir
    valid_args = _get_valid_args(train_script, python)
    if valid_args:  # empty = can't parse, skip check
        # Only require the two essential args (idea-id, results-dir)
        # ideas-md and config are optional (some scripts don't use them)
        has_idea_id = bool({"idea_id", "idea-id"} & valid_args)
        has_results_dir = bool({"results_dir", "results-dir"} & valid_args)
        missing_contract = set()
        if not has_idea_id:
            missing_contract.add("idea-id")
        if not has_results_dir:
            missing_contract.add("results-dir")
        if missing_contract:
            return False, (f"{train_script} does not follow orze contract "
                           f"(missing: {', '.join(sorted(missing_contract))}). "
                           f"Implement the method in an orze-compatible script instead.")

    # Check 2: method config keys — script must accept the overrides + grid
    sample = dict(backbone.get("overrides", {}))
    for k, v in grid.items():
        sample[k] = v[0] if isinstance(v, list) and v else v

    return validate_idea(train_script, sample, python)


def trigger_implementation(backbone: dict, method: dict, error: str,
                           results_dir: Path) -> None:
    """Write _trigger_professor with specific implementation instructions.

    Called when a portfolio backbone fails validation — the train_script
    doesn't support the required config. Triggers the professor (not thinker)
    because the professor already has the method context from reading
    the source code and creating the portfolio. Porting a known method
    is engineering, not a paradigm shift.
    """
    method_name = method.get("name", "unknown")
    source = method.get("source", "unknown")
    train_script = backbone.get("train_script", "unknown")
    config_mapping = method.get("config_mapping", {}).get(train_script, {})
    loss_components = method.get("loss_components", {})

    # Find orze-compatible scripts (accept --idea-id) as alternatives
    import glob
    compatible_scripts = []
    for py in sorted(glob.glob(str(results_dir.parent / "train_*.py"))):
        try:
            from orze_pro.engine.sops import _get_valid_args
            args = _get_valid_args(py, "python3")
            if args and ("idea_id" in args or "idea-id" in args):
                compatible_scripts.append(Path(py).name)
        except Exception:
            pass

    is_contract_issue = "orze contract" in error

    if is_contract_issue:
        task = (
            f"IMPLEMENTATION TASK: Port {method_name} to an orze-compatible script\n\n"
            f"The script {train_script} does NOT follow the orze contract "
            f"(missing --idea-id, --results-dir, etc.). Do NOT try to fix "
            f"{train_script}. Instead, add the {method_name} loss function "
            f"to one of these orze-compatible scripts:\n"
            f"  {', '.join(compatible_scripts)}\n\n"
            f"Error: {error}\n\n"
            f"Source code to reference: {source}\n"
            f"Loss components: {loss_components}\n\n"
            f"Steps:\n"
            f"1. Read the original implementation at {source}\n"
            f"2. Choose the best orze-compatible script from: {', '.join(compatible_scripts)}\n"
            f"3. Add the loss function and argparse flags to that script\n"
        )
    else:
        task = (
            f"IMPLEMENTATION TASK: Add {method_name} support to {train_script}\n\n"
            f"The script {train_script} follows the orze contract but is "
            f"missing the required config keys for {method_name}.\n\n"
            f"Error: {error}\n\n"
            f"Source code to reference: {source}\n"
            f"Required config mapping: {config_mapping}\n"
            f"Loss components: {loss_components}\n\n"
            f"Steps:\n"
            f"1. Read the original implementation at {source}\n"
            f"2. Add the loss function and argparse flags to {train_script}\n"
            f"3. Test with: python {train_script} --help\n"
        )

    # Trigger engineer (primary — implements features + fixes bugs)
    tdir = _triggers_dir(results_dir)
    eng_trigger = tdir / "_trigger_engineer"
    try:
        eng_trigger.write_text(task, encoding="utf-8")
        logger.info("Triggered engineer: implement %s in %s", method_name, train_script)
    except OSError as e:
        logger.warning("Failed to write engineer trigger: %s", e)

    # Also trigger professor as backup (has method context)
    prof_trigger = tdir / "_trigger_professor"
    if not prof_trigger.exists():
        try:
            prof_trigger.write_text(
                f"BLOCKED PORTFOLIO: {method_name} on {train_script}. "
                f"Error: {error}. Implement the missing feature or fix the script.",
                encoding="utf-8")
        except OSError:
            pass


def generate_portfolio_ideas(portfolio: dict, results_dir: Path,
                             methods: Dict[str, dict]) -> List[dict]:
    """Generate concrete ideas from a portfolio template.

    Cross-products backbones with hyperparameter grid.
    Returns list of idea dicts ready for lake insertion.
    """
    if portfolio.get("status") != "active":
        return []

    method_name = portfolio.get("method", "")
    method = methods.get(method_name, {})
    backbones = portfolio.get("backbones", [])
    grid = portfolio.get("hyperparameter_grid", {})
    already_generated = set(portfolio.get("generated_ideas", []))

    # Build cross-product
    import itertools
    grid_keys = sorted(grid.keys())
    grid_values = [grid[k] if isinstance(grid[k], list) else [grid[k]]
                   for k in grid_keys]

    ideas = []
    for backbone in backbones:
        for combo in itertools.product(*grid_values):
            config = dict(backbone.get("overrides", {}))
            for k, v in zip(grid_keys, combo):
                config[k] = v
            if backbone.get("train_script"):
                config["train_script"] = backbone["train_script"]

            # Generate deterministic idea ID from config
            import hashlib
            config_str = yaml.dump(config, sort_keys=True)
            h = hashlib.sha256(config_str.encode()).hexdigest()[:6]
            idea_id = f"idea-pf-{h}"

            if idea_id in already_generated:
                continue

            bb_name = backbone.get("name", backbone.get("train_script", "?"))
            grid_desc = ", ".join(f"{k}={v}" for k, v in zip(grid_keys, combo))
            title = f"Portfolio: {method_name} on {bb_name} ({grid_desc})"

            ideas.append({
                "idea_id": idea_id,
                "title": title,
                "config": config,
                "priority": "high",
                "approach_family": "portfolio",
                "portfolio": portfolio.get("name", ""),
            })

    return ideas


# ---------------------------------------------------------------------------
# Professor SOP instructions
# ---------------------------------------------------------------------------

def build_sop_instructions(results_dir: Path) -> str:
    """Build prompt section teaching professor to write YAML specs.

    Injected into professor's prompt when the role is 'professor'.
    """
    methods_dir = results_dir / "_methods"
    validators_dir = results_dir / "_validators"
    portfolios_dir = results_dir / "_portfolios"

    instructions = """

## SOP Writing (MANDATORY when you discover a new method)

When you find a proven method (external codebase, paper, or manual result), write structured YAML specs that the engine enforces automatically. Do NOT write English prose in RESEARCH_RULES.md — write YAML that the engine reads.

### 1. Method Spec — Write after reading source code
```bash
# Create with Write tool:
Write results/_methods/<method_name>.yaml
```
Schema:
```yaml
name: adalea_riskprop
source: /path/to/source/code
proven_score: 0.8337
loss_components:
  pos: "-exp(penalty) * log(p)"
  neg: "-log(1-p)"
  consistency: "max(0, p_t - p_{t+1})"
hyperparameters:
  adalea_beta: 0.1
  adalea_lambda_neg: 0.2
config_mapping:
  train_e2e.py: {risk_regression: true}
  train_vmae_risk_regression.py: {loss: adalea, adalea_beta: 0.1}
```

### 2. Validator — Write to enforce direction changes (MANDATORY)

**WHEN to write a validator:**
- You decide a direction is dead → write a validator to block it
- You identify the winning approach → write a validator to require it
- You purge the queue → write a validator so it STAYS purged

**Do NOT just write prose in RESEARCH_RULES.md.** Prose gets ignored. Validators are enforced by the engine — ideas that violate them are auto-rejected before GPU launch.

```bash
Write results/_validators/<rule_name>.yaml
```
Schema:
```yaml
name: require_risk_regression
description: "Binary BCE is capped at 0.74"
rules:
  - field: risk_regression
    operator: equals
    value: true
severity: error
```

Operators: equals, not_equals, in, not_in, exists, not_exists, gt, gte, lt, lte

### 3. Portfolio — Write to auto-generate experiments
```bash
Write results/_portfolios/<plan_name>.yaml
```
Schema:
```yaml
name: port_adalea_to_all_backbones
method: adalea_riskprop
status: active
backbones:
  - name: mvitv2_s
    train_script: train_e2e.py
    overrides: {risk_regression: true}
  - name: videomaev2_large
    train_script: train_vmae_risk_regression.py
    overrides: {loss: adalea}
hyperparameter_grid:
  lr: [1e-5, 5e-5]
```

The engine will auto-generate ideas from portfolios and reject ideas that violate validators. Method specs are injected into the research agent's context so it knows the exact loss formulation.
"""

    # Show existing specs
    existing = []
    for d, label in [(methods_dir, "methods"), (validators_dir, "validators"),
                     (portfolios_dir, "portfolios")]:
        if d.exists():
            files = list(d.glob("*.yaml")) + list(d.glob("*.yml"))
            if files:
                existing.append(f"- `_/{label}/`: {', '.join(f.stem for f in files)}")

    if existing:
        instructions += "\n### Existing specs:\n" + "\n".join(existing) + "\n"

    return instructions
