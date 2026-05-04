"""SOP CLI handler (invoked from orze.cli when ``orze sop ...`` runs).

Kept in orze-pro because SOPs are a pro feature — the SOP registry lives
here too. The argparse subparser is registered in ``orze.cli`` (so users
see ``orze sop --help`` out of the box) but the handler delegates here
via a guarded import that silently degrades when orze-pro is absent.

CALLING SPEC:
    run_sop_subcommand(args) -> int
        args: argparse Namespace with sop_command in {list, check, status}.
        Returns process exit code (0 = success, 1 = wiring errors, 2 = usage).
"""
from __future__ import annotations

from orze_pro._gate import require_license; require_license()

from pathlib import Path

from orze_pro.skills.registry import discover_skills, validate_wiring


def _collect_tier2_sops(project_root: Path, results_dir: str) -> list:
    """Return rows for engine-enforced YAML SOPs (Tier 2).

    Tier 2 SOPs are YAML specs the professor writes at runtime that the
    engine loads and enforces. Three kinds:
      - validator: <results>/_validators/*.yaml  (blocks ideas)
      - method:    <results>/_methods/*.yaml     (knowledge/recipe spec)
      - portfolio: <results>/_portfolios/*.yaml  (experiment generator)

    Returns rows as dicts shaped like the skill rows used by _cmd_list.
    Every Tier 2 SOP is 'dynamic' (authored at runtime, not bundled).
    """
    try:
        from orze_pro.engine.sop_tier2 import (
            load_method_specs, load_validators, load_portfolios,
        )
    except ImportError:
        return []
    results_path = project_root / results_dir
    rows: list = []

    for kind, loader in (
        ("validator", load_validators),
        ("portfolio", load_portfolios),
    ):
        try:
            items = loader(results_path)
        except Exception:
            items = []
        for it in items or []:
            name = it.get("name") or Path(it.get("_file", "?")).stem
            rows.append({
                "id": name,
                "kind": kind,
                "tier": "dynamic",
                "role": "-",
                "name": name,
            })

    try:
        methods = load_method_specs(results_path) or {}
    except Exception:
        methods = {}
    for name, _spec in methods.items():
        rows.append({
            "id": name,
            "kind": "method",
            "tier": "dynamic",
            "role": "-",
            "name": name,
        })
    return rows


def _cmd_list(args) -> int:
    project_root = Path(args.project_root).resolve()
    skills = discover_skills(project_root)
    tier2 = _collect_tier2_sops(project_root, getattr(args, "results_dir",
                                                      "orze_results"))

    rows: list = []
    for s in skills:
        rows.append({
            "id": s.id,
            "kind": "skill",
            "tier": s.tier,
            "role": s.role or "-",
            "name": s.name,
        })
    rows.extend(tier2)

    if not rows:
        print(f"No SOPs found (bundled, {project_root}/skills/, or "
              f"{project_root}/{args.results_dir}/_methods|_validators|_portfolios)")
        return 0

    print(f"{'ID':<34} {'KIND':<10} {'TIER':<8} {'ROLE':<14} NAME")
    print("-" * 92)
    for r in sorted(rows, key=lambda x: (x["kind"], x["tier"],
                                          x["role"], x["id"])):
        print(f"{r['id']:<34} {r['kind']:<10} {r['tier']:<8} "
              f"{r['role']:<14} {r['name']}")
    print(f"\n{len(rows)} SOP(s): "
          f"{sum(1 for r in rows if r['kind']=='skill')} skill, "
          f"{sum(1 for r in rows if r['kind']=='method')} method, "
          f"{sum(1 for r in rows if r['kind']=='validator')} validator, "
          f"{sum(1 for r in rows if r['kind']=='portfolio')} portfolio")
    return 0


def _cmd_check(args) -> int:
    project_root = Path(args.project_root).resolve()
    skills = discover_skills(project_root)
    issues = validate_wiring(skills)
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    if not issues:
        print(f"OK: {len(skills)} skills, no wiring issues")
        return 0
    for i in errors:
        print(f"ERROR [{i.skill_id}] {i.message}")
    for i in warnings:
        print(f"WARN  [{i.skill_id}] {i.message}")
    print(f"\n{len(skills)} skills, {len(errors)} error(s), "
          f"{len(warnings)} warning(s)")
    return 1 if errors else 0


def _cmd_status(args) -> int:
    try:
        from orze.skills.receipts import read_receipt
    except ImportError:
        print("orze.skills.receipts not available — upgrade orze to read "
              "execution receipts.")
        return 2
    project_root = Path(args.project_root).resolve()
    skills = discover_skills(project_root)
    receipts_dir = project_root / args.results_dir / "_receipts"
    last_evidence: dict = {}
    if receipts_dir.exists():
        for rpath in sorted(receipts_dir.glob("*.json")):
            try:
                r = read_receipt(rpath)
            except Exception:
                continue
            for sid in r.skills_declared:
                evidenced = sid in r.skills_evidenced
                prev = last_evidence.get(sid)
                if prev is None or r.cycle > prev[1]:
                    last_evidence[sid] = (r.role, r.cycle, evidenced)
    print(f"{'ID':<22} {'ROLE':<14} {'LAST_CYCLE':<11} EVIDENCED")
    print("-" * 65)
    for s in sorted(skills, key=lambda x: ((x.role or ''), x.order)):
        ev = last_evidence.get(s.id)
        if ev is None:
            print(f"{s.id:<22} {(s.role or '-'):<14} "
                  f"{'never':<11} -")
        else:
            _role, cycle, flag = ev
            print(f"{s.id:<22} {(s.role or '-'):<14} "
                  f"{cycle:<11} {'yes' if flag else 'NO'}")
    return 0


_DISPATCH = {"list": _cmd_list, "check": _cmd_check, "status": _cmd_status}


def run_sop_subcommand(args) -> int:
    sub = getattr(args, "sop_command", None)
    handler = _DISPATCH.get(sub)
    if handler is None:
        print("usage: orze sop {list|check|status}")
        return 2
    return handler(args)
