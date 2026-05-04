"""SOP registry — discover and validate SOP skills with frontmatter metadata.

Each project SOP skill lives at ``<project_root>/skills/<name>.skill.md``
with YAML frontmatter:

    ---
    id: sop-da-anomaly              # required for registry tracking
    name: anomaly_driven_hypotheses # display name
    role: data_analyst              # role that composes this skill
    order: 30                       # composition order (low first)
    produces: [results/_X.md, ...]  # files this skill writes
    consumed_by: [research, thinker]# downstream roles / skills
    requires: [sop-da-base, ...]    # upstream skill ids that must have produced
    trigger: periodic_research_cycles(5)  # activation gate
    overrides: sop-other-id         # replace another skill
    ---

Generic skill composition (loader, trigger grammar, execution receipts)
lives in ``orze.skills``. This module is the SOP-specific layer on top.

CALLING SPEC:
    discover_skills(project_root) -> List[SkillMetadata]
    validate_wiring(skills) -> List[WiringIssue]
"""
from __future__ import annotations

from orze_pro._gate import require_license; require_license()

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set

from orze.skills.loader import parse_frontmatter


@dataclass
class SkillMetadata:
    id: str
    name: str
    role: Optional[str] = None
    order: int = 100
    produces: List[str] = field(default_factory=list)
    consumed_by: List[str] = field(default_factory=list)
    requires: List[str] = field(default_factory=list)
    trigger: Optional[str] = None
    overrides: Optional[str] = None
    path: Optional[Path] = None
    tier: str = "dynamic"  # 'static' (bundled in orze-pro) or 'dynamic' (project)


@dataclass
class WiringIssue:
    severity: str
    skill_id: str
    message: str


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    return [str(value)]


def _parse_skill_file(path: Path, tier: str) -> Optional[SkillMetadata]:
    try:
        meta, _ = parse_frontmatter(path.read_text(encoding="utf-8"))
    except OSError:
        return None
    sid = meta.get("id")
    if not sid:
        return None
    return SkillMetadata(
        id=str(sid),
        name=str(meta.get("name", path.stem.replace(".skill", ""))),
        role=meta.get("role"),
        order=int(meta.get("order", 100)),
        produces=_as_list(meta.get("produces")),
        consumed_by=_as_list(meta.get("consumed_by")),
        requires=_as_list(meta.get("requires")),
        trigger=meta.get("trigger"),
        overrides=meta.get("overrides"),
        path=path,
        tier=tier,
    )


def discover_skills(project_root: Path,
                    include_bundled: bool = True) -> List[SkillMetadata]:
    """Discover static (bundled) and dynamic (project) SOPs.

    Returns the union of:
    - Static SOPs shipped with orze-pro (``orze_pro/sops/*.skill.md``)
    - Dynamic SOPs in ``<project_root>/skills/*.skill.md``

    If a static SOP and a dynamic SOP share an id, the dynamic (project)
    entry wins — this lets projects override bundled defaults.
    """
    project_root = Path(project_root)
    by_id: dict = {}

    if include_bundled:
        try:
            from orze_pro.skills.bundled import list_bundled_skill_paths
            for path in list_bundled_skill_paths():
                meta = _parse_skill_file(path, tier="static")
                if meta is not None:
                    by_id[meta.id] = meta
        except ImportError:
            pass  # unreachable in practice — we ARE orze_pro

    skills_dir = project_root / "skills"
    if skills_dir.exists():
        for path in sorted(skills_dir.glob("*.skill.md")):
            meta = _parse_skill_file(path, tier="dynamic")
            if meta is not None:
                by_id[meta.id] = meta  # dynamic overrides static on id clash

    return sorted(by_id.values(),
                  key=lambda s: ((s.role or ""), s.order, s.id))


def validate_wiring(skills: List[SkillMetadata]) -> List[WiringIssue]:
    by_id = {s.id: s for s in skills}
    issues: List[WiringIssue] = []

    # requires must point to known skills
    for s in skills:
        for req in s.requires:
            if req not in by_id:
                issues.append(WiringIssue(
                    severity="error",
                    skill_id=s.id,
                    message=(f"{s.id} requires '{req}' but no such skill "
                             f"is registered"),
                ))

    # produces must have a consumer — another skill's ``requires`` or
    # a ``consumed_by`` target that is a known role / skill id.
    referenced_ids: Set[str] = set()
    for s in skills:
        for r in s.requires:
            referenced_ids.add(r)
    all_roles = {s.role for s in skills if s.role}

    for s in skills:
        if not s.produces:
            continue
        has_consumer = s.id in referenced_ids
        if not has_consumer:
            for target in s.consumed_by:
                if target in all_roles or target in by_id:
                    has_consumer = True
                    break
        if not has_consumer:
            issues.append(WiringIssue(
                severity="warning",
                skill_id=s.id,
                message=(f"{s.id} produces {s.produces} but has no consumer "
                         f"(no skill requires it and consumed_by targets "
                         f"{s.consumed_by or 'are empty'})"),
            ))

    # overrides target must exist
    for s in skills:
        if s.overrides and s.overrides not in by_id:
            issues.append(WiringIssue(
                severity="error",
                skill_id=s.id,
                message=(f"{s.id} overrides '{s.overrides}' but target "
                         f"does not exist"),
            ))

    return issues
