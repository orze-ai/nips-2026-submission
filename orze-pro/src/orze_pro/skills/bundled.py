"""Bundled static SOPs shipped with orze-pro.

Static SOPs (tier 1) live alongside the orze-pro package at
``orze_pro/sops/<name>.skill.md``. They ship with the wheel and are
available to every project that installs orze-pro.

Dynamic SOPs (tier 2) live in a project's ``<project_root>/skills/``
directory and are authored at runtime (by the user or the professor).

CALLING SPEC:
    BUNDLED_SOPS_DIR -> Path
        Directory containing bundled .skill.md files.

    load_bundled_skill(name) -> Tuple[str, Path]
        name: bare name (no extension), e.g. 'professor_base'
        returns: (file content, resolved path)
        raises: FileNotFoundError when the bundled SOP does not exist.

    list_bundled_skill_paths() -> List[Path]
        All bundled .skill.md paths.
"""
from __future__ import annotations

from orze_pro._gate import require_license; require_license()

from pathlib import Path
from typing import List, Tuple

BUNDLED_SOPS_DIR = Path(__file__).resolve().parent.parent / "sops"


def load_bundled_skill(name: str) -> Tuple[str, Path]:
    path = BUNDLED_SOPS_DIR / f"{name}.skill.md"
    if not path.exists():
        raise FileNotFoundError(
            f"Bundled SOP '@sop:{name}' not found at {path}")
    return path.read_text(encoding="utf-8"), path


def list_bundled_skill_paths() -> List[Path]:
    if not BUNDLED_SOPS_DIR.exists():
        return []
    return sorted(BUNDLED_SOPS_DIR.glob("*.skill.md"))
