"""Tests for orze_pro.skills.registry — SOP discovery + wiring validation."""
import textwrap

from orze_pro.skills.registry import (
    SkillMetadata,
    discover_skills,
    validate_wiring,
)


def _write_skill(dirpath, filename, body_yaml):
    path = dirpath / filename
    path.write_text(textwrap.dedent(body_yaml).lstrip("\n"), encoding="utf-8")
    return path


def test_discover_skills_scans_project_skills_dir(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "a.skill.md", """
        ---
        id: sop-a
        role: data_analyst
        produces: [out_a.md]
        ---
        body a
    """)
    _write_skill(skills_dir, "b.skill.md", """
        ---
        id: sop-b
        role: professor
        consumed_by: [research]
        ---
        body b
    """)
    skills = discover_skills(tmp_path)
    ids = {s.id: s for s in skills}
    assert "sop-a" in ids
    assert ids["sop-a"].role == "data_analyst"
    assert ids["sop-a"].produces == ["out_a.md"]
    assert "sop-b" in ids
    assert ids["sop-b"].consumed_by == ["research"]


def test_discover_ignores_skill_without_id(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "no_id.skill.md", """
        ---
        name: anonymous
        ---
        body
    """)
    assert discover_skills(tmp_path, include_bundled=False) == []


def test_discover_handles_no_skills_dir(tmp_path):
    assert discover_skills(tmp_path, include_bundled=False) == []


def test_dangling_requires_is_reported():
    skills = [
        SkillMetadata(id="sop-b", name="b", role="thinker",
                      requires=["sop-nonexistent"]),
    ]
    issues = validate_wiring(skills)
    errs = [i for i in issues if i.severity == "error"]
    assert any("sop-nonexistent" in i.message and "requires" in i.message
               for i in errs)


def test_dangling_produces_is_reported_as_warning():
    skills = [
        SkillMetadata(id="sop-orphan", name="orphan", role="data_analyst",
                      produces=["out_orphan.md"]),
    ]
    issues = validate_wiring(skills)
    warns = [i for i in issues if i.severity == "warning"]
    assert any("out_orphan.md" in i.message and "no consumer" in i.message
               for i in warns)


def test_requires_counts_as_consumption():
    skills = [
        SkillMetadata(id="sop-a", name="a", role="data_analyst",
                      produces=["out_a.md"]),
        SkillMetadata(id="sop-b", name="b", role="thinker",
                      requires=["sop-a"]),
    ]
    issues = validate_wiring(skills)
    assert all("sop-a" not in i.message or i.severity != "warning"
               for i in issues)


def test_consumed_by_role_name_counts_as_consumption():
    skills = [
        SkillMetadata(id="sop-a", name="a", role="data_analyst",
                      produces=["out_a.md"], consumed_by=["thinker"]),
        SkillMetadata(id="sop-b", name="b", role="thinker"),
    ]
    issues = validate_wiring(skills)
    warns_for_a = [i for i in issues if i.skill_id == "sop-a"
                   and i.severity == "warning"]
    assert warns_for_a == []


def test_overrides_target_must_exist():
    skills = [
        SkillMetadata(id="sop-override", name="o", role="thinker",
                      overrides="sop-missing"),
    ]
    issues = validate_wiring(skills)
    errs = [i for i in issues if i.severity == "error"]
    assert any("sop-missing" in i.message and "overrides" in i.message
               for i in errs)


def test_clean_wiring_produces_no_issues():
    skills = [
        SkillMetadata(id="sop-a", name="a", role="data_analyst",
                      produces=["out_a.md"], consumed_by=["thinker"]),
        SkillMetadata(id="sop-b", name="b", role="thinker",
                      requires=["sop-a"]),
    ]
    assert validate_wiring(skills) == []


def test_discover_includes_bundled_static_sops(tmp_path):
    """At least one bundled SOP exists after discover_skills; all are static."""
    skills = discover_skills(tmp_path, include_bundled=True)
    assert skills, "no bundled SOPs discovered"
    assert all(s.tier == "static" for s in skills)
    # thinker_base is a stable landmark bundled with orze-pro.
    ids = {s.id for s in skills}
    assert "sop-th-base" in ids


def test_bundled_discovery_can_be_disabled(tmp_path):
    skills = discover_skills(tmp_path, include_bundled=False)
    assert skills == []


def test_project_skill_overrides_bundled_on_id_clash(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    # Override the bundled thinker_base with a project-local SOP of the
    # same id; discover_skills must return the project version.
    _write_skill(skills_dir, "override.skill.md", """
        ---
        id: sop-th-base
        name: project_override
        role: thinker
        ---
        dynamic body wins
    """)
    skills = discover_skills(tmp_path, include_bundled=True)
    matches = [s for s in skills if s.id == "sop-th-base"]
    assert len(matches) == 1
    assert matches[0].tier == "dynamic"
    assert matches[0].name == "project_override"


def test_receipt_discover_outputs_handles_sop_prefix(tmp_path, monkeypatch):
    """_receipt_discover_outputs must not crash on @sop:/@builtin refs."""
    from orze_pro.engine.role_runner import _receipt_discover_outputs
    # thinker_base declares no produces, so it produces an empty dict
    # without crashing. Generic @core is skipped (no SOP metadata).
    role_cfg = {"skills": ["@sop:thinker_base", "@core"]}
    out = _receipt_discover_outputs(role_cfg, tmp_path)
    assert isinstance(out, dict)
    assert "sop-th-base" not in out  # no produces declared


def test_receipt_discover_outputs_captures_sop_with_produces(tmp_path):
    """When a bundled SOP declares 'produces', receipts should capture it."""
    from orze_pro.engine.role_runner import _receipt_discover_outputs
    # Use one of the real static creativity SOPs.
    role_cfg = {"skills": ["@sop:data_analyst_anomaly_hypotheses"]}
    out = _receipt_discover_outputs(role_cfg, tmp_path)
    assert "sop-da-anomaly" in out
    assert any("_failure_hypotheses.md" in p for p in out["sop-da-anomaly"])


def test_tier_field_defaults_to_dynamic_for_project_skills(tmp_path):
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    _write_skill(skills_dir, "x.skill.md", """
        ---
        id: sop-x
        role: thinker
        ---
        body
    """)
    skills = discover_skills(tmp_path, include_bundled=False)
    assert len(skills) == 1
    assert skills[0].tier == "dynamic"
