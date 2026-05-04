"""Test knowledge_dir auto-inclusion in research_agent.

Regression: the research agent only loaded a single retrospection file,
so any data_analyst / professor finding dropped into ``results/knowledge/``
was invisible to the research LLM unless someone hand-wired it. This led
to mode collapse (one strategy dominating the queue while documented
failure modes went unaddressed).

The fix is to auto-include every ``*.md`` under a ``knowledge_dir`` and
prefix each with a per-file header so the LLM can attribute claims.
"""
from pathlib import Path

from orze_pro.agents.research import _load_knowledge_dir


def test_load_knowledge_dir_concatenates_with_headers(tmp_path: Path) -> None:
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    (kdir / "alpha.md").write_text("alpha body line 1\nalpha body line 2", encoding="utf-8")
    (kdir / "beta.md").write_text("beta body", encoding="utf-8")

    out = _load_knowledge_dir(kdir)
    # Deterministic alphabetical order
    assert out.index("# alpha.md") < out.index("# beta.md")
    assert "alpha body line 1" in out
    assert "beta body" in out
    # Each file gets its filename header
    assert "# alpha.md\n\nalpha body line 1" in out
    assert "# beta.md\n\nbeta body" in out


def test_load_knowledge_dir_ignores_non_md(tmp_path: Path) -> None:
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    (kdir / "good.md").write_text("good", encoding="utf-8")
    (kdir / "ignored.txt").write_text("ignored", encoding="utf-8")
    (kdir / "ignored.json").write_text('{"x": 1}', encoding="utf-8")

    out = _load_knowledge_dir(kdir)
    assert "good" in out
    assert "ignored" not in out


def test_load_knowledge_dir_skips_empty_files(tmp_path: Path) -> None:
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    (kdir / "empty.md").write_text("   \n\n  ", encoding="utf-8")
    (kdir / "real.md").write_text("real content", encoding="utf-8")
    out = _load_knowledge_dir(kdir)
    assert "# empty.md" not in out
    assert "real content" in out


def test_load_knowledge_dir_missing_returns_empty(tmp_path: Path) -> None:
    assert _load_knowledge_dir(tmp_path / "nope") == ""


def test_load_knowledge_dir_not_a_directory(tmp_path: Path) -> None:
    f = tmp_path / "afile.md"
    f.write_text("x", encoding="utf-8")
    # Path points at a file, not a dir — must not crash
    assert _load_knowledge_dir(f) == ""


def test_load_knowledge_dir_handles_unreadable_files(tmp_path: Path) -> None:
    """One unreadable file must not poison the whole context."""
    kdir = tmp_path / "knowledge"
    kdir.mkdir()
    (kdir / "good.md").write_text("good content", encoding="utf-8")
    bad = kdir / "bad.md"
    bad.write_text("bad", encoding="utf-8")
    bad.chmod(0)  # remove all perms
    try:
        out = _load_knowledge_dir(kdir)
        assert "good content" in out
    finally:
        bad.chmod(0o644)  # restore so pytest cleanup works
