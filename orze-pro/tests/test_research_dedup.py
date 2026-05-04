"""Regression test: research dedup must read both ideas.md and the lake."""
import sqlite3
import tempfile
from pathlib import Path

import pytest

from orze_pro.agents.research_context import get_existing_idea_ids


def _make_lake(p: Path, ids):
    conn = sqlite3.connect(str(p))
    conn.execute("CREATE TABLE ideas (idea_id TEXT PRIMARY KEY)")
    conn.executemany("INSERT INTO ideas (idea_id) VALUES (?)", [(i,) for i in ids])
    conn.commit()
    conn.close()


def test_md_only_returns_md_ids(tmp_path: Path) -> None:
    md = tmp_path / "ideas.md"
    md.write_text("## idea-aaa: x\n\n## idea-bbb: y\n", encoding="utf-8")
    assert get_existing_idea_ids(md) == ["idea-aaa", "idea-bbb"]


def test_md_truncated_lake_full(tmp_path: Path) -> None:
    md = tmp_path / "ideas.md"
    md.write_text("## idea-aaa: latest\n", encoding="utf-8")
    lake = tmp_path / "lake.db"
    _make_lake(lake, ["idea-bbb", "idea-ccc", "idea-aaa"])
    out = get_existing_idea_ids(md, lake_db_path=lake)
    assert set(out) == {"idea-aaa", "idea-bbb", "idea-ccc"}
    # md ordering preserved first; aaa not duplicated
    assert out[0] == "idea-aaa"
    assert out.count("idea-aaa") == 1


def test_missing_lake_falls_back_to_md(tmp_path: Path) -> None:
    md = tmp_path / "ideas.md"
    md.write_text("## idea-aaa: x\n", encoding="utf-8")
    lake = tmp_path / "missing.db"
    assert get_existing_idea_ids(md, lake_db_path=lake) == ["idea-aaa"]


def test_lake_with_bad_schema_warns_does_not_crash(tmp_path: Path) -> None:
    md = tmp_path / "ideas.md"
    md.write_text("## idea-aaa: x\n", encoding="utf-8")
    lake = tmp_path / "lake.db"
    conn = sqlite3.connect(str(lake))
    conn.execute("CREATE TABLE other (k TEXT)")
    conn.commit()
    conn.close()
    assert get_existing_idea_ids(md, lake_db_path=lake) == ["idea-aaa"]
