import pytest
import tempfile
import os
import json
import time
from pathlib import Path
from unittest.mock import patch, MagicMock
from orze_pro.telemetry import _build_payload, _is_enabled, _get_key_hash


def test_telemetry_disabled_by_config():
    assert _is_enabled({"telemetry": False}) is False
    assert _is_enabled({"telemetry": True}) is True
    assert _is_enabled({}) is True


def test_build_payload_minimal():
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results")
        os.makedirs(results_dir)

        cfg = {
            "train_script": "train.py",
            "report": {"title": "Test Project", "primary_metric": "accuracy", "sort": "descending"},
            "roles": {"research": {}},
        }

        with patch("orze_pro.license.check_license", return_value={"customer": "test", "tier": "pro"}):
            with patch("orze_pro.license.get_activation_status", return_value={}):
                payload = _build_payload(cfg, results_dir)

        assert "projects" in payload
        assert len(payload["projects"]) == 1
        assert payload["projects"][0]["name"] == "Test Project"
        assert payload["customer"] == "test"
        assert "system" in payload


def test_build_payload_with_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results")
        os.makedirs(results_dir)

        # Create idea_lake.db at the correct path (.orze/idea_lake.db)
        import sqlite3
        orze_dir = os.path.join(tmpdir, ".orze")
        os.makedirs(orze_dir)
        db_path = os.path.join(orze_dir, "idea_lake.db")
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE ideas (idea_id TEXT PRIMARY KEY, id_num INTEGER, "
                     "title TEXT, status TEXT, approach_family TEXT DEFAULT 'other')")
        conn.execute("INSERT INTO ideas VALUES ('idea-001', 1, 'test1', 'completed', 'resnet')")
        conn.execute("INSERT INTO ideas VALUES ('idea-002', 2, 'test2', 'failed', 'vit')")
        conn.execute("INSERT INTO ideas VALUES ('idea-003', 3, 'test3', 'completed', 'resnet')")
        conn.commit()
        conn.close()

        cfg = {
            "train_script": "train.py",
            "report": {"title": "Test", "primary_metric": "acc"},
            "roles": {},
        }

        with patch("orze_pro.license.check_license", return_value={"customer": "test"}):
            with patch("orze_pro.license.get_activation_status", return_value={}):
                payload = _build_payload(cfg, results_dir)

        proj = payload["projects"][0]
        assert proj["experiments"]["total"] == 3
        assert proj["experiments"]["completed"] == 2
        assert proj["experiments"]["failed"] == 1
        assert proj["families"] == {"resnet": 2, "vit": 1}
        assert proj["hit_rate"] == 0.67
