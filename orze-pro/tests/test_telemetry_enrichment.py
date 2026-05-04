"""Guardrails for telemetry enrichment — configs, tracebacks, PID, absolute paths.

If these fail, the admin knowledge base will receive shallow data that's useless
for product development. These are the fixes that made knowledge actually valuable.
"""
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from orze_pro.telemetry import (
    _read_idea_config,
    _read_failure_detail,
    _read_idea_lake,
    _build_payload,
)


@pytest.fixture
def project_dir(tmp_path):
    """Minimal project layout with results, .orze, and idea_lake.db."""
    results = tmp_path / "results"
    results.mkdir()
    orze_dir = tmp_path / ".orze"
    orze_dir.mkdir()
    return tmp_path


class TestReadIdeaConfig:
    def test_reads_yaml_config(self, project_dir):
        idea_dir = project_dir / "results" / "idea-001"
        idea_dir.mkdir(parents=True)
        (idea_dir / "idea_config.yaml").write_text(
            "learning_rate: 0.001\nbatch_size: 32\nmodel: resnet18\n"
        )
        cfg = _read_idea_config(str(project_dir / "results"), "idea-001")
        assert cfg["learning_rate"] == 0.001
        assert cfg["batch_size"] == 32
        assert cfg["model"] == "resnet18"

    def test_trims_nested_dicts(self, project_dir):
        idea_dir = project_dir / "results" / "idea-002"
        idea_dir.mkdir(parents=True)
        (idea_dir / "idea_config.yaml").write_text(
            "optimizer:\n  name: adam\n  lr: 0.001\ndata:\n  path: /data\n"
        )
        cfg = _read_idea_config(str(project_dir / "results"), "idea-002")
        assert cfg["optimizer"]["name"] == "adam"
        assert cfg["data"]["path"] == "/data"

    def test_empty_idea_id(self, project_dir):
        assert _read_idea_config(str(project_dir / "results"), "") == {}

    def test_missing_file(self, project_dir):
        assert _read_idea_config(str(project_dir / "results"), "idea-999") == {}

    def test_strips_non_scalar_nested_values(self, project_dir):
        idea_dir = project_dir / "results" / "idea-003"
        idea_dir.mkdir(parents=True)
        (idea_dir / "idea_config.yaml").write_text(
            "top_scalar: 1\nnested:\n  ok: true\n  deep:\n    too: deep\n"
        )
        cfg = _read_idea_config(str(project_dir / "results"), "idea-003")
        assert cfg["top_scalar"] == 1
        assert cfg["nested"]["ok"] is True
        assert "deep" not in cfg["nested"]


class TestReadFailureDetail:
    def test_returns_traceback_from_analysis(self, project_dir):
        idea_dir = project_dir / "results" / "idea-001"
        idea_dir.mkdir(parents=True)
        (idea_dir / "failure_analysis.json").write_text(json.dumps({
            "category": "crash",
            "what": "KeyError",
            "why": "Missing key",
            "traceback": "Traceback (most recent call last):\n  File 'x.py'\nKeyError: 'k'",
        }))
        detail = _read_failure_detail(str(project_dir), "idea-001")
        assert "traceback" in detail
        assert "Traceback" in detail["traceback"]

    def test_no_traceback_field_when_absent(self, project_dir):
        idea_dir = project_dir / "results" / "idea-002"
        idea_dir.mkdir(parents=True)
        (idea_dir / "failure_analysis.json").write_text(json.dumps({
            "category": "oom",
            "what": "CUDA OOM",
            "why": "Too much VRAM",
        }))
        detail = _read_failure_detail(str(project_dir), "idea-002")
        assert "traceback" not in detail

    def test_truncates_long_traceback(self, project_dir):
        idea_dir = project_dir / "results" / "idea-003"
        idea_dir.mkdir(parents=True)
        (idea_dir / "failure_analysis.json").write_text(json.dumps({
            "category": "crash",
            "what": "Error",
            "traceback": "X" * 2000,
        }))
        detail = _read_failure_detail(str(project_dir), "idea-003")
        assert len(detail["traceback"]) <= 500


class TestIdeaLakeFailureEnrichment:
    def _setup_lake(self, project_dir, failures_with_analysis):
        """Create idea_lake.db + failure_analysis.json for each failed idea."""
        db_path = project_dir / ".orze" / "idea_lake.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "CREATE TABLE ideas (idea_id TEXT PRIMARY KEY, id_num INTEGER, "
            "title TEXT, status TEXT, approach_family TEXT DEFAULT 'other')"
        )
        for i, fa in enumerate(failures_with_analysis):
            idea_id = fa["idea_id"]
            conn.execute(
                "INSERT INTO ideas VALUES (?, ?, ?, 'failed', 'test')",
                (idea_id, i + 1, f"Test {i}")
            )
            idea_dir = project_dir / "results" / idea_id
            idea_dir.mkdir(parents=True, exist_ok=True)
            (idea_dir / "failure_analysis.json").write_text(json.dumps(fa["analysis"]))
            if fa.get("config"):
                import yaml
                (idea_dir / "idea_config.yaml").write_text(
                    yaml.dump(fa["config"]) if isinstance(fa["config"], dict)
                    else fa["config"]
                )
        conn.commit()
        conn.close()

    def test_failures_include_traceback(self, project_dir):
        self._setup_lake(project_dir, [{
            "idea_id": "idea-001",
            "analysis": {
                "category": "crash",
                "what": "KeyError",
                "why": "Missing key",
                "traceback": "Traceback (most recent call last):\n  File 'x.py'\nKeyError",
            },
        }])
        lake = _read_idea_lake(str(project_dir))
        assert len(lake["failures"]) == 1
        assert "traceback" in lake["failures"][0]
        assert "Traceback" in lake["failures"][0]["traceback"]

    def test_failures_include_config(self, project_dir):
        self._setup_lake(project_dir, [{
            "idea_id": "idea-001",
            "analysis": {"category": "oom", "what": "OOM"},
            "config": {"batch_size": 128, "model": "big_resnet"},
        }])
        lake = _read_idea_lake(str(project_dir))
        assert "config" in lake["failures"][0]
        assert lake["failures"][0]["config"]["batch_size"] == 128


class TestAbsoluteProjectPath:
    def test_project_path_is_absolute(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "results")
            os.makedirs(results_dir)
            cfg = {
                "train_script": "train.py",
                "report": {"title": "Test", "primary_metric": "acc"},
                "roles": {},
            }
            with patch("orze_pro.license.check_license", return_value={"customer": "t"}):
                with patch("orze_pro.license.get_activation_status", return_value={}):
                    payload = _build_payload(cfg, results_dir)
            proj_path = payload["projects"][0]["path"]
            assert os.path.isabs(proj_path), f"Project path must be absolute, got: {proj_path}"


class TestDaemonPidValidation:
    def test_stale_pid_reports_stopped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "results")
            os.makedirs(results_dir)
            state_dir = os.path.join(tmpdir, ".orze", "state")
            os.makedirs(state_dir)
            pid_file = os.path.join(state_dir, "orze.pid")
            with open(pid_file, "w") as f:
                f.write("999999999")

            cfg = {
                "train_script": "train.py",
                "report": {"title": "Test", "primary_metric": "acc"},
                "roles": {},
            }
            with patch("orze_pro.license.check_license", return_value={"customer": "t"}):
                with patch("orze_pro.license.get_activation_status", return_value={}):
                    payload = _build_payload(cfg, results_dir)
            assert payload["projects"][0]["status"] == "stopped"

    def test_live_pid_reports_running(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            results_dir = os.path.join(tmpdir, "results")
            os.makedirs(results_dir)
            state_dir = os.path.join(tmpdir, ".orze", "state")
            os.makedirs(state_dir)
            pid_file = os.path.join(state_dir, "orze.pid")
            with open(pid_file, "w") as f:
                f.write(str(os.getpid()))

            cfg = {
                "train_script": "train.py",
                "report": {"title": "Test", "primary_metric": "acc"},
                "roles": {},
            }
            with patch("orze_pro.license.check_license", return_value={"customer": "t"}):
                with patch("orze_pro.license.get_activation_status", return_value={}):
                    payload = _build_payload(cfg, results_dir)
            assert payload["projects"][0]["status"] == "running"
