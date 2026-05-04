"""Guardrail: verify telemetry payload schema matches what gadmin expects.

If this test fails, a change in orze-pro's telemetry broke the contract with
the admin server. Update orze-admin's ingest.py to match, or fix the payload.
"""
import json
import os
import tempfile
from unittest.mock import patch

import pytest

from orze_pro.telemetry import _build_payload, _is_enabled, _get_machine_id


REQUIRED_TOP_KEYS = {"hostname", "customer", "tier", "timestamp", "projects", "system"}
REQUIRED_PROJECT_KEYS = {
    "id", "name", "path", "status", "primary_metric", "experiments",
    "gpu_count", "leaderboard", "failures", "families", "hit_rate",
    "roles", "receipts", "signals", "config_summary",
}
REQUIRED_EXPERIMENT_KEYS = {"total", "completed", "failed", "active"}


@pytest.fixture
def minimal_payload():
    with tempfile.TemporaryDirectory() as tmpdir:
        results_dir = os.path.join(tmpdir, "results")
        os.makedirs(results_dir)
        cfg = {
            "train_script": "train.py",
            "report": {"title": "Contract Test", "primary_metric": "acc", "sort": "descending"},
            "roles": {"research": {}},
        }
        with patch("orze_pro.license.check_license", return_value={"customer": "test", "tier": "pro"}):
            with patch("orze_pro.license.get_activation_status", return_value={}):
                yield _build_payload(cfg, results_dir)


class TestPayloadTopLevel:
    def test_has_required_keys(self, minimal_payload):
        missing = REQUIRED_TOP_KEYS - set(minimal_payload.keys())
        assert not missing, f"Payload missing top-level keys: {missing}. gadmin ingest will fail."

    def test_timestamp_is_numeric(self, minimal_payload):
        assert isinstance(minimal_payload["timestamp"], (int, float))

    def test_projects_is_list(self, minimal_payload):
        assert isinstance(minimal_payload["projects"], list)


class TestPayloadProject:
    def test_has_required_keys(self, minimal_payload):
        proj = minimal_payload["projects"][0]
        missing = REQUIRED_PROJECT_KEYS - set(proj.keys())
        assert not missing, f"Project missing keys: {missing}. gadmin ingest will fail."

    def test_experiments_has_required_keys(self, minimal_payload):
        exp = minimal_payload["projects"][0]["experiments"]
        missing = REQUIRED_EXPERIMENT_KEYS - set(exp.keys())
        assert not missing, f"experiments missing keys: {missing}. gadmin ingest will fail."

    def test_families_is_dict(self, minimal_payload):
        assert isinstance(minimal_payload["projects"][0]["families"], dict)

    def test_failures_is_list(self, minimal_payload):
        assert isinstance(minimal_payload["projects"][0]["failures"], list)

    def test_roles_is_dict(self, minimal_payload):
        assert isinstance(minimal_payload["projects"][0]["roles"], dict)

    def test_receipts_is_dict(self, minimal_payload):
        assert isinstance(minimal_payload["projects"][0]["receipts"], dict)

    def test_signals_is_dict(self, minimal_payload):
        assert isinstance(minimal_payload["projects"][0]["signals"], dict)

    def test_leaderboard_is_list(self, minimal_payload):
        assert isinstance(minimal_payload["projects"][0]["leaderboard"], list)

    def test_config_summary_is_dict(self, minimal_payload):
        assert isinstance(minimal_payload["projects"][0]["config_summary"], dict)

    def test_hit_rate_is_none_or_float(self, minimal_payload):
        hr = minimal_payload["projects"][0]["hit_rate"]
        assert hr is None or isinstance(hr, float)

    def test_status_is_string(self, minimal_payload):
        assert isinstance(minimal_payload["projects"][0]["status"], str)
        assert minimal_payload["projects"][0]["status"] in ("running", "stopped", "dead", "stalled", "unknown")


class TestPayloadSerializable:
    """The payload must be JSON-serializable since it's POSTed over HTTP."""

    def test_json_round_trip(self, minimal_payload):
        serialized = json.dumps(minimal_payload)
        deserialized = json.loads(serialized)
        assert set(deserialized.keys()) == set(minimal_payload.keys())
        assert len(deserialized["projects"]) == len(minimal_payload["projects"])
