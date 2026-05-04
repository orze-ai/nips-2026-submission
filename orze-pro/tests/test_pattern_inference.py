"""Tests for orze_pro.agents.pattern_inference — LLM-backed regex inferrer."""
from __future__ import annotations

from pathlib import Path
from unittest import mock

from orze_pro.agents.pattern_inference import (
    _parse_patterns,
    _sample_log,
    infer_metric_patterns,
)


def test_parse_patterns_returns_list_from_clean_json():
    resp = '["test_mAP\\\\s*=\\\\s*([0-9.]+)", "val_mAP\\\\s*=\\\\s*([0-9.]+)"]'
    out = _parse_patterns(resp)
    assert len(out) == 2
    assert all(p.startswith(("test_mAP", "val_mAP")) for p in out)


def test_parse_patterns_strips_markdown_fences():
    resp = '```json\n["score\\\\s*=\\\\s*([0-9.]+)"]\n```'
    out = _parse_patterns(resp)
    assert out == ["score\\s*=\\s*([0-9.]+)"]


def test_parse_patterns_rejects_uncompilable():
    resp = '["[unclosed", "valid\\\\s*=\\\\s*([0-9.]+)"]'
    out = _parse_patterns(resp)
    assert out == ["valid\\s*=\\s*([0-9.]+)"]


def test_parse_patterns_rejects_patterns_without_capture_group():
    resp = '["no_group_here", "with_group\\\\s*=\\\\s*([0-9.]+)"]'
    out = _parse_patterns(resp)
    assert out == ["with_group\\s*=\\s*([0-9.]+)"]


def test_parse_patterns_empty_on_no_json():
    assert _parse_patterns("I dunno man.") == []


def test_parse_patterns_empty_on_non_string_elements():
    resp = '[123, true, "valid\\\\s*=\\\\s*([0-9.]+)"]'
    out = _parse_patterns(resp)
    assert out == ["valid\\s*=\\s*([0-9.]+)"]


def test_parse_patterns_empty_array_returns_empty():
    assert _parse_patterns("[]") == []


def test_sample_log_shortens_long_logs():
    text = "\n".join(f"line {i}" for i in range(500))
    out = _sample_log(text)
    assert "elided" in out
    assert "line 0" in out
    assert "line 499" in out


def test_sample_log_short_log_untouched():
    text = "line 1\nline 2\nline 3\n"
    assert _sample_log(text) == text


def test_infer_returns_empty_when_log_blank(tmp_path):
    ts = tmp_path / "train.py"
    ts.write_text("pass\n")
    assert infer_metric_patterns(ts, "   \n  \n", "map") == []


def test_infer_returns_empty_when_cli_missing(tmp_path):
    ts = tmp_path / "train.py"
    ts.write_text("pass\n")
    with mock.patch("shutil.which", return_value=None), \
         mock.patch("subprocess.run", side_effect=FileNotFoundError):
        out = infer_metric_patterns(ts, "some log content\n", "map",
                                    claude_bin="nonexistent-claude-xyz")
    assert out == []


def test_infer_handles_cli_timeout(tmp_path):
    import subprocess as sp
    ts = tmp_path / "train.py"
    ts.write_text("pass\n")
    with mock.patch("subprocess.run",
                    side_effect=sp.TimeoutExpired(cmd="x", timeout=1)):
        out = infer_metric_patterns(ts, "log content\n", "map")
    assert out == []


def test_infer_parses_cli_response(tmp_path):
    ts = tmp_path / "train.py"
    ts.write_text("pass\n")
    fake = mock.Mock()
    fake.stdout = '["ndcg\\\\s*came in at\\\\s*([0-9.]+)"]'
    fake.returncode = 0
    with mock.patch("subprocess.run", return_value=fake):
        out = infer_metric_patterns(ts, "NDCG came in at 0.82\n", "ndcg")
    assert out == ["ndcg\\s*came in at\\s*([0-9.]+)"]
