"""Tests for orze_pro.benchmarks.HFOpenASRLeaderboard preset."""

from __future__ import annotations

import json
import pytest

from orze.benchmarks import ComplianceViolation
from orze_pro.benchmarks import HFOpenASRLeaderboard
from orze_pro.benchmarks.hf_open_asr import LEADERBOARD_DATASETS


def test_locked_decode_invariants_reject_overrides(tmp_path):
    p = HFOpenASRLeaderboard(wrapper_script=tmp_path / "fake.sh")
    with pytest.raises(ComplianceViolation, match="no_repeat_ngram_size"):
        p.run(
            model="bosonai/higgs-audio-v3-8b-stt-v2",
            out_dir=tmp_path / "out",
            no_repeat_ngram_size=4,
        )


def test_compliance_invariants_reject_custom_normalizer(tmp_path):
    p = HFOpenASRLeaderboard(wrapper_script=tmp_path / "fake.sh")
    with pytest.raises(ComplianceViolation, match="normalizer"):
        p.run(
            model="x",
            out_dir=tmp_path / "out",
            normalizer="custom-thing",
        )


def test_parse_summary_canonical_shape(tmp_path):
    summary = tmp_path / "SUMMARY.json"
    summary.write_text(json.dumps({
        "lora": "/path/to/lora",
        "per_dataset": {
            ds: {"wer": 5.0 + i * 0.1, "n": 1000 + i}
            for i, ds in enumerate(LEADERBOARD_DATASETS)
        },
        "macro_avg_wer": 5.35,
    }))
    p = HFOpenASRLeaderboard(wrapper_script=tmp_path / "x.sh")
    r = p.parse_summary(summary, model="bosonai/test")
    assert r["preset"] == "hf-open-asr-leaderboard-v1"
    assert r["model"] == "bosonai/test"
    assert r["macro_wer"] == 5.35
    assert set(r["per_dataset"].keys()) == set(LEADERBOARD_DATASETS)
    assert r["compliant"] is True
    assert r["n_total"] == sum(1000 + i for i in range(len(LEADERBOARD_DATASETS)))


def test_parse_summary_missing_dataset_raises(tmp_path):
    summary = tmp_path / "SUMMARY.json"
    incomplete = {ds: {"wer": 5.0, "n": 100} for ds in LEADERBOARD_DATASETS[:5]}
    summary.write_text(json.dumps({"per_dataset": incomplete, "macro_avg_wer": 5.0}))
    p = HFOpenASRLeaderboard(wrapper_script=tmp_path / "x.sh")
    with pytest.raises(ComplianceViolation, match="missing required leaderboard datasets"):
        p.parse_summary(summary)


def test_describe_includes_invariants():
    p = HFOpenASRLeaderboard(wrapper_script="x.sh")
    s = p.describe()
    assert "hf-open-asr-leaderboard-v1" in s
    assert "uniform_decode_across_datasets" in s
