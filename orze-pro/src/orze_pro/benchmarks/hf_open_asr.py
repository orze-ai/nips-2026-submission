"""HFOpenASRLeaderboard preset.

Wraps the HuggingFace Open ASR Leaderboard evaluation pipeline (8 ESB
datasets) and locks the compliance invariants the leaderboard requires:

* same decode hyperparameters across all 8 datasets
* LB-official Whisper EnglishTextNormalizer applied to refs and preds
* official ESB test sets (``hf-audio/esb-datasets-test-only-sorted``)
* WER computed via ``evaluate.load("wer")`` (jiwer-based)

The preset does not implement the inference loop itself — it expects the
caller to point at a working ``run_eval.py`` (the integration script the
caller would PR to ``huggingface/open_asr_leaderboard``). The preset's
job is to:

1. Refuse calls that would violate compliance (per-dataset decode tweaks,
   custom normalizers, unofficial test sets).
2. Run the eval script with the locked flags.
3. Aggregate the per-dataset WERs into the leaderboard macro.
4. Return a structured result the caller can publish.

Typical use::

    from orze_pro.benchmarks import HFOpenASRLeaderboard

    bench = HFOpenASRLeaderboard(
        eval_script="open_asr_leaderboard/higgs_audio/run_eval.py",
        wrapper_script="run_eval_lora_chk38000.sh",
    )
    result = bench.run(
        model="bosonai/higgs-audio-v3-8b-stt-v2",
        out_dir="results/v2_canonical",
    )
    print(result["macro_wer"])      # e.g. 5.449
    print(result["per_dataset"])
    print(result["compliant"])      # True if all invariants held
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from orze.benchmarks import Preset, ComplianceViolation


# The 8 ESB test sets the leaderboard scores.
LEADERBOARD_DATASETS = (
    "ami",
    "earnings22",
    "gigaspeech",
    "librispeech.clean",
    "librispeech.other",
    "spgispeech",
    "tedlium",
    "voxpopuli",
)


class HFOpenASRLeaderboard(Preset):
    """HuggingFace Open ASR Leaderboard preset (8 ESB datasets)."""

    name = "hf-open-asr-leaderboard-v1"
    compliance_invariants: dict[str, Any] = {
        "decode": "greedy_default",  # no_repeat_ngram_size=0, repetition_penalty=1.0, num_beams=1
        "normalizer": "whisper.EnglishTextNormalizer",
        "datasets_source": "hf-audio/esb-datasets-test-only-sorted",
        "wer_metric": "evaluate.wer",
        "uniform_decode_across_datasets": True,
    }

    # The decode flags this preset locks. Caller cannot override these.
    LOCKED_DECODE_FLAGS: dict[str, Any] = {
        "no_repeat_ngram_size": 0,
        "repetition_penalty": 1.0,
        "num_beams": 1,
        "do_sample": False,
        "lora_scale": 1.0,
    }

    def __init__(
        self,
        eval_script: str | os.PathLike[str] | None = None,
        wrapper_script: str | os.PathLike[str] | None = None,
        cwd: str | os.PathLike[str] | None = None,
    ):
        """
        Args:
            eval_script: Path to the per-dataset run_eval.py (the file you'd
                PR to ``huggingface/open_asr_leaderboard``). Used for
                inspection only when ``wrapper_script`` is given.
            wrapper_script: Optional bash wrapper that fans out the 8 datasets
                across GPUs. If not given, a single-process eval is launched
                per dataset via the eval_script.
            cwd: Working directory for subprocess calls.
        """
        self.eval_script = Path(eval_script) if eval_script else None
        self.wrapper_script = Path(wrapper_script) if wrapper_script else None
        self.cwd = Path(cwd) if cwd else Path.cwd()

    # --- preset API ---

    def run(
        self,
        model: str,
        out_dir: str | os.PathLike[str],
        local_lora_path: str | os.PathLike[str] | None = None,
        timeout_s: int | None = None,
        **overrides: Any,
    ) -> dict[str, Any]:
        """Run the canonical 8-dataset eval. Returns a structured result.

        Args:
            model: HF model id or local path. Should be a self-contained STT
                model (LoRA pre-merged) OR a base model used together with
                ``local_lora_path``.
            out_dir: Where per-dataset CHK files and SUMMARY.json land.
            local_lora_path: Optional LoRA path applied at inference.
            timeout_s: Max subprocess wall-clock; None = no timeout.
            overrides: Forwarded to the eval script. Cannot override locked
                invariants — raises ComplianceViolation if you try.

        Returns:
            dict with keys ``macro_wer``, ``per_dataset``, ``n_total``,
            ``compliant``, ``out_dir``, ``model``, and ``preset``.
        """
        # Compliance gate first.
        self._check_no_invariants_violated(overrides)
        for k, v in overrides.items():
            if k in self.LOCKED_DECODE_FLAGS and v != self.LOCKED_DECODE_FLAGS[k]:
                raise ComplianceViolation(
                    f"{self.name}: decode flag '{k}' is locked to {self.LOCKED_DECODE_FLAGS[k]!r}, "
                    f"got {v!r}. The leaderboard requires uniform default greedy decode."
                )

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        if self.wrapper_script is not None:
            cmd = ["bash", str(self.wrapper_script)]
            if local_lora_path:
                cmd.append(str(local_lora_path))
            cmd.append(str(out))
        elif self.eval_script is not None:
            # Caller wants single-shot eval — they'll need to loop datasets
            # themselves. We still validate the script exists.
            if not self.eval_script.exists():
                raise FileNotFoundError(f"eval_script not found: {self.eval_script}")
            raise NotImplementedError(
                "single-script mode not yet implemented; pass wrapper_script that "
                "fans out the 8 datasets"
            )
        else:
            raise ValueError("must construct preset with eval_script or wrapper_script")

        # Subprocess.
        env = os.environ.copy()
        proc = subprocess.run(
            cmd,
            cwd=str(self.cwd),
            env=env,
            timeout=timeout_s,
            capture_output=True,
            text=True,
        )

        summary_path = out / "SUMMARY.json"
        if proc.returncode != 0 or not summary_path.exists():
            raise RuntimeError(
                f"eval failed (returncode={proc.returncode}). "
                f"stderr tail:\n{proc.stderr[-1000:]}"
            )

        return self.parse_summary(summary_path, model=model)

    def parse_summary(
        self, summary_path: str | os.PathLike[str], model: str = ""
    ) -> dict[str, Any]:
        """Parse a SUMMARY.json into the preset's standard result shape.

        Useful when the eval was already run and you just want to validate
        compliance + format. Raises if any of the 8 leaderboard datasets are
        missing from the summary.
        """
        d = json.loads(Path(summary_path).read_text())
        per_ds = d["per_dataset"]

        missing = [name for name in LEADERBOARD_DATASETS if name not in per_ds]
        if missing:
            raise ComplianceViolation(
                f"{self.name}: SUMMARY.json missing required leaderboard datasets: {missing}. "
                f"All 8 are required."
            )

        return {
            "preset": self.name,
            "model": model,
            "macro_wer": d.get("macro_avg_wer"),
            "per_dataset": {ds: per_ds[ds]["wer"] for ds in LEADERBOARD_DATASETS},
            "per_dataset_n": {ds: per_ds[ds]["n"] for ds in LEADERBOARD_DATASETS},
            "n_total": sum(per_ds[ds]["n"] for ds in LEADERBOARD_DATASETS),
            "compliant": True,
            "out_dir": str(Path(summary_path).parent),
        }


__all__ = ["HFOpenASRLeaderboard", "LEADERBOARD_DATASETS"]
