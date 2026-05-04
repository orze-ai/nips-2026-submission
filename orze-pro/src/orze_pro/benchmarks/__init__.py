"""orze_pro.benchmarks — curated, maintained benchmark presets.

These build on `orze.benchmarks.Preset`. Each concrete preset locks the
compliance invariants for one well-known leaderboard so callers can't
accidentally produce a non-comparable number.

Available presets:

* :class:`HFOpenASRLeaderboard` — HuggingFace Open ASR Leaderboard
  (8 ESB datasets, Whisper EnglishTextNormalizer, uniform decode).
"""

from __future__ import annotations

from orze_pro.benchmarks.hf_open_asr import HFOpenASRLeaderboard

__all__ = ["HFOpenASRLeaderboard"]
