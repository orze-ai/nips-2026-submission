"""Small utilities shared by orze-pro agents.

CALLING SPEC:
    detect_gpu_info() -> dict
        Returns {'gpu_name': str, 'vram_mb': int, 'gpu_count': int} from
        nvidia-smi; empty dict on failure.

    load_env_file(path='.env') -> None
        Parse a simple key=value .env file into os.environ (setdefault).
        No-op if the file is missing.

    call_first_available_llm(prompt) -> str
        Try Gemini / Anthropic / OpenAI in order (whichever has an API key)
        and return the response, or '' if all backends fail.
"""
from __future__ import annotations

from orze_pro._gate import require_license; require_license()

import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger("orze_pro.agents")


def detect_gpu_info() -> dict:
    """Detect GPU type and VRAM from nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {}
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if not lines:
            return {}
        parts = lines[0].split(", ")
        gpu_name = parts[0] if len(parts) > 0 else "unknown"
        vram_mb = int(parts[1]) if len(parts) > 1 else 0
        return {
            "gpu_name": gpu_name,
            "vram_mb": vram_mb,
            "gpu_count": len(lines),
        }
    except Exception as e:
        logger.debug("nvidia-smi detection failed: %s", e)
        return {}


def load_env_file(env_path: str | Path = ".env") -> None:
    """Parse a simple key=value .env file into os.environ (setdefault)."""
    p = Path(env_path)
    if not p.exists():
        return
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and value:
                    os.environ.setdefault(key, value)
    except Exception:
        pass


def call_first_available_llm(prompt: str) -> str:
    """Try available LLM backends in order; return first good response."""
    from orze_pro.agents.research_llm import call_llm

    backends = []
    if os.environ.get("GEMINI_API_KEY"):
        backends.append(("gemini", os.environ["GEMINI_API_KEY"],
                         "gemini-3.1-pro-preview"))
    if os.environ.get("ANTHROPIC_API_KEY"):
        backends.append(("anthropic", os.environ["ANTHROPIC_API_KEY"],
                         "claude-opus-4-6"))
    if os.environ.get("OPENAI_API_KEY"):
        backends.append(("openai", os.environ["OPENAI_API_KEY"], "o3"))
    kimi_key = os.environ.get("KIMI_API_KEY") or os.environ.get("MOONSHOT_API_KEY")
    if kimi_key:
        backends.append(("kimi", kimi_key, "kimi-k2.6"))

    if not backends:
        logger.error("No API keys found (GEMINI_API_KEY, ANTHROPIC_API_KEY, "
                     "OPENAI_API_KEY, KIMI_API_KEY). Set one in .env or environment.")
        return ""

    for backend, api_key, model in backends:
        try:
            logger.info("Calling %s (%s)...", backend, model)
            response = call_llm(prompt, backend=backend, api_key=api_key,
                                model=model)
            if response and len(response) > 200:
                return response
            logger.warning("%s returned insufficient response (%d chars)",
                           backend, len(response))
        except Exception as e:
            logger.warning("%s failed: %s", backend, e)
    return ""
