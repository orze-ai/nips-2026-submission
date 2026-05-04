"""LLM backend implementations for the orze research agent.

CALLING SPEC:
    call_llm(prompt, backend, api_key="", model="", endpoint="") -> str
        Route to the appropriate LLM backend and return response text.
        Supported backends: gemini, openai, anthropic, kimi, ollama, custom.

    call_gemini(prompt, api_key, model="gemini-3.1-pro-preview", ...) -> str
    call_openai(prompt, api_key, model="o3", ...) -> str
    call_anthropic(prompt, api_key, model="claude-opus-4-6", ...) -> str
    call_kimi(prompt, api_key, model="kimi-k2.6", ...) -> str
    call_ollama(prompt, model="llama3", ...) -> str

    DEFAULT_SYSTEM_PROMPT: str
        Base system prompt for the research agent LLM.
"""

from __future__ import annotations

from orze_pro._gate import require_license; require_license()
import json
import logging
import os
import urllib.request

logger = logging.getLogger("orze.research_agent")


# ---------------------------------------------------------------------------
#  LLM backends
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, headers: dict,
               timeout: int = 120) -> dict:
    """POST JSON and return parsed response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_gemini(prompt: str, api_key: str,
                model: str = "gemini-2.5-pro",
                max_tokens: int = 32768,
                web_search: bool = False) -> str:
    """Call Gemini API. Tries multiple models on failure.

    Args:
        web_search: Enable Google Search grounding so Gemini can fetch
            live web results alongside its own knowledge.

    Gemini 2.5 Pro charges internal reasoning ("thinking") tokens against
    ``maxOutputTokens``. With ``web_search=True`` the search planner can
    balloon the thinking budget; an 8192-token cap was hitting
    ``finishReason=MAX_TOKENS`` with only a few chars of real output,
    which callers couldn't distinguish from a valid answer. Raise the
    default to 32768 and treat MAX_TOKENS-with-trivial-output as a miss
    so the fallback chain has a chance.
    """
    # Fallback chain: caller's choice first, then current stable, then
    # older-stable, then previews. The previews used to be the default
    # but return empty responses on generally-available keys, silently
    # bricking any caller that doesn't override `model` (e.g. the FSM
    # idea_verifier, which doesn't plumb config through to the LLM).
    models = [model, "gemini-2.5-pro", "gemini-1.5-pro-latest",
              "gemini-3-pro-preview", "gemini-3.1-pro-preview"]
    # Deduplicate while preserving order
    seen = set()
    unique_models = []
    for m in models:
        if m not in seen:
            seen.add(m)
            unique_models.append(m)

    for m in unique_models:
        try:
            url = (f"https://generativelanguage.googleapis.com/v1beta/"
                   f"models/{m}:generateContent?key={api_key}")
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens},
            }
            if web_search:
                payload["tools"] = [{"google_search": {}}]
            result = _post_json(url, payload, {"Content-Type": "application/json"})
            candidate = (result.get("candidates") or [{}])[0]
            finish_reason = candidate.get("finishReason", "")
            parts = candidate.get("content", {}).get("parts", []) or []
            # Thinking parts have ``thought: true`` and should not count
            # toward the user-visible answer.
            text = "\n".join(
                p.get("text", "") for p in parts
                if p.get("text") and not p.get("thought"))
            # MAX_TOKENS + effectively-empty text = thinking ate the
            # budget before any real answer emerged. Treat as miss so
            # the fallback chain keeps trying.
            if finish_reason == "MAX_TOKENS" and len(text.strip()) < 32:
                logger.warning(
                    "Gemini (%s) hit MAX_TOKENS with %d chars — "
                    "treating as empty (thinking likely exhausted budget)",
                    m, len(text))
                continue
            if text.strip():
                logger.info("Gemini (%s) returned %d chars", m, len(text))
                return text
            logger.warning(
                "Gemini (%s) returned empty response (finish=%s)",
                m, finish_reason or "unknown")
        except Exception as e:
            logger.warning("Gemini (%s) failed: %s", m, e)
    return ""


def call_openai(prompt: str, api_key: str,
                model: str = "o3",
                max_tokens: int = 8192,
                endpoint: str = "https://api.openai.com/v1") -> str:
    """Call OpenAI-compatible API (OpenAI, Azure, vLLM, etc.)."""
    url = f"{endpoint.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        result = _post_json(url, payload, headers)
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info("OpenAI (%s) returned %d chars", model, len(text))
        return text
    except Exception as e:
        logger.error("OpenAI (%s) failed: %s", model, e)
        return ""


def call_anthropic(prompt: str, api_key: str,
                   model: str = "claude-opus-4-6",
                   max_tokens: int = 8192) -> str:
    """Call Anthropic API directly."""
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    try:
        result = _post_json(url, payload, headers)
        text = "".join(
            b.get("text", "") for b in result.get("content", [])
            if b.get("type") == "text"
        )
        logger.info("Anthropic (%s) returned %d chars", model, len(text))
        return text
    except Exception as e:
        msg = str(e).lower()
        quota_markers = ("429", "401", "overloaded", "rate_limit",
                         "quota", "credit", "insufficient",
                         "permission_error", "billing")
        if any(m in msg for m in quota_markers):
            logger.error("Anthropic (%s) quota/auth failure: %s", model, e)
            raise _AnthropicQuotaError(str(e)) from e
        logger.error("Anthropic (%s) failed: %s", model, e)
        return ""


class _AnthropicQuotaError(RuntimeError):
    """Raised when Anthropic rejects for quota/auth reasons so caller can fall back."""


def call_kimi(prompt: str, api_key: str,
              model: str = "kimi-k2.6",
              max_tokens: int = 8192,
              endpoint: str = "https://api.moonshot.ai/v1") -> str:
    """Call Moonshot Kimi (OpenAI-compatible) API.

    kimi-k2.6 (and other reasoning kimi-k2.* models) reject any temperature
    other than 1, so we send temperature=1 unconditionally instead of going
    through ``call_openai``'s default 0.7.
    """
    url = f"{endpoint.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 1,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        result = _post_json(url, payload, headers)
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info("Kimi (%s) returned %d chars", model, len(text))
        return text
    except Exception as e:
        logger.error("Kimi (%s) failed: %s", model, e)
        return ""


def call_ollama(prompt: str, model: str = "llama3",
                endpoint: str = "http://localhost:11434") -> str:
    """Call local Ollama instance."""
    url = f"{endpoint.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        result = _post_json(url, payload, {"Content-Type": "application/json"},
                            timeout=300)
        text = result.get("response", "")
        logger.info("Ollama (%s) returned %d chars", model, len(text))
        return text
    except Exception as e:
        logger.error("Ollama (%s) failed: %s", model, e)
        return ""


def call_llm(prompt: str, backend: str, api_key: str = "",
             model: str = "", endpoint: str = "") -> str:
    """Route to the appropriate LLM backend."""
    if backend == "gemini":
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            logger.error("GEMINI_API_KEY not set")
            return ""
        return call_gemini(prompt, key, model=model or "gemini-2.5-pro",
                          web_search=True)

    elif backend == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            logger.error("OPENAI_API_KEY not set")
            return ""
        return call_openai(prompt, key, model=model or "o3",
                           endpoint=endpoint or "https://api.openai.com/v1")

    elif backend == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            logger.error("ANTHROPIC_API_KEY not set")
            # Fall through to fallback below rather than return ""
        else:
            try:
                return call_anthropic(prompt, key,
                                      model=model or "claude-opus-4-6")
            except _AnthropicQuotaError as e:
                logger.warning("Anthropic quota — evaluating fallback chain")
                # fall through to fallback block
        # Fallback chain: anthropic -> gemini (when ORZE_CLAUDE_FALLBACK=gemini
        # or ORZE_LLM_FALLBACK=gemini). Preserves the user-documented priority
        # order "claude subscription -> claude api -> gemini".
        fallback = (os.environ.get("ORZE_CLAUDE_FALLBACK")
                    or os.environ.get("ORZE_LLM_FALLBACK") or "").lower()
        # Accept truthy boolean shorthand "1"/"true"/"yes"/"on" as
        # "enable the documented default chain". Matches the CLI shim so
        # users don't have to know the exact backend name.
        if fallback in ("1", "true", "yes", "on"):
            fallback = "gemini"
        gkey = os.environ.get("GEMINI_API_KEY", "")
        if fallback == "gemini" and gkey:
            logger.info("Anthropic unavailable — falling back to Gemini")
            return call_gemini(prompt, gkey,
                               model="gemini-2.5-pro", web_search=True)
        return ""

    elif backend == "kimi":
        key = api_key or os.environ.get("KIMI_API_KEY",
                                        os.environ.get("MOONSHOT_API_KEY", ""))
        if not key:
            logger.error("KIMI_API_KEY (or MOONSHOT_API_KEY) not set")
            return ""
        return call_kimi(prompt, key,
                         model=model or "kimi-k2.6",
                         endpoint=endpoint or "https://api.moonshot.ai/v1")

    elif backend == "ollama":
        return call_ollama(prompt, model=model or "llama3",
                           endpoint=endpoint or "http://localhost:11434")

    elif backend == "custom":
        # OpenAI-compatible custom endpoint
        key = api_key or os.environ.get("LLM_API_KEY", "")
        return call_openai(prompt, key, model=model or "default",
                           endpoint=endpoint)

    else:
        logger.error("Unknown backend: %s", backend)
        return ""


# ---------------------------------------------------------------------------
#  Default system prompt
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """\
You are a research agent for an automated ML experiment system called orze.
Your job is to analyze past results and generate new experiment ideas.

## How It Works
- You receive the full leaderboard, failure analysis, and performance patterns
- You propose new experiments as structured ideas
- Each idea needs: a title, hypothesis, and YAML config
- The system will automatically train and evaluate your ideas

## Strategy Guidelines
- **Study the leaderboard**: understand what makes top performers successful
- **Study the failures**: avoid config patterns that consistently fail
- **Use performance patterns**: config dimensions show which values correlate with
  better results — build on high-performing values, avoid low-performing ones
- **Set parent IDs**: when iterating on a successful experiment, set "parent" to
  its idea ID so the lineage is tracked
- **Balance exploitation and exploration**: use the rules below

## Escaping Local Minima

The biggest risk is getting stuck: proposing minor variations of the same approach
hundreds of times while real improvement requires a fundamentally different idea.

**Self-check before every batch of ideas:**
1. Look at your last 10-20 proposed ideas in the context. Are they variations of
   the same approach? (same config keys tweaked, same family, same data mix with
   slightly different numbers?) If yes — you are stuck. STOP iterating and switch.
2. Look at the leaderboard plateau: if the top-5 scores haven't changed meaningfully
   in the last batch, your current approach is exhausted.

**When stuck, change the leg entirely:**
- If your last 10 ideas were all data-centric → propose model-centric ideas
- If your last 10 ideas were all model-centric → propose data-centric ideas
- If you've been varying one config key → try changing a completely different one
- If all your ideas use the same base checkpoint → try a different one

**Concretely, each batch of ideas MUST include at least one idea from a different
approach_family than your most-used family.** Check the "Approach Family Distribution"
in your context — if "data" has 500 ideas and "optimization" has 3, your next batch
must include an optimization idea.

**Exploration means questioning assumptions**, not tweaking parameters:
- "What if the loss function is wrong for this data distribution?"
- "What if the model capacity is the bottleneck, not the data?"
- "What if we're training on the wrong data entirely?"
- "What if the evaluation metric is misleading us?"

## Error-Driven Improvement
When error analysis data is present in your context, use it to drive your ideas:

1. **Diagnose**: For each error pattern, reason about the root cause. Why does the
   model fail on these inputs? Is it a data gap (model never saw this type), a
   capacity issue (model can't represent this), or a training issue (wrong objective)?
2. **Intervene**: Propose changes that address the root cause, not the symptom.
   If the model fails on short inputs because training data was mostly long inputs,
   the fix is training data composition — not inference-time filtering.
3. **Verify**: After proposing a fix, think about what side effects it might have.
   Adding more data of type X might degrade performance on type Y. Propose mitigations
   (e.g., include type Y data to prevent regression).

The strongest research ideas come from connecting specific error patterns to specific
training interventions. Generic ideas ("try a bigger model") are weak. Targeted ideas
("the model outputs empty string on inputs < 1 second because only 5% of training
data is that short — increase to 30%") are strong.

## Two Legs of Improvement

Every ML problem improves along two axes. Alternate between them:

**Data-centric** — fix what the model sees:
- More data where it fails (oversample hard cases)
- Better data quality (clean labels, remove noise)
- Data augmentation (simulate real-world conditions)
- Rebalance data mix (match the eval distribution)

**Model-centric** — fix how the model learns:
- Loss function (focal loss for hard examples, label smoothing for noisy data)
- Training procedure (curriculum learning, sample weighting by difficulty)
- Model capacity (LoRA rank, which layers to train, learning rate)
- Regularization (dropout, weight decay, early stopping)

When data-centric changes plateau (more data stops helping), switch to model-centric.
When model-centric changes plateau (architecture/loss changes stop helping), switch
back to data-centric. Each leg unblocks the other.

## Output Format
Return a JSON array of ideas. Each idea is an object with:
- "title": short descriptive name (string)
- "hypothesis": why this might work, referencing evidence from the context (string)
- "config": YAML-compatible dict with experiment config (object)
- "priority": "critical" | "high" | "medium" | "low" (string, optional)
- "category": free-form label like "architecture", "hyperparameter", "loss" (string, optional)
- "approach_family": one of: "architecture", "training_config", "data", "infrastructure", "optimization", "regularization", "ensemble", "other" (string, required)
- "parent": "none" or an existing idea ID if building on a previous idea (string, optional)

Note: idea IDs are auto-generated as 6-char content hashes (e.g. "idea-a7f3b2").
You do NOT need to assign IDs — just provide the config and orze will hash it.

Example:
```json
[
  {
    "title": "Larger learning rate with cosine schedule",
    "hypothesis": "Current best uses lr=1e-4. A 3x larger lr with cosine decay may converge faster and find a better minimum.",
    "config": {
      "model": {"type": "resnet50", "pretrained": true},
      "training": {"lr": 3e-4, "scheduler": "cosine", "epochs": 20}
    },
    "priority": "high",
    "category": "hyperparameter",
    "approach_family": "training_config",
    "parent": "none"
  }
]
```

Output ONLY the JSON array, no markdown fences or extra text.
"""
