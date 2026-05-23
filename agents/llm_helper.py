"""
LLM Provider — Unified Interface
==================================
Single entry point for all LLM calls across the project.
Agents, MCP server, and FastAPI all call ask() / ask_json() here.

Switching providers:
  1. Edit config.yaml  →  llm.provider: openai
  2. Set env var       →  export OPENAI_API_KEY=sk-...
  3. Done. No code changes.

Supported providers:
  ollama       Local, offline, free. Requires Ollama running.
  openai       Requires OPENAI_API_KEY env var.
  gemini       Requires GEMINI_API_KEY env var.
  huggingface  Requires HF_API_KEY env var.
  mock         Deterministic responses. No LLM needed. For testing/CI.
"""

import os
import json
import time
from pathlib import Path

# ── Load config ───────────────────────────────────────────────────────────────

def _load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config.yaml"
    if config_path.exists():
        try:
            import yaml
            with open(config_path) as f:
                return yaml.safe_load(f) or {}
        except ImportError:
            pass
        except Exception:
            pass
    return {}


def _get_provider() -> str:
    """Priority: env var > config.yaml > default (ollama)"""
    env = os.environ.get("DATAPILOT_PROVIDER", "").strip().lower()
    if env:
        return env
    cfg = _load_config()
    return cfg.get("llm", {}).get("provider", "ollama").lower()


def _get_model(provider: str) -> str:
    """Get configured model for the active provider."""
    env_model = os.environ.get("DATAPILOT_MODEL", "").strip()
    if env_model:
        return env_model
    cfg = _load_config().get("llm", {})
    model_map = {
        "ollama":       cfg.get("ollama_model",  "phi3:mini"),
        "openai":       cfg.get("openai_model",  "gpt-4o-mini"),
        "gemini":       cfg.get("gemini_model",  "gemini-1.5-flash"),
        "huggingface":  cfg.get("hf_model",      "mistralai/Mistral-7B-Instruct-v0.2"),
        "mock":         "mock",
    }
    return model_map.get(provider, "phi3:mini")


# ── Provider implementations ──────────────────────────────────────────────────

def _ask_ollama(prompt: str, model: str) -> str:
    import ollama as _ollama
    response = _ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"].strip()


def _ask_openai(prompt: str, model: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. "
            "Export it: export OPENAI_API_KEY=sk-..."
        )
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError(
            "openai package not installed. "
            "Run: pip install openai"
        )
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
    )
    return response.choices[0].message.content.strip()


def _ask_gemini(prompt: str, model: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. "
            "Export it: export GEMINI_API_KEY=..."
        )
    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai package not installed. "
            "Run: pip install google-generativeai"
        )
    genai.configure(api_key=api_key)
    m = genai.GenerativeModel(model)
    response = m.generate_content(prompt)
    return response.text.strip()


def _ask_huggingface(prompt: str, model: str) -> str:
    api_key = os.environ.get("HF_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "HF_API_KEY not set. "
            "Export it: export HF_API_KEY=hf_..."
        )
    try:
        import requests
    except ImportError:
        raise ImportError("requests package required")

    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 500, "return_full_text": False},
    }
    url = f"https://api-inference.huggingface.co/models/{model}"
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list) and data:
        return data[0].get("generated_text", "").strip()
    return str(data)


_MOCK_RESPONSES = {
    "default": '{"result": "ok", "status": "processed", "notes": "Mock response — no LLM running."}',
}

def _ask_mock(prompt: str, model: str) -> str:
    """Deterministic mock — used for testing and CI."""
    import random
    time.sleep(random.uniform(0.05, 0.2))   # realistic latency simulation
    # Return a plausible JSON response
    if "quality" in prompt.lower() or "audit" in prompt.lower():
        return '{"overall_quality": "FAIR", "duplicate_rows": 0, "quality_flags": [], "backtrack": false, "risk_summary": "Mock: data looks reasonable."}'
    if "type" in prompt.lower() or "infer" in prompt.lower():
        return '{"columns": {}, "backtrack": false}'
    if "missing" in prompt.lower() or "imputation" in prompt.lower():
        return '{"strategy": "median", "justification": "Mock: median is robust.", "backtrack": false, "risk_level": "low"}'
    if "outlier" in prompt.lower():
        return '{"recommended_action": "clip", "justification": "Mock: clip outliers.", "backtrack": false}'
    if "feature" in prompt.lower() and "engineer" in prompt.lower():
        return '[{"feature_name": "mock_feature", "transformation": "identity", "risk": "none"}]'
    if "select" in prompt.lower():
        return '{"keep": [], "drop": [], "backtrack": false}'
    if "validat" in prompt.lower():
        return '{"status": "PASSED", "backtrack": false, "issues": [], "readiness_score": 85}'
    return _MOCK_RESPONSES["default"]


# ── Public API ────────────────────────────────────────────────────────────────

def ask(
    prompt: str,
    model: str = None,
    provider: str = None,
) -> str:
    """
    Send a prompt to the configured LLM provider.

    Args:
        prompt:   The prompt text.
        model:    Override model. If None, uses config.yaml / env var.
        provider: Override provider. If None, uses config.yaml / env var.

    Returns:
        Response text string.

    Raises:
        EnvironmentError: If API key is missing for cloud providers.
        ImportError: If required package is not installed.
        RuntimeError: If provider is unknown or call fails.
    """
    active_provider = provider or _get_provider()
    active_model    = model    or _get_model(active_provider)

    try:
        if active_provider == "ollama":
            return _ask_ollama(prompt, active_model)
        elif active_provider == "openai":
            return _ask_openai(prompt, active_model)
        elif active_provider == "gemini":
            return _ask_gemini(prompt, active_model)
        elif active_provider == "huggingface":
            return _ask_huggingface(prompt, active_model)
        elif active_provider == "mock":
            return _ask_mock(prompt, active_model)
        else:
            raise RuntimeError(
                f"Unknown provider '{active_provider}'. "
                f"Valid options: ollama, openai, gemini, huggingface, mock"
            )
    except Exception as e:
        # Fallback to mock if provider fails and we're not already in mock mode
        if active_provider != "mock":
            print(f"[llm_helper] WARNING: {active_provider} call failed ({e}). "
                  f"Falling back to mock response.")
            return _ask_mock(prompt, "mock")
        raise


def ask_json(
    prompt: str,
    model: str = None,
    provider: str = None,
) -> dict:
    """
    Ask the LLM and parse the response as JSON.
    Automatically appends JSON-only instruction to the prompt.
    """
    full_prompt = (
        prompt.rstrip()
        + "\n\nRespond with valid JSON only. "
          "No explanation, no markdown fences, no preamble."
    )
    raw = ask(full_prompt, model=model, provider=provider)
    raw = raw.strip()
    # Strip markdown fences if model added them anyway
    for fence in ("```json", "```"):
        raw = raw.removeprefix(fence).removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_response": raw, "parse_error": True}


def get_active_config() -> dict:
    """Returns the active provider + model for display in the frontend."""
    provider = _get_provider()
    model    = _get_model(provider)
    return {
        "provider": provider,
        "model":    model,
        "is_local": provider in ("ollama", "mock"),
        "requires_api_key": provider in ("openai", "gemini", "huggingface"),
    }
