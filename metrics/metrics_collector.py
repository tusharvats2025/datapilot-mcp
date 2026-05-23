"""
MetricsCollector
================
Instruments every agent call and records:
  - prompt_strategy   : 'baseline' | 'cdp'
  - agent_name        : which agent ran
  - prompt_text       : what was sent to Ollama
  - output_text       : what Ollama returned
  - latency_ms        : wall-clock time for LLM call
  - token_estimate    : rough token count (chars / 4)
  - quality_score     : LLM self-eval 1–5 (optional)
  - silent_failure    : bool — output looks plausible but contains known bad pattern
  - backtrack_triggered: bool — CDP backtrack condition fired
  - rule_llm_agree    : bool — rule-based and LLM decisions match
  - error             : any exception string

All records saved to:  metrics/raw_metrics.csv  (appended, never overwritten)
"""

import csv
import time
import os
import re
from pathlib import Path
from datetime import datetime

METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)
RAW_CSV = METRICS_DIR / "raw_metrics.csv"

FIELDNAMES = [
    "run_id", "timestamp", "prompt_strategy", "agent_name",
    "prompt_text", "output_text",
    "latency_ms", "token_estimate_in", "token_estimate_out",
    "quality_score", "silent_failure", "backtrack_triggered",
    "rule_llm_agree", "error",
]


def _ensure_header():
    if not RAW_CSV.exists():
        with open(RAW_CSV, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def record(
    run_id: str,
    prompt_strategy: str,
    agent_name: str,
    prompt_text: str,
    output_text: str,
    latency_ms: float,
    quality_score: float = None,
    silent_failure: bool = False,
    backtrack_triggered: bool = False,
    rule_llm_agree: bool = None,
    error: str = "",
):
    _ensure_header()
    row = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "prompt_strategy": prompt_strategy,
        "agent_name": agent_name,
        "prompt_text": prompt_text[:800],          # truncate for CSV sanity
        "output_text": output_text[:800],
        "latency_ms": round(latency_ms, 2),
        "token_estimate_in": len(prompt_text) // 4,
        "token_estimate_out": len(output_text) // 4,
        "quality_score": quality_score,
        "silent_failure": int(silent_failure),
        "backtrack_triggered": int(backtrack_triggered),
        "rule_llm_agree": "" if rule_llm_agree is None else int(rule_llm_agree),
        "error": error,
    }
    with open(RAW_CSV, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writerow(row)
    return row


# ── Silent failure detector ───────────────────────────────────────────────────
SILENT_FAILURE_PATTERNS = [
    r"\bmean\b.*\bfraud\b",          # mean imputation on fraud col
    r"one.hot.*high.cardinality",    # one-hot on high-cardinality col
    r"fit.*full.*dataset",           # scaler fit on full data = leakage
    r"drop.*low.*correlation",       # dropping cols by solo correlation
    r"error.*none",                  # error swallowed as None
]

def detect_silent_failure(output_text: str) -> bool:
    txt = output_text.lower()
    return any(re.search(p, txt) for p in SILENT_FAILURE_PATTERNS)


# ── Quality scorer (asks Ollama to rate 1–5) ──────────────────────────────────
def score_quality(agent_name: str, prompt: str, output: str, model: str = "llama3") -> float:
    """Ask Ollama to self-evaluate output quality. Returns float 1.0–5.0."""
    try:
        import ollama
        eval_prompt = f"""
You are a strict ML engineering reviewer.

Agent: {agent_name}
Prompt given: {prompt[:400]}
Output produced: {output[:400]}

Rate the output quality from 1 to 5:
5 = correct, complete, no ML mistakes
4 = mostly correct, minor gaps
3 = partially correct, some errors
2 = significant errors or missing info
1 = wrong, harmful, or silent failure

Reply with a single number only. No explanation.
"""
        resp = ollama.chat(model=model, messages=[{"role": "user", "content": eval_prompt}])
        score_text = resp["message"]["content"].strip()
        return float(re.search(r"[1-5]", score_text).group())
    except Exception:
        return None


# ── Timer context manager ─────────────────────────────────────────────────────
class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
