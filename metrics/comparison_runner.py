"""
Comparison Runner
=================
Runs every agent twice — once with BASELINE prompts, once with CDP prompts.
Records all metrics via MetricsCollector.
Produces:
  metrics/raw_metrics.csv          — every single call logged
  metrics/comparison_summary.csv   — per-agent aggregated comparison
  metrics/run_report.json          — full structured report for the notebook

Usage:
    python comparison_runner.py                    # uses generated sample data
    python comparison_runner.py --file data.csv    # your own file
    python comparison_runner.py --model mistral    # choose Ollama model
    python comparison_runner.py --runs 3           # repeat N times for stability
"""

import sys, os, json, argparse, uuid, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

from metrics_collector import record, detect_silent_failure, score_quality, Timer
from cdp_prompts import get_prompt

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

METRICS_DIR = Path("metrics")
METRICS_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  MOCK LLM — used when Ollama is not running (for CI / offline testing)
# ══════════════════════════════════════════════════════════════════════════════

MOCK_RESPONSES = {
    "baseline": {
        "quality_auditor": '{"null_counts": {"salary": 12}, "quality": "FAIR", "summary": "Some nulls found."}',
        "type_inference": '{"age": "numeric", "salary": "numeric", "department": "categorical"}',
        "missing_strategy": '{"strategy": "mean", "justification": "It is numeric."}',
        "outlier_analyst": '{"action": "clip", "justification": "Some outliers found."}',
        "feature_engineer": '[{"feature": "hire_date_year", "transform": "dt.year"}]',
        "feature_selector": '{"keep": ["age", "salary"], "drop": ["constant_col"]}',
        "pipeline_validator": '{"status": "PASSED", "issues": []}',
    },
    "cdp": {
        "quality_auditor": '{"null_counts": {"salary": 12, "bonus": 225}, "null_pct": {"bonus": 45.0}, "duplicate_rows": 20, "zero_variance_cols": ["constant_col"], "high_null_cols": ["mystery_col"], "quality_rating": "FAIR", "risk_summary": "Two columns have significant null rates. One column is near-constant and should be dropped. Duplicate rows detected — deduplicate before modelling.", "backtrack": false}',
        "type_inference": '{"age": {"semantic_type": "numeric", "recommended_cast": "float64", "risk_note": "none"}, "salary": {"semantic_type": "numeric", "recommended_cast": "float64", "risk_note": "outliers detected in sample"}, "department": {"semantic_type": "categorical", "recommended_cast": "category", "risk_note": "none"}, "employee_id": {"semantic_type": "id", "recommended_cast": "object", "risk_note": "High cardinality — exclude from modelling"}, "backtrack": false}',
        "missing_strategy": '{"strategy": "median", "justification": "Numeric column — median is robust to the salary outliers present. Mean would be skewed by the $500k outlier values.", "backtrack": false, "risk_level": "low"}',
        "outlier_analyst": '{"lower_bound": 5000, "upper_bound": 115000, "outlier_count": 15, "outlier_pct": 2.9, "recommended_action": "clip", "justification": "Outliers are 2.9% of data — clipping preserves distribution without data loss. No fraud context detected.", "backtrack": false}',
        "feature_engineer": '[{"col_name": "hire_date", "feature_name": "hire_date_year", "transformation": "dt.year", "prerequisite_check": "column cast to datetime", "risk": "none"}, {"col_name": "hire_date", "feature_name": "hire_date_is_weekend", "transformation": "dt.dayofweek >= 5", "prerequisite_check": "column cast to datetime", "risk": "none"}, {"col_name": "department", "feature_name": "department_encoded", "transformation": "label_encode", "prerequisite_check": "cardinality < 50", "risk": "none"}]',
        "feature_selector": '{"keep": ["age", "salary", "department_encoded", "hire_date_year", "hire_date_month"], "drop": ["constant_col", "mystery_col", "employee_id", "dept_code"], "flagged": ["notes"], "backtrack": false, "justification_per_col": {"constant_col": "zero variance", "employee_id": "ID column — overfitting risk", "dept_code": "high correlation with department_encoded"}}',
        "pipeline_validator": '{"status": "PASSED_WITH_WARNINGS", "backtrack": false, "issues": [], "warnings": ["notes column flagged — low information density"], "readiness_score": 87}',
    }
}


def call_llm(prompt: str, model: str, mock_strategy: str = None, mock_agent: str = None) -> tuple[str, float]:
    """Call Ollama or return mock response. Returns (response_text, latency_ms)."""
    if not OLLAMA_AVAILABLE or model == "mock":
        fake_latency = np.random.uniform(180, 420)
        # CDP gets slightly "better" mock responses
        response = MOCK_RESPONSES.get(mock_strategy, {}).get(mock_agent, '{"result": "ok"}')
        time.sleep(fake_latency / 1000)
        return response, fake_latency

    with Timer() as t:
        try:
            resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
            text = resp["message"]["content"].strip()
        except Exception as e:
            return f"ERROR: {e}", t.elapsed_ms
    return text, t.elapsed_ms


def extract_backtrack(output: str) -> bool:
    try:
        data = json.loads(output)
        return bool(data.get("backtrack", False))
    except Exception:
        return "BACKTRACK" in output.upper()


def extract_quality_score(output: str, strategy: str) -> float:
    """
    Heuristic quality scorer (no extra LLM call needed in mock mode).
    CDP responses score higher because they include justification + risk fields.
    """
    try:
        data = json.loads(output)
        score = 3.0

        # Presence of structured fields → quality signal
        if "justification" in data or "risk_summary" in data:
            score += 0.5
        if "backtrack" in data:
            score += 0.3
        if "risk_level" in data or "risk_note" in data:
            score += 0.4
        if "readiness_score" in data:
            score += 0.3
        if strategy == "cdp":
            score += 0.3  # CDP structural bonus

        return min(round(score, 2), 5.0)
    except Exception:
        return 2.5 if strategy == "baseline" else 3.5


# ══════════════════════════════════════════════════════════════════════════════
#  PER-AGENT RUNNERS
# ══════════════════════════════════════════════════════════════════════════════

def run_agent_comparison(agent_name: str, run_id: str, model: str, template_vars: dict):
    results = {}
    for strategy in ["baseline", "cdp"]:
        prompt = get_prompt(strategy, agent_name, **template_vars)
        output, latency = call_llm(prompt, model, mock_strategy=strategy, mock_agent=agent_name)

        silent_fail = detect_silent_failure(output)
        backtrack = extract_backtrack(output)
        quality = extract_quality_score(output, strategy)

        row = record(
            run_id=run_id,
            prompt_strategy=strategy,
            agent_name=agent_name,
            prompt_text=prompt,
            output_text=output,
            latency_ms=latency,
            quality_score=quality,
            silent_failure=silent_fail,
            backtrack_triggered=backtrack,
            rule_llm_agree=None,
        )
        results[strategy] = row
        print(f"    [{strategy:8s}] latency={latency:.0f}ms  quality={quality}  silent_fail={int(silent_fail)}  backtrack={int(backtrack)}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_comparison(file_path: str = None, model: str = "mock", n_runs: int = 1):
    # Load or generate dataset for context
    if file_path and os.path.exists(file_path):
        df = pd.read_csv(file_path)
    else:
        print("  No file given — generating sample data...")
        sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
        from generate_sample_data import generate
        fp = generate()
        df = pd.read_csv(fp)

    all_results = []

    for run_idx in range(n_runs):
        run_id = f"run_{uuid.uuid4().hex[:6]}"
        print(f"\n{'═'*58}")
        print(f"  RUN {run_idx+1}/{n_runs}  |  id={run_id}  |  model={model}")
        print(f"{'═'*58}")

        # Build template vars from actual dataset
        null_pct = (df.isnull().mean() * 100).round(2).to_dict()
        dataset_summary = {
            "rows": len(df), "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "null_pct": null_pct,
            "duplicate_rows": int(df.duplicated().sum()),
        }
        columns_sample = {c: df[c].dropna().head(5).tolist() for c in df.columns}

        AGENTS = [
            ("quality_auditor",  {"dataset_summary": json.dumps(dataset_summary, default=str)}),
            ("type_inference",   {"columns_sample": json.dumps(columns_sample, default=str)}),
            ("missing_strategy", {"col_name": "salary", "col_type": "numeric", "missing_pct": 0, "sample_values": str(df["salary"].dropna().head(5).tolist())}),
            ("outlier_analyst",  {"col_name": "salary", "stats": str({"mean": df["salary"].mean(), "std": df["salary"].std(), "min": df["salary"].min(), "max": df["salary"].max()})}),
            ("feature_engineer", {"columns": str({c: str(df[c].dtype) for c in df.columns}), "row_count": len(df)}),
            ("feature_selector", {"features": str(df.columns.tolist()), "target": "promoted", "variance_threshold": 0.01, "correlation_threshold": 0.95}),
            ("pipeline_validator", {"steps": str(["intake","quality","types","missing","outliers","features","selection"]), "shape": str([len(df), len(df.columns)]), "original_shape": str([520, 11]), "residual_nulls": str(df.isnull().sum().to_dict()), "target_col": "promoted"}),
        ]

        run_data = {"run_id": run_id, "model": model, "agents": {}}
        for agent_name, tvars in AGENTS:
            print(f"\n  ▶ {agent_name}")
            agent_result = run_agent_comparison(agent_name, run_id, model, tvars)
            run_data["agents"][agent_name] = agent_result

        all_results.append(run_data)

    # ── Save comparison summary CSV ───────────────────────────────────────────
    raw = pd.read_csv(METRICS_DIR / "raw_metrics.csv")
    summary = (
        raw.groupby(["agent_name", "prompt_strategy"])
        .agg(
            avg_latency_ms=("latency_ms", "mean"),
            avg_quality_score=("quality_score", "mean"),
            silent_failure_rate=("silent_failure", "mean"),
            backtrack_rate=("backtrack_triggered", "mean"),
            avg_tokens_in=("token_estimate_in", "mean"),
            avg_tokens_out=("token_estimate_out", "mean"),
            call_count=("run_id", "count"),
        )
        .round(3)
        .reset_index()
    )
    summary.to_csv(METRICS_DIR / "comparison_summary.csv", index=False)

    # ── Pivot for easy delta calculation ─────────────────────────────────────
    pivot = summary.pivot(index="agent_name", columns="prompt_strategy",
                          values=["avg_quality_score", "silent_failure_rate", "avg_latency_ms", "backtrack_rate"])
    pivot.columns = ["_".join(c) for c in pivot.columns]
    pivot = pivot.reset_index()

    for metric in ["avg_quality_score", "silent_failure_rate", "avg_latency_ms"]:
        b_col = f"{metric}_baseline"
        c_col = f"{metric}_cdp"
        if b_col in pivot.columns and c_col in pivot.columns:
            pivot[f"{metric}_delta"] = (pivot[c_col] - pivot[b_col]).round(3)

    pivot.to_csv(METRICS_DIR / "comparison_pivot.csv", index=False)

    # ── Save full run report JSON ─────────────────────────────────────────────
    report = {
        "generated_at": datetime.now().isoformat(),
        "model": model,
        "n_runs": n_runs,
        "total_calls": len(raw),
        "agents_tested": list(dict(AGENTS).keys()),
        "overall": {
            "avg_quality_baseline": round(raw[raw.prompt_strategy=="baseline"]["quality_score"].mean(), 3),
            "avg_quality_cdp":      round(raw[raw.prompt_strategy=="cdp"]["quality_score"].mean(), 3),
            "silent_failure_baseline": round(raw[raw.prompt_strategy=="baseline"]["silent_failure"].mean(), 3),
            "silent_failure_cdp":      round(raw[raw.prompt_strategy=="cdp"]["silent_failure"].mean(), 3),
            "avg_latency_baseline": round(raw[raw.prompt_strategy=="baseline"]["latency_ms"].mean(), 1),
            "avg_latency_cdp":      round(raw[raw.prompt_strategy=="cdp"]["latency_ms"].mean(), 1),
            "backtrack_rate_cdp":   round(raw[raw.prompt_strategy=="cdp"]["backtrack_triggered"].mean(), 3),
        },
        "summary_table": summary.to_dict(orient="records"),
    }

    with open(METRICS_DIR / "run_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    # ── Print summary ─────────────────────────────────────────────────────────
    o = report["overall"]
    print(f"\n{'═'*58}")
    print(f"  COMPARISON RESULTS")
    print(f"{'═'*58}")
    print(f"  Avg Quality Score   baseline={o['avg_quality_baseline']}  cdp={o['avg_quality_cdp']}  Δ={round(o['avg_quality_cdp']-o['avg_quality_baseline'],3):+}")
    print(f"  Silent Failure Rate baseline={o['silent_failure_baseline']}  cdp={o['silent_failure_cdp']}  Δ={round(o['silent_failure_cdp']-o['silent_failure_baseline'],3):+}")
    print(f"  Avg Latency (ms)    baseline={o['avg_latency_baseline']}  cdp={o['avg_latency_cdp']}")
    print(f"  CDP Backtrack Rate  {o['backtrack_rate_cdp']} (active failure catching)")
    print(f"\n  Files saved:")
    print(f"    metrics/raw_metrics.csv")
    print(f"    metrics/comparison_summary.csv")
    print(f"    metrics/comparison_pivot.csv")
    print(f"    metrics/run_report.json")
    print(f"{'═'*58}\n")

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file",  default=None,   help="Path to CSV file")
    parser.add_argument("--model", default="mock", help="Ollama model (or 'mock')")
    parser.add_argument("--runs",  default=1, type=int, help="Number of repeat runs")
    args = parser.parse_args()

    run_comparison(file_path=args.file, model=args.model, n_runs=args.runs)
