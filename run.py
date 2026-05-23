#!/usr/bin/env python3
"""
CLI Test Runner for Data Pipeline MCP
Run the full pipeline from terminal — no Claude Desktop needed.

Usage:
    python run.py                          # generates sample data + runs full pipeline
    python run.py --file /path/to/data.csv --target promoted
    python run.py --file data.csv --llm    # enable Ollama LLM reasoning
    python run.py --step                   # run step-by-step (interactive)
"""

import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(__file__))

from generate_sample_data import generate
from agents.orchestrator_agent import run as run_pipeline
from agents import (
    data_intake_agent, quality_auditor_agent, type_inference_agent,
    missing_strategy_agent, outlier_analyst_agent, feature_engineer_agent,
    feature_selector_agent, pipeline_validator_agent,
)

CYAN  = "\033[96m"
GREEN = "\033[92m"
YELLOW= "\033[93m"
RED   = "\033[91m"
BOLD  = "\033[1m"
RESET = "\033[0m"

def banner(text):
    print(f"\n{BOLD}{CYAN}{'─'*55}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'─'*55}{RESET}")

def ok(label, data):
    print(f"{GREEN}  ✓ {label}{RESET}")
    for k, v in data.items():
        if k not in ("status", "session_id"):
            print(f"    {k}: {v}")

def warn(msg):
    print(f"{YELLOW}  ⚠ {msg}{RESET}")

def err(msg):
    print(f"{RED}  ✗ {msg}{RESET}")

def print_result(result: dict):
    if result.get("status") == "error":
        err(result.get("error", "Unknown error"))
    else:
        for k, v in result.items():
            if k in ("status", "session_id"):
                continue
            print(f"    {BOLD}{k}{RESET}: {json.dumps(v, default=str)[:120]}")


def run_full(file_path, file_type, target_col, use_llm, outlier_action, polynomial):
    banner("Running Full Pipeline via Orchestrator")
    print(f"  File    : {file_path}")
    print(f"  Target  : {target_col or 'not set'}")
    print(f"  LLM     : {'enabled (Ollama)' if use_llm else 'disabled (rule-based)'}")

    result = run_pipeline(
        file_path=file_path,
        file_type=file_type,
        target_col=target_col,
        use_llm=use_llm,
        outlier_action=outlier_action,
        polynomial=polynomial,
    )

    banner("Pipeline Result")
    status = result.get("pipeline_status", result.get("status"))
    color = GREEN if "PASS" in str(status) else (YELLOW if "WARN" in str(status) else RED)
    print(f"  Status        : {color}{BOLD}{status}{RESET}")
    print(f"  Session ID    : {result.get('session_id', 'N/A')}")
    print(f"  Final Shape   : {result.get('final_shape')}")
    print(f"  Steps Done    : {result.get('steps_completed')}")

    warnings = result.get("warnings", [])
    errors = result.get("validation_errors", [])
    if errors:
        for e in errors: err(e)
    if warnings:
        for w in warnings: warn(w)

    stats = result.get("stats", {})
    if stats:
        print(f"\n  {BOLD}Stats:{RESET}")
        for k, v in stats.items():
            print(f"    {k}: {v}")

    if result.get("llm_summary"):
        print(f"\n  {BOLD}LLM Summary:{RESET}")
        print(f"  {result['llm_summary']}")

    return result.get("session_id")


def run_step_by_step(file_path, file_type, target_col, use_llm):
    banner("Step-by-Step Pipeline")

    steps = [
        ("1. Data Intake",      lambda sid: data_intake_agent.run(file_path, file_type, sid)),
        ("2. Quality Audit",    lambda sid: quality_auditor_agent.run(sid, use_llm)),
        ("3. Type Inference",   lambda sid: type_inference_agent.run(sid, True, use_llm)),
        ("4. Handle Missing",   lambda sid: missing_strategy_agent.run(sid, use_llm)),
        ("5. Detect Outliers",  lambda sid: outlier_analyst_agent.run(sid, "clip")),
        ("6. Engineer Features",lambda sid: feature_engineer_agent.run(sid, use_llm)),
        ("7. Select Features",  lambda sid: feature_selector_agent.run(sid, target_col, use_llm=use_llm)),
        ("8. Validate Pipeline",lambda sid: pipeline_validator_agent.run(sid, use_llm)),
    ]

    sid = None
    for name, fn in steps:
        banner(name)
        result = fn(sid)
        if result.get("status") == "error":
            err(result.get("error"))
            sys.exit(1)
        if sid is None:
            sid = result.get("session_id")
            print(f"  Session ID: {BOLD}{sid}{RESET}")
        print_result(result)
        input(f"\n  {YELLOW}Press Enter to continue...{RESET}")

    return sid


def main():
    parser = argparse.ArgumentParser(description="Data Pipeline MCP CLI Runner")
    parser.add_argument("--file",    default=None,    help="Path to input CSV/JSON file")
    parser.add_argument("--type",    default="csv",   help="File type: csv or json")
    parser.add_argument("--target",  default=None,    help="Target column name")
    parser.add_argument("--llm",     action="store_true", help="Enable Ollama LLM reasoning")
    parser.add_argument("--outlier", default="clip",  choices=["flag","clip","drop"])
    parser.add_argument("--poly",    action="store_true", help="Enable polynomial features")
    parser.add_argument("--step",    action="store_true", help="Run step-by-step interactively")
    args = parser.parse_args()

    # Generate sample data if no file given
    file_path = args.file
    if not file_path:
        banner("Generating Sample Messy Data")
        file_path = generate()

    if args.step:
        run_step_by_step(file_path, args.type, args.target, args.llm)
    else:
        run_full(file_path, args.type, args.target, args.llm, args.outlier, args.poly)

    print(f"\n{GREEN}{BOLD}✓ Done. Session data saved in /tmp/data_pipeline_sessions/{RESET}\n")


if __name__ == "__main__":
    main()
