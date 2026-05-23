"""
Orchestrator Agent
Runs the full pipeline in sequence: all 8 agents chained automatically.
This is registered as a single MCP tool: run_pipeline()
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agents import (
    data_intake_agent,
    quality_auditor_agent,
    type_inference_agent,
    missing_strategy_agent,
    outlier_analyst_agent,
    feature_engineer_agent,
    feature_selector_agent,
    pipeline_validator_agent,
)


def run(
    file_path: str,
    file_type: str = "csv",
    target_col: str = None,
    outlier_action: str = "clip",
    polynomial: bool = False,
    use_llm: bool = False,
    drop_threshold: float = 0.7,
    correlation_threshold: float = 0.95,
) -> dict:
    """
    Runs all 8 agents in sequence and returns a consolidated report.
    """
    log = []
    results = {}

    def step(name, fn, **kwargs):
        print(f"  [{name}] running...")
        result = fn(**kwargs)
        status = result.get("status", "unknown")
        log.append({"step": name, "status": status})
        results[name] = result
        if status == "error":
            raise RuntimeError(f"{name} failed: {result.get('error')}")
        return result

    try:
        # 1. Ingest
        r = step("ingest_data", data_intake_agent.run,
                 file_path=file_path, file_type=file_type)
        sid = r["session_id"]

        # 2. Quality Audit
        step("audit_quality", quality_auditor_agent.run,
             session_id=sid, use_llm=use_llm)

        # 3. Type Inference
        step("infer_types", type_inference_agent.run,
             session_id=sid, cast_types=True, use_llm=use_llm)

        # 4. Missing Strategy
        step("handle_missing", missing_strategy_agent.run,
             session_id=sid, use_llm=use_llm, drop_threshold=drop_threshold)

        # 5. Outlier Detection
        step("detect_outliers", outlier_analyst_agent.run,
             session_id=sid, action=outlier_action)

        # 6. Feature Engineering
        step("engineer_features", feature_engineer_agent.run,
             session_id=sid, use_llm=use_llm, polynomial=polynomial)

        # 7. Feature Selection
        step("select_features", feature_selector_agent.run,
             session_id=sid, target_col=target_col,
             correlation_threshold=correlation_threshold, use_llm=use_llm)

        # 8. Validate
        validation = step("validate_pipeline", pipeline_validator_agent.run,
                          session_id=sid, use_llm=use_llm)

        return {
            "status": "ok",
            "session_id": sid,
            "pipeline_status": validation.get("pipeline_status"),
            "final_shape": validation.get("final_shape"),
            "steps_completed": [s["step"] for s in log],
            "warnings": validation.get("warnings", []),
            "validation_errors": validation.get("validation_errors", []),
            "stats": validation.get("stats", {}),
            "llm_summary": validation.get("llm_summary"),
        }

    except RuntimeError as e:
        return {
            "status": "error",
            "error": str(e),
            "steps_completed": [s["step"] for s in log if s["status"] == "ok"],
            "failed_at": next((s["step"] for s in log if s["status"] == "error"), None),
        }
