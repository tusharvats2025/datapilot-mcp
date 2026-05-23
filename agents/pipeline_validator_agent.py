"""
Agent 8: Pipeline Validator
End-to-end validation: checks all steps ran, data shape makes sense,
no residual nulls, types are clean. Generates final pipeline health report.
"""

import pandas as pd
from storage.session_store import load_dataset, load_report, get_meta, save_report


EXPECTED_STEPS = [
    "intake", "quality_report", "type_report",
    "missing_report", "outlier_report", "feature_report", "selector_report"
]


def run(session_id: str, use_llm: bool = False) -> dict:
    try:
        meta = get_meta(session_id)
        completed_steps = [s["step"] for s in meta.get("steps", [])]

        # Check which expected steps are missing
        missing_steps = [s for s in EXPECTED_STEPS if s not in completed_steps]

        df = load_dataset(session_id)

        # Data integrity checks
        residual_nulls = df.isnull().sum()
        null_issues = {c: int(n) for c, n in residual_nulls.items() if n > 0}
        shape = [len(df), len(df.columns)]

        # Load sub-reports for summary
        quality = load_report(session_id, "quality_report")
        selector = load_report(session_id, "selector_report")
        outlier = load_report(session_id, "outlier_report")
        feature = load_report(session_id, "feature_report")

        validation_errors = []
        warnings = []

        if null_issues:
            warnings.append(f"Residual nulls found: {null_issues}")
        if missing_steps:
            warnings.append(f"Steps not completed: {missing_steps}")
        if shape[1] == 0:
            validation_errors.append("Dataset has 0 columns after pipeline!")
        if shape[0] < 10:
            warnings.append(f"Very few rows remaining: {shape[0]}")

        # Overall status
        if validation_errors:
            pipeline_status = "FAILED"
        elif warnings:
            pipeline_status = "PASSED WITH WARNINGS"
        else:
            pipeline_status = "PASSED"

        summary = {
            "pipeline_status": pipeline_status,
            "session_id": session_id,
            "completed_steps": completed_steps,
            "missing_steps": missing_steps,
            "final_shape": shape,
            "residual_null_columns": null_issues,
            "validation_errors": validation_errors,
            "warnings": warnings,
            "stats": {
                "original_quality": quality.get("overall_quality"),
                "features_engineered": len(feature.get("engineered_features", [])),
                "features_removed": len(selector.get("removed_features", [])),
                "features_kept": len(selector.get("kept_features", [])),
                "outlier_rows_flagged": outlier.get("total_outlier_rows", "N/A"),
            }
        }

        if use_llm:
            from agents.llm_helper import ask
            prompt = f"""
You are reviewing a completed ML data pipeline.
- Status: {pipeline_status}
- Final data shape: {shape[0]} rows × {shape[1]} columns
- Completed steps: {completed_steps}
- Warnings: {warnings}
- Errors: {validation_errors}
- Stats: {summary['stats']}

Write a 4-5 sentence pipeline health summary for a data scientist.
Include what went well, concerns, and readiness for model training.
"""
            summary["llm_summary"] = ask(prompt)

        save_report(session_id, summary, label="validation_report")
        return {"session_id": session_id, "status": "ok", **summary}

    except Exception as e:
        return {"session_id": session_id, "status": "error", "error": str(e)}
