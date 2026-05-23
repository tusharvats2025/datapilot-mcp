"""
Agent 2: Quality Auditor
Produces a quality report: nulls, duplicates, zero-variance cols, data shape.
Optionally uses Ollama to generate a plain-English summary.
"""

import pandas as pd
from storage.session_store import load_dataset, save_report


def run(session_id: str, use_llm: bool = False) -> dict:
    try:
        df = load_dataset(session_id)

        null_counts = df.isnull().sum().to_dict()
        null_pct = (df.isnull().mean() * 100).round(2).to_dict()
        duplicate_rows = int(df.duplicated().sum())
        zero_variance_cols = [c for c in df.select_dtypes(include="number").columns
                              if df[c].nunique() <= 1]

        quality_flags = []
        for col, pct in null_pct.items():
            if pct > 50:
                quality_flags.append(f"HIGH NULLS: '{col}' is {pct}% missing")
        if duplicate_rows > 0:
            quality_flags.append(f"DUPLICATES: {duplicate_rows} duplicate rows found")
        for col in zero_variance_cols:
            quality_flags.append(f"ZERO VARIANCE: '{col}' has only one unique value")

        report = {
            "rows": len(df),
            "columns": len(df.columns),
            "duplicate_rows": duplicate_rows,
            "null_counts": null_counts,
            "null_percentage": null_pct,
            "zero_variance_columns": zero_variance_cols,
            "quality_flags": quality_flags,
            "overall_quality": "POOR" if len(quality_flags) > 3 else
                               "FAIR" if len(quality_flags) > 0 else "GOOD",
        }

        if use_llm:
            from agents.llm_helper import ask
            prompt = f"""
You are a data quality expert. Here is a quality audit report for a dataset:
- Rows: {report['rows']}, Columns: {report['columns']}
- Duplicate rows: {report['duplicate_rows']}
- Columns with >50% nulls: {[c for c, p in null_pct.items() if p > 50]}
- Quality flags: {quality_flags}

Write a concise 3-sentence summary of the data quality and key concerns.
"""
            report["llm_summary"] = ask(prompt)

        save_report(session_id, report, label="quality_report")
        return {"session_id": session_id, "status": "ok", **report}

    except Exception as e:
        return {"session_id": session_id, "status": "error", "error": str(e)}
