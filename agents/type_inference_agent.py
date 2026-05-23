"""
Agent 3: Type Inference
Infers semantic types (numeric, categorical, datetime, text, id) 
and optionally casts columns using Ollama reasoning.
"""

import pandas as pd
from storage.session_store import load_dataset, save_dataset, save_report


def _infer_semantic_type(series: pd.Series) -> str:
    if series.dtype in ["int64", "float64"]:
        if series.nunique() / max(len(series), 1) > 0.95:
            return "id"
        return "numeric"
    if series.dtype == "object":
        sample = series.dropna().astype(str)
        # Try datetime
        try:
            pd.to_datetime(sample.head(20), infer_datetime_format=True)
            return "datetime"
        except Exception:
            pass
        # High cardinality → text; low cardinality → categorical
        if series.nunique() / max(len(series), 1) < 0.05:
            return "categorical"
        return "text"
    if str(series.dtype).startswith("datetime"):
        return "datetime"
    return "unknown"


def run(session_id: str, cast_types: bool = True, use_llm: bool = False) -> dict:
    try:
        df = load_dataset(session_id)
        type_map = {}
        cast_log = []

        for col in df.columns:
            semantic = _infer_semantic_type(df[col])
            type_map[col] = {
                "pandas_dtype": str(df[col].dtype),
                "semantic_type": semantic,
            }

            if cast_types:
                try:
                    if semantic == "datetime":
                        df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
                        cast_log.append(f"{col}: cast to datetime")
                    elif semantic == "categorical":
                        df[col] = df[col].astype("category")
                        cast_log.append(f"{col}: cast to category")
                    elif semantic == "numeric" and df[col].dtype == "object":
                        df[col] = pd.to_numeric(df[col], errors="coerce")
                        cast_log.append(f"{col}: cast to numeric")
                except Exception as e:
                    cast_log.append(f"{col}: cast failed — {e}")

        if cast_types:
            save_dataset(session_id, df, label="type_inference")

        report = {
            "type_map": type_map,
            "cast_log": cast_log,
        }

        if use_llm:
            from agents.llm_helper import ask
            summary_input = {c: v["semantic_type"] for c, v in type_map.items()}
            prompt = f"""
Given these column semantic types: {summary_input}
Identify any columns that seem misclassified or need special attention.
Be concise — one sentence per concern.
"""
            report["llm_notes"] = ask(prompt)

        save_report(session_id, report, label="type_report")
        return {"session_id": session_id, "status": "ok", **report}

    except Exception as e:
        return {"session_id": session_id, "status": "error", "error": str(e)}
