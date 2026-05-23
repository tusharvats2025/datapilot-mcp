"""
Agent 6: Feature Engineer
Creates new features: datetime decomposition, label encoding, 
polynomial features, interaction terms.
Optionally asks Ollama to suggest creative features.
"""

import pandas as pd
import numpy as np
from storage.session_store import load_dataset, save_dataset, save_report, load_report


def run(session_id: str, use_llm: bool = False, polynomial: bool = False) -> dict:
    try:
        df = load_dataset(session_id)
        type_report = load_report(session_id, "type_report")
        type_map = type_report.get("type_map", {})

        engineered = []

        for col, info in type_map.items():
            if col not in df.columns:
                continue
            semantic = info.get("semantic_type", "unknown")

            # Datetime decomposition
            if semantic == "datetime":
                try:
                    df[col] = pd.to_datetime(df[col])
                    df[f"{col}_year"] = df[col].dt.year
                    df[f"{col}_month"] = df[col].dt.month
                    df[f"{col}_day"] = df[col].dt.day
                    df[f"{col}_dayofweek"] = df[col].dt.dayofweek
                    df[f"{col}_is_weekend"] = df[col].dt.dayofweek >= 5
                    engineered.append(f"{col}: datetime decomposed → year/month/day/dayofweek/is_weekend")
                except Exception as e:
                    engineered.append(f"{col}: datetime decompose failed — {e}")

            # Label encode categoricals
            elif semantic == "categorical":
                try:
                    df[f"{col}_encoded"] = df[col].cat.codes if hasattr(df[col], 'cat') else pd.factorize(df[col])[0]
                    engineered.append(f"{col}: label encoded → {col}_encoded")
                except Exception as e:
                    engineered.append(f"{col}: encoding failed — {e}")

        # Polynomial features for numeric cols (degree 2, top 3 cols only)
        if polynomial:
            numeric_cols = df.select_dtypes(include="number").columns.tolist()[:3]
            for col in numeric_cols:
                df[f"{col}_sq"] = df[col] ** 2
                engineered.append(f"{col}: squared → {col}_sq")
            # Interaction: first two numeric
            if len(numeric_cols) >= 2:
                a, b = numeric_cols[0], numeric_cols[1]
                df[f"{a}_x_{b}"] = df[a] * df[b]
                engineered.append(f"interaction: {a} × {b}")

        llm_suggestions = None
        if use_llm:
            from agents.llm_helper import ask
            cols_summary = {c: v.get("semantic_type") for c, v in type_map.items()}
            prompt = f"""
Dataset columns with types: {cols_summary}
Suggest 3 creative feature engineering ideas for this dataset.
Be specific about column names. Respond in bullet points.
"""
            llm_suggestions = ask(prompt)

        save_dataset(session_id, df, label="features_engineered")
        report = {
            "engineered_features": engineered,
            "new_shape": [len(df), len(df.columns)],
            "llm_suggestions": llm_suggestions,
        }
        save_report(session_id, report, label="feature_report")

        return {
            "session_id": session_id,
            "status": "ok",
            "engineered_features": engineered,
            "new_columns": len(df.columns),
            "llm_suggestions": llm_suggestions,
        }

    except Exception as e:
        return {"session_id": session_id, "status": "error", "error": str(e)}
