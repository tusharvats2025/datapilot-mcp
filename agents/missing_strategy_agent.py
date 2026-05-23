"""
Agent 4: Missing Strategy
Decides and applies imputation strategy per column.
Strategy: mean/median for numeric, mode for categorical, drop if >70% missing.
Optionally asks Ollama for strategy recommendations.
"""

import pandas as pd
from storage.session_store import load_dataset, save_dataset, save_report, load_report


def run(session_id: str, use_llm: bool = False, drop_threshold: float = 0.7) -> dict:
    try:
        df = load_dataset(session_id)
        type_report = load_report(session_id, "type_report")
        type_map = type_report.get("type_map", {})

        strategy_log = {}
        dropped_cols = []

        for col in df.columns:
            null_pct = df[col].isnull().mean()
            if null_pct == 0:
                strategy_log[col] = "no_action (no nulls)"
                continue

            if null_pct >= drop_threshold:
                dropped_cols.append(col)
                strategy_log[col] = f"dropped ({null_pct:.0%} missing)"
                continue

            semantic = type_map.get(col, {}).get("semantic_type", "unknown")

            if use_llm:
                from agents.llm_helper import ask_json
                prompt = f"""
Column: "{col}", semantic type: "{semantic}", null percentage: {null_pct:.1%}.
Choose the best imputation strategy.
Respond with JSON: {{"strategy": "mean"|"median"|"mode"|"constant"|"drop", "reason": "..."}}
"""
                result = ask_json(prompt)
                chosen = result.get("strategy", "median")
            else:
                # Rule-based fallback
                if semantic == "numeric":
                    chosen = "median"
                elif semantic in ("categorical", "text"):
                    chosen = "mode"
                else:
                    chosen = "mode"

            # Apply strategy
            try:
                if chosen == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif chosen == "median":
                    df[col] = df[col].fillna(df[col].median())
                elif chosen == "mode":
                    mode_val = df[col].mode()
                    df[col] = df[col].fillna(mode_val[0] if len(mode_val) else "UNKNOWN")
                elif chosen == "constant":
                    df[col] = df[col].fillna(0)
                strategy_log[col] = f"applied: {chosen} (was {null_pct:.0%} null)"
            except Exception as e:
                strategy_log[col] = f"failed: {e}"

        if dropped_cols:
            df.drop(columns=dropped_cols, inplace=True)

        save_dataset(session_id, df, label="missing_handled")
        report = {"strategy_log": strategy_log, "dropped_columns": dropped_cols}
        save_report(session_id, report, label="missing_report")

        return {
            "session_id": session_id,
            "status": "ok",
            "strategy_log": strategy_log,
            "dropped_columns": dropped_cols,
        }

    except Exception as e:
        return {"session_id": session_id, "status": "error", "error": str(e)}
