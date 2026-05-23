"""
Agent 7: Feature Selector
Selects best features using:
- Variance threshold (drop near-zero variance)
- Correlation filter (drop highly correlated features)
- Optional: LLM reasoning on which features to keep
"""

import pandas as pd
import numpy as np
from storage.session_store import load_dataset, save_dataset, save_report


def run(
    session_id: str,
    target_col: str = None,
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    use_llm: bool = False,
) -> dict:
    try:
        df = load_dataset(session_id)
        removed = []
        kept = list(df.columns)

        # Separate target if given
        target = None
        if target_col and target_col in df.columns:
            target = df[target_col]
            df = df.drop(columns=[target_col])

        numeric_df = df.select_dtypes(include="number")

        # Step 1: Variance threshold
        low_var = [c for c in numeric_df.columns if numeric_df[c].var() < variance_threshold]
        df.drop(columns=low_var, inplace=True)
        removed.extend([f"{c} (low variance)" for c in low_var])

        # Step 2: Correlation filter — keep one from each correlated pair
        numeric_df = df.select_dtypes(include="number")
        corr_matrix = numeric_df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        high_corr_cols = [c for c in upper.columns if any(upper[c] > correlation_threshold)]
        df.drop(columns=high_corr_cols, inplace=True)
        removed.extend([f"{c} (high correlation)" for c in high_corr_cols])

        kept = list(df.columns)

        llm_notes = None
        if use_llm:
            from agents.llm_helper import ask
            prompt = f"""
After automatic feature selection, these features remain: {kept}
These features were removed: {removed}
Target variable: {target_col or 'not specified'}

Comment on whether the selected features make sense and flag any concerns.
Keep it concise — 3-4 sentences.
"""
            llm_notes = ask(prompt)

        # Reattach target
        if target is not None:
            df[target_col] = target

        save_dataset(session_id, df, label="features_selected")
        report = {
            "kept_features": kept,
            "removed_features": removed,
            "final_column_count": len(df.columns),
            "llm_notes": llm_notes,
        }
        save_report(session_id, report, label="selector_report")

        return {"session_id": session_id, "status": "ok", **report}

    except Exception as e:
        return {"session_id": session_id, "status": "error", "error": str(e)}
