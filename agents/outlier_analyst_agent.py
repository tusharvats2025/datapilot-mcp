"""
Agent 5: Outlier Analyst
Detects outliers using IQR method on numeric columns.
Actions: flag, clip, or drop.
"""

import pandas as pd
import numpy as np
from storage.session_store import load_dataset, save_dataset, save_report


def run(session_id: str, action: str = "flag", z_threshold: float = 3.0) -> dict:
    """
    Args:
        action: 'flag' (add boolean col), 'clip' (winsorize), 'drop' (remove rows)
        z_threshold: used only for z-score method; IQR used by default
    """
    try:
        df = load_dataset(session_id)
        outlier_report = {}
        outlier_mask = pd.Series(False, index=df.index)

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            col_mask = (df[col] < lower) | (df[col] > upper)
            count = int(col_mask.sum())
            outlier_mask |= col_mask

            outlier_report[col] = {
                "outlier_count": count,
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
                "pct_outliers": round(count / len(df) * 100, 2),
            }

            if action == "clip":
                df[col] = df[col].clip(lower=lower, upper=upper)
            elif action == "flag":
                df[f"{col}_outlier"] = col_mask

        if action == "drop":
            rows_before = len(df)
            df = df[~outlier_mask]
            dropped = rows_before - len(df)
        else:
            dropped = 0

        save_dataset(session_id, df, label="outliers_handled")
        report = {
            "action": action,
            "numeric_columns_checked": numeric_cols,
            "total_outlier_rows": int(outlier_mask.sum()),
            "rows_dropped": dropped,
            "per_column": outlier_report,
        }
        save_report(session_id, report, label="outlier_report")

        return {"session_id": session_id, "status": "ok", **report}

    except Exception as e:
        return {"session_id": session_id, "status": "error", "error": str(e)}
