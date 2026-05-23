"""
Agent 1: Data Intake
Reads CSV or JSON from a file path and stores it in the session.
"""

import pandas as pd
from storage.session_store import new_session, save_dataset, save_report


def run(file_path: str, file_type: str = "csv", session_id: str = None) -> dict:
    """
    Args:
        file_path: Local path to the data file.
        file_type: 'csv' or 'json'
        session_id: Reuse existing session or None to create new.
    Returns:
        dict with session_id and basic info.
    """
    sid = session_id or new_session()

    try:
        if file_type == "csv":
            df = pd.read_csv(file_path)
        elif file_type == "json":
            df = pd.read_json(file_path)
        else:
            return {"error": f"Unsupported file_type: {file_type}"}

        save_dataset(sid, df, label="intake")

        report = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "file_path": file_path,
            "file_type": file_type,
        }
        save_report(sid, report, label="intake_report")

        return {
            "session_id": sid,
            "status": "ok",
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "message": f"Data loaded successfully into session {sid}",
        }

    except Exception as e:
        return {"session_id": sid, "status": "error", "error": str(e)}
