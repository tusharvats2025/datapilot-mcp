"""
Shared in-memory + disk session store.
Agents communicate via session_id references - never raw data over MCP.

"""
import json
import uuid
import pickle
from pathlib import Path
from datetime import datetime
from typing import Any

STORE_DIR = Path("/tmp/data_pipeline_sessions")
STORE_DIR.mkdir(exist_ok=True)

def new_session() -> str:
    sid = str(uuid.uuid4())[:8]
    meta = {"session_id": sid, "created_at": datetime.now().isoformat(), "steps":[]}
    _write_meta(sid, meta)

def save_dataset(session_id: str, df, label: str = "dataset"):
    path = STORE_DIR / f"{session_id}_data.pkl"
    with open(path, "wb") as f:
        pickle.dump(df, f)
    _append_step(session_id, label)

def load_dataset(session_id: str):
    path = STORE_DIR / f"{session_id}_data.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No dataset for session {session_id}")
    with open(path, "rb") as f:
        return pickle.load(f)

def save_report(session_id: str, report: dict, label: str):
    path = STORE_DIR / f"{session_id}_{label}.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    _append_step(session_id, label)

def load_report(session_id: str, label: str) -> dict:
    path = STORE_DIR / f"{session_id}_{label}.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)
    
def get_meta(session_id: str) -> dict:
    return _read_meta(session_id)
   
def _write_meta(sid, meta):
    path = STORE_DIR / f"{sid}_meta.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2)

def _read_meta(sid) -> dict:
    path = STORE_DIR / f"{sid}_meta.json"
    if not path.exists():
        return {}
    with open(path, "r") as f:
        return json.load(f)

def _append_step(sid, step):
    meta = _read_meta(sid)
    meta.setdefault("steps", []).append({"step": step, "at": datetime.now().isoformat()})
    _write_meta(sid, meta)