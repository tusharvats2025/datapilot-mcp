"""
Data Pipeline MCP Server
Exposes 8 agents as MCP tools for use with any MCP client (Claude Desktop, etc.)
All agents use local Ollama models — no API keys required.

Run:  python server.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from mcp.server.fastmcp import FastMCP

# Import all agents
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
from agents import orchestrator_agent

mcp = FastMCP("DataPipelineMCP")


# ─── Tool 1: Data Intake ──────────────────────────────────────────────────────

@mcp.tool()
def ingest_data(file_path: str, file_type: str = "csv", session_id: str = "") -> dict:
    """
    Load a local CSV or JSON file into a pipeline session.
    Returns session_id to use with all downstream agents.

    Args:
        file_path: Absolute path to the data file.
        file_type: 'csv' or 'json'
        session_id: Leave empty to start a new session.
    """
    return data_intake_agent.run(
        file_path=file_path,
        file_type=file_type,
        session_id=session_id or None,
    )


# ─── Tool 2: Quality Auditor ─────────────────────────────────────────────────

@mcp.tool()
def audit_quality(session_id: str, use_llm: bool = False) -> dict:
    """
    Audit data quality: nulls, duplicates, zero-variance columns.
    Set use_llm=True to get an Ollama-generated plain-English summary.

    Args:
        session_id: Session ID from ingest_data.
        use_llm: Whether to generate an LLM summary (requires Ollama running).
    """
    return quality_auditor_agent.run(session_id=session_id, use_llm=use_llm)


# ─── Tool 3: Type Inference ──────────────────────────────────────────────────

@mcp.tool()
def infer_types(session_id: str, cast_types: bool = True, use_llm: bool = False) -> dict:
    """
    Infer semantic column types (numeric, categorical, datetime, text, id).
    Optionally casts columns to correct dtypes.

    Args:
        session_id: Session ID from ingest_data.
        cast_types: Whether to actually cast columns in the dataset.
        use_llm: Ask Ollama to review and flag type mismatches.
    """
    return type_inference_agent.run(
        session_id=session_id,
        cast_types=cast_types,
        use_llm=use_llm,
    )


# ─── Tool 4: Missing Strategy ────────────────────────────────────────────────

@mcp.tool()
def handle_missing(
    session_id: str,
    use_llm: bool = False,
    drop_threshold: float = 0.7,
) -> dict:
    """
    Apply imputation strategy for missing values.
    Columns with >drop_threshold fraction missing are dropped.
    Numeric → median, Categorical → mode (or LLM-chosen if use_llm=True).

    Args:
        session_id: Session ID.
        use_llm: Let Ollama choose strategy per column.
        drop_threshold: Drop columns missing more than this fraction (default 0.7).
    """
    return missing_strategy_agent.run(
        session_id=session_id,
        use_llm=use_llm,
        drop_threshold=drop_threshold,
    )


# ─── Tool 5: Outlier Analyst ─────────────────────────────────────────────────

@mcp.tool()
def detect_outliers(
    session_id: str,
    action: str = "flag",
) -> dict:
    """
    Detect outliers in numeric columns using IQR method.

    Args:
        session_id: Session ID.
        action: One of:
            'flag'  – add a boolean column per feature (default)
            'clip'  – winsorize values to IQR bounds
            'drop'  – remove outlier rows entirely
    """
    return outlier_analyst_agent.run(session_id=session_id, action=action)


# ─── Tool 6: Feature Engineer ────────────────────────────────────────────────

@mcp.tool()
def engineer_features(
    session_id: str,
    use_llm: bool = False,
    polynomial: bool = False,
) -> dict:
    """
    Create new features: datetime decomposition, label encoding,
    optional polynomial/interaction terms.

    Args:
        session_id: Session ID.
        use_llm: Ask Ollama to suggest additional feature ideas.
        polynomial: Create squared and interaction features for top numeric cols.
    """
    return feature_engineer_agent.run(
        session_id=session_id,
        use_llm=use_llm,
        polynomial=polynomial,
    )


# ─── Tool 7: Feature Selector ────────────────────────────────────────────────

@mcp.tool()
def select_features(
    session_id: str,
    target_col: str = "",
    variance_threshold: float = 0.01,
    correlation_threshold: float = 0.95,
    use_llm: bool = False,
) -> dict:
    """
    Remove low-variance and highly correlated features.

    Args:
        session_id: Session ID.
        target_col: Target/label column to preserve (won't be selected against).
        variance_threshold: Drop features with variance below this (default 0.01).
        correlation_threshold: Drop one of any pair with correlation above this (default 0.95).
        use_llm: Ask Ollama to review the selected feature set.
    """
    return feature_selector_agent.run(
        session_id=session_id,
        target_col=target_col or None,
        variance_threshold=variance_threshold,
        correlation_threshold=correlation_threshold,
        use_llm=use_llm,
    )


# ─── Tool 8: Pipeline Validator ──────────────────────────────────────────────

@mcp.tool()
def validate_pipeline(session_id: str, use_llm: bool = False) -> dict:
    """
    Validate the full pipeline: checks all steps ran, final data integrity,
    residual nulls, and generates a health report.

    Args:
        session_id: Session ID.
        use_llm: Ask Ollama to write a pipeline health narrative.
    """
    return pipeline_validator_agent.run(session_id=session_id, use_llm=use_llm)


# ─── Tool 9: Full Pipeline Orchestrator ──────────────────────────────────────

@mcp.tool()
def run_pipeline(
    file_path: str,
    file_type: str = "csv",
    target_col: str = "",
    outlier_action: str = "clip",
    polynomial: bool = False,
    use_llm: bool = False,
    drop_threshold: float = 0.7,
    correlation_threshold: float = 0.95,
) -> dict:
    """
    Run the FULL data pipeline in one call — chains all 8 agents automatically.
    Returns a consolidated report with session_id, final shape, and health status.

    Args:
        file_path: Local path to CSV or JSON file.
        file_type: 'csv' or 'json'
        target_col: Target/label column (preserved during feature selection).
        outlier_action: 'flag', 'clip', or 'drop'
        polynomial: Create polynomial/interaction features.
        use_llm: Enable Ollama LLM reasoning in all agents.
        drop_threshold: Drop columns with more missing than this fraction.
        correlation_threshold: Drop correlated features above this threshold.
    """
    return orchestrator_agent.run(
        file_path=file_path,
        file_type=file_type,
        target_col=target_col or None,
        outlier_action=outlier_action,
        polynomial=polynomial,
        use_llm=use_llm,
        drop_threshold=drop_threshold,
        correlation_threshold=correlation_threshold,
    )


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Starting Data Pipeline MCP Server...")
    print("Agents: ingest_data, audit_quality, infer_types, handle_missing,")
    print("        detect_outliers, engineer_features, select_features, validate_pipeline")
    mcp.run(transport="stdio")
