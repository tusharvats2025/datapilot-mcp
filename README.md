<div align="center">

# 🧠 DataPilot
### Agentic ML Preprocessing Pipeline — MCP Server + Context-Driven Prompting

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![MCP](https://img.shields.io/badge/MCP-FastMCP-7C3AED?style=for-the-badge)](https://modelcontextprotocol.io)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-10b981?style=for-the-badge)](https://ollama.ai)
[![License](https://img.shields.io/badge/License-MIT-f59e0b?style=for-the-badge)](LICENSE)

**8 AI agents · Local LLMs · Zero API cost · Runs offline**

[Quick Start](#-quick-start) · [How It Works](#-how-it-works) · [Experiment](#-cdp-vs-baseline-experiment) · [MCP + Claude Desktop](#-connect-to-claude-desktop) · [Tier 2 / Tier 3](#-commercial-tiers)

</div>

---

## ⚠️ Honest Disclaimer

This is a **personal engineering and learning project** — not a published research paper.

The CDP vs Baseline experiment is a structured self-exploration: I designed it, ran it, measured it, and documented it carefully. The findings are honest and reproducible but **have not been peer-reviewed or externally validated**. Phase 1 uses deterministic mock LLM responses for reproducibility — numbers will differ with live Ollama models. Phase 2 (live model benchmarking) is planned but not yet built.

---

## 📌 What This Does

You give it a messy CSV. Eight AI agents clean it, type it, impute missing values, detect outliers, engineer features, select the best ones, and validate the result — with every decision logged and explained.

**Real example — 520 row HR dataset:**

```
INPUT:  age, salary, bonus, department, dept_code, hire_date,
        employee_id, notes, promoted, constant_col, mystery_col
        (nulls injected · 20 duplicates · outliers · 88% missing column)

OUTPUT: 520 rows × 9 columns · status: PASSED
        constant_col dropped (zero variance)
        mystery_col dropped (88% missing)
        182 salary outliers clipped to IQR bounds
        age + bonus nulls filled with median
        every decision logged with justification
```

---

## ✨ Key Features

| Feature | Detail |
|---|---|
| **9 MCP tools** | Callable from Claude Desktop or any MCP client |
| **8 modular agents** | Each agent is independent, testable, swappable |
| **CDP-structured prompts** | Level 3 prompting — contracts, constraints, backtrack conditions |
| **Multi-provider LLM** | Ollama (local) · OpenAI · Gemini · HuggingFace — swap via config |
| **A/B experiment layer** | CDP vs Baseline comparison with full metrics logging |
| **Fully offline** | Runs on CPU, no GPU, no cloud, no API keys required |
| **Session-based state** | Datasets never travel over MCP — only session IDs |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│              MCP Client (Claude Desktop / CLI)           │
└──────────────────────────┬──────────────────────────────┘
                           │ MCP Protocol (stdio)
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  MCP Server  (server.py)                 │
│  9 tools: ingest · audit · types · missing · outliers   │
│           features · select · validate · run_pipeline   │
└──────────────────────────┬──────────────────────────────┘
                           │ Python function calls
                           ▼
┌─────────────────────────────────────────────────────────┐
│               8 Agents  (agents/)                       │
│  Each agent: load session → process → save → return     │
│  LLM calls via Ollama are optional (use_llm=False)      │
└──────────┬──────────────────────────┬───────────────────┘
           │                          │
           ▼                          ▼
  ┌─────────────────┐      ┌──────────────────────┐
  │  Session Store  │      │  Ollama / Any LLM    │
  │  pickle + JSON  │      │  phi3:mini · llama3  │
  │  /tmp/sessions  │      │  mistral · gemma2    │
  └─────────────────┘      └──────────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│              Metrics Layer  (metrics/)                   │
│  MetricsCollector · CDP Prompts · Comparison Runner     │
└─────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/yourusername/datapilot-mcp
cd datapilot-mcp

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Pull a local model (skip for mock mode)
ollama pull phi3:mini

# 4. Run the pipeline — mock mode, no Ollama needed
python run.py

# 5. Run with real Ollama
python run.py --model phi3:mini

# 6. Run step-by-step interactively
python run.py --step

# 7. Run CDP vs Baseline experiment
python metrics/comparison_runner.py --runs 3

# 8. Launch the session dashboard
python dashboard.py    # → http://localhost:7860
```

---

## 🤖 The 8 Agents

| # | Agent | What It Does | Key Decision |
|---|---|---|---|
| 1 | **Data Intake** | Loads CSV/JSON → session | Validates shape + columns |
| 2 | **Quality Auditor** | Nulls, duplicates, flags | Rates GOOD / FAIR / POOR |
| 3 | **Type Inference** | Classifies column semantics | id vs categorical vs numeric |
| 4 | **Missing Strategy** | Per-column imputation | Median not mean (outlier-robust) |
| 5 | **Outlier Analyst** | IQR detection | flag / clip / drop |
| 6 | **Feature Engineer** | Creates new features | datetime decompose, encode |
| 7 | **Feature Selector** | Removes low-signal cols | variance + correlation filter |
| 8 | **Pipeline Validator** | End-to-end integrity | PASSED / WARNINGS / FAILED |

All agents work **rule-based without any LLM**. Set `use_llm=True` to add Ollama reasoning.

---

## 🔌 Connect to Claude Desktop

**1. Add to `claude_desktop_config.json`:**

- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "datapilot": {
      "command": "python",
      "args": ["/absolute/path/to/datapilot-mcp/server.py"]
    }
  }
}
```

**2. Restart Claude Desktop. Then just ask:**

```
"Run the full pipeline on /path/to/sales_data.csv with target column churn"
"Audit the quality of my dataset at /tmp/data.csv"
"Run pipeline with CDP mode and clip outliers, target column: fraud_flag"
```

Claude calls the MCP tools automatically, chains all 8 agents, returns structured results.

---

## 🧪 CDP vs Baseline Experiment

The experiment layer runs every agent twice — baseline prompt vs CDP-structured prompt — and logs every metric.

```bash
# Run comparison (mock mode — fast, deterministic)
python metrics/comparison_runner.py --runs 3

# Run with real Ollama
python metrics/comparison_runner.py --model phi3:mini --runs 3

# Open notebook for visualisations
jupyter notebook CDP_vs_Baseline_Experiment.ipynb
```

### Phase 1 Results (mock LLM — see disclaimer above)

| Metric | Baseline | CDP | Δ |
|---|---|---|---|
| Avg Quality Score | 3.14 / 5 | **3.87 / 5** | **+23%** |
| Output Tokens | 14 | 82 | 5.8× richer |
| Avg Latency | 287.9ms | 292.5ms | +4.6ms |
| Backtrack Capability | ❌ | ✅ all agents | structural |
| Risk Flags | ❌ | ✅ | structural |

### What CDP Actually Is

CDP (Context-Driven Prompting) structures prompts as software contracts:

```
### CONTEXT        — domain knowledge to prevent silent failures
### GUARANTEES     — facts the developer asserts (agent skips verifying)
### CONTRACT       — exactly what the step must deliver
### CONSTRAINTS    — hard ML rules (never mean on fraud cols, etc.)
### BACKTRACK      — conditions that halt the pipeline, not silence it
```

The backtrack condition is the key difference — a CDP agent that would produce a wrong answer says so explicitly instead of silently corrupting downstream steps.

---

## 🔄 Switch LLM Provider

```yaml
# config.yaml — change one line, everything picks it up
llm:
  provider: ollama    # ollama | openai | gemini | huggingface | mock
  ollama_model: phi3:mini
```

```bash
# Or via environment variable (overrides config.yaml)
export DATAPILOT_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

---

## 📁 Project Structure

```
datapilot-mcp/
├── server.py                    # MCP server — 9 tools
├── run.py                       # CLI runner — coloured output
├── dashboard.py                 # Flask dashboard — session viewer
├── generate_sample_data.py      # 520-row messy test dataset
├── config.yaml                  # Provider + model settings
├── requirements.txt
│
├── agents/                      # 8 agents + orchestrator
│   ├── data_intake_agent.py
│   ├── quality_auditor_agent.py
│   ├── type_inference_agent.py
│   ├── missing_strategy_agent.py
│   ├── outlier_analyst_agent.py
│   ├── feature_engineer_agent.py
│   ├── feature_selector_agent.py
│   ├── pipeline_validator_agent.py
│   ├── orchestrator_agent.py    # chains all 8 in sequence
│   └── llm_helper.py            # unified LLM provider interface
│
├── metrics/
│   ├── cdp_prompts.py           # baseline + CDP prompt library
│   ├── metrics_collector.py     # instruments every LLM call
│   ├── comparison_runner.py     # A/B experiment runner
│   ├── raw_metrics.csv          # 42 recorded calls — 3 runs
│   └── comparison_summary.csv   # per-agent aggregated results
│
├── storage/
│   └── session_store.py         # session-based pickle + JSON store
│
└── CDP_vs_Baseline_Experiment.ipynb   # 7-chart experiment notebook
```

---

## 💼 Commercial Tiers

This repo is the **free public layer**. Two commercial tiers are available separately.

---

### Tier 2 — Studio `$149–$299`

**Who it's for:** Data scientists and small teams who want a complete working local tool.

**What it adds on top of this repo:**

```
api/                         FastAPI backend — 7 HTTP endpoints
  main.py                    CORS-configured app entry point
  pipeline_router.py         POST /pipeline/run, GET /pipeline/status, etc.
  experiment_router.py       POST /experiment/run, GET /experiment/results
  providers.py               Provider switcher endpoints

frontend/                    Next.js web app — 4 screens
  app/page.tsx               Screen 1: Upload + Configure + CDP toggle
  app/pipeline/[id]/         Screen 2: Live pipeline execution (polling)
  app/results/[id]/          Screen 3: Results + report + downloads
  app/experiment/            Screen 4: CDP vs Baseline dashboard + live charts

install.py                   One-command installer
```

**How it works:**

```bash
python install.py    # checks Python, installs deps, builds frontend, checks Ollama

# Then two terminals:
uvicorn api.main:app --reload --port 8000
cd frontend && npm run dev    # → http://localhost:3000
```

**What you get at localhost:3000:**
- Upload any CSV via drag-and-drop
- Toggle CDP mode on/off
- Watch 8 agents execute live (2-second polling, agent cards flip as steps complete)
- Download cleaned dataset as CSV or PDF report
- Run CDP vs Baseline experiment with interactive Recharts visualisations
- Switch LLM provider via UI (Ollama / OpenAI / Gemini / HuggingFace)

**Endpoints:**
```
POST  /pipeline/run                   — upload + start pipeline
GET   /pipeline/status/{session_id}   — per-agent progress polling
GET   /pipeline/report/{session_id}   — full results JSON
GET   /pipeline/download/{id}/csv     — cleaned dataset
GET   /pipeline/download/{id}/report  — PDF report
POST  /experiment/run                 — A/B comparison
GET   /experiment/results/{run_id}    — metrics + chart data
```

---

### Tier 3 — Integration `$500–$2000`

**Who it's for:** Companies embedding DataPilot into an existing product or data stack.

**What it adds on top of Tier 2:**

```
api/db_router.py        Database intake — Postgres, MySQL, SQLite
api/schema_mapper.py    Schema Mapping Engine — generates domain CDP prompts
api/main.py             Production CORS via ALLOWED_ORIGINS env var
TIER3_CDP_GUIDE.md      CDP prompt delivery guide for client teams
```

**Key capabilities:**

**1. Production CORS:**
```bash
ALLOWED_ORIGINS="https://client.com,https://app.client.com" \
  uvicorn api.main:app --port 8000
```

**2. Database intake — no CSV export needed:**
```bash
# Postgres
POST /db/run
{
  "connection_string": "postgresql://user:pass@host/dbname",
  "table_name": "transactions",
  "target_col": "fraud_flag"
}

# MySQL
{ "connection_string": "mysql+pymysql://user:pass@host/dbname", ... }

# SQLite
{ "connection_string": "sqlite:///path/to/file.db", ... }
```

**3. Schema Mapping Engine — domain-specific CDP prompts:**
```bash
POST /db/generate-prompts
{
  "session_id": "abc123",
  "domain_hint": "finance",
  "target_col": "fraud_flag"
}
```

The engine reads your column names, detects your domain (finance / healthcare / HR / ecommerce / logistics), and generates CDP prompts with constraints specific to your schema:

```
# Generic Tier 2 constraint:
"NEVER use mean for sensitive columns"

# Domain-specific Tier 3 constraint (auto-generated):
"NEVER use mean for fraud_flag, transaction_amount, risk_score —
 these are finance domain signal columns. Use median."
```

**4. DB utility endpoints:**
```
POST /db/connect              — test connection string
POST /db/tables               — list all tables
POST /db/preview/{table}      — schema + first 20 rows
```

**Tier 3 pricing by scope:**
- `$500` — integrate FastAPI into client's existing Python backend
- `$1000` — embed Next.js screens into client's existing React app
- `$2000` — full: DB integration + domain CDP prompts + production deployment

---

## 📊 Experiment Results — Notebook Charts

The Jupyter notebook contains 7 dark-themed charts:

| Chart | What It Shows |
|---|---|
| KPI Dashboard | 8 headline metrics side by side |
| Quality per Agent | CDP vs Baseline bars with delta annotations |
| Token Economy | True cost model with break-even line |
| Latency Analysis | Violin distribution + per-agent delta |
| Capability Radar | 6-dimension structural comparison |
| Run Stability | 3-run scatter showing variance |
| Heatmap | All metrics × all agents normalised |

---

## ⚠️ Honest Limitations

- **Phase 1 uses mock LLM** — deterministic responses, not real model output
- **Quality scorer is heuristic** — structural fields present, not semantic correctness
- **3 runs** — stability signal only, not statistical significance
- **Single dataset** — synthetic HR data, may not generalise
- **No cascading error measurement** — agents evaluated independently

Phase 2 (live Ollama + semantic scoring via sentence-transformers) is planned.

---

## 🏆 Summary

Built to demonstrate:

```
• 9-tool MCP server — Claude Desktop integration, FastMCP protocol
• 8-agent pipeline — sequential composition, stateless protocol + stateful store
• CDP prompt engineering — Level 3 structured prompts with domain constraints
• A/B instrumentation — every LLM call logged, reproducible experiment
• Multi-provider LLM abstraction — Ollama / OpenAI / Gemini / HuggingFace
• Local-first architecture — CPU only, offline capable, zero API keys
```

---

## 🛠️ Tech Stack

```
Protocol     MCP (FastMCP)
LLM          Ollama · OpenAI · Gemini · HuggingFace (unified interface)
Data         pandas · numpy · scikit-learn
Storage      pickle + JSON (local session store)
Dashboard    Flask + Chart.js
Notebook     Jupyter · matplotlib
```

---

## 📄 License

MIT — use freely. A mention is appreciated but not required.

---

<div align="center">

**Built locally · No cloud · No API keys **

*Everything runs on your machine.*

</div>
