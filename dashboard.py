"""
Pipeline Dashboard — local Flask web UI
Reads session reports from /tmp/data_pipeline_sessions/ and renders them.

Run:  python dashboard.py
Open: http://localhost:7860
"""

import sys, os, json, glob
sys.path.insert(0, os.path.dirname(__file__))

from pathlib import Path
from flask import Flask, render_template_string, jsonify

STORE_DIR = Path("/tmp/data_pipeline_sessions")
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Data Pipeline Dashboard</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e2e8f0; min-height: 100vh; }
  header { background: #1a1d2e; border-bottom: 1px solid #2d3748; padding: 18px 32px; display: flex; align-items: center; gap: 14px; }
  header h1 { font-size: 1.3rem; font-weight: 700; color: #a78bfa; }
  header span { font-size: 0.8rem; color: #64748b; }
  .container { max-width: 1200px; margin: 0 auto; padding: 28px 24px; }

  /* Session selector */
  .session-bar { display: flex; align-items: center; gap: 12px; margin-bottom: 28px; flex-wrap: wrap; }
  select { background: #1e2235; border: 1px solid #2d3748; color: #e2e8f0; padding: 8px 14px; border-radius: 8px; font-size: 0.9rem; cursor: pointer; }
  .btn { background: #7c3aed; color: white; border: none; padding: 9px 20px; border-radius: 8px; cursor: pointer; font-size: 0.85rem; font-weight: 600; transition: background 0.2s; }
  .btn:hover { background: #6d28d9; }
  .btn-sm { padding: 6px 14px; font-size: 0.8rem; background: #1e2235; border: 1px solid #374151; }
  .btn-sm:hover { background: #2d3748; }

  /* Status badge */
  .badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 700; }
  .badge-pass { background: #052e16; color: #4ade80; border: 1px solid #166534; }
  .badge-warn { background: #422006; color: #fb923c; border: 1px solid #9a3412; }
  .badge-fail { background: #3b0000; color: #f87171; border: 1px solid #991b1b; }
  .badge-unknown { background: #1e2235; color: #94a3b8; border: 1px solid #374151; }

  /* Grid */
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 18px; margin-bottom: 24px; }
  .grid-2 { grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); }

  /* Cards */
  .card { background: #1a1d2e; border: 1px solid #2d3748; border-radius: 12px; padding: 20px; }
  .card h3 { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #64748b; margin-bottom: 10px; }
  .card .big { font-size: 2rem; font-weight: 800; color: #e2e8f0; }
  .card .sub { font-size: 0.8rem; color: #64748b; margin-top: 3px; }

  /* Steps timeline */
  .steps { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 24px; }
  .step-chip { display: flex; align-items: center; gap: 7px; background: #1a1d2e; border: 1px solid #2d3748; border-radius: 20px; padding: 6px 14px; font-size: 0.8rem; }
  .step-chip .dot { width: 8px; height: 8px; border-radius: 50%; background: #4ade80; }
  .step-chip .dot.missing { background: #374151; }

  /* Tables */
  table { width: 100%; border-collapse: collapse; font-size: 0.82rem; }
  th { text-align: left; padding: 8px 12px; color: #64748b; border-bottom: 1px solid #2d3748; font-weight: 600; }
  td { padding: 8px 12px; border-bottom: 1px solid #1e2235; }
  tr:hover td { background: #1e2235; }
  .null-high { color: #f87171; }
  .null-med  { color: #fb923c; }
  .null-low  { color: #4ade80; }

  /* Warnings / errors */
  .alert { border-radius: 8px; padding: 10px 14px; margin-bottom: 8px; font-size: 0.83rem; }
  .alert-warn { background: #422006; border: 1px solid #9a3412; color: #fb923c; }
  .alert-err  { background: #3b0000; border: 1px solid #991b1b; color: #f87171; }

  /* LLM summary */
  .llm-box { background: #12161f; border: 1px solid #4c1d95; border-radius: 10px; padding: 16px; font-size: 0.85rem; line-height: 1.65; color: #c4b5fd; }
  .llm-label { font-size: 0.73rem; text-transform: uppercase; letter-spacing: 0.1em; color: #7c3aed; margin-bottom: 8px; font-weight: 700; }

  .section-title { font-size: 1rem; font-weight: 700; color: #a78bfa; margin: 24px 0 14px; }
  .chart-wrap { height: 220px; }
  .empty { color: #475569; font-size: 0.85rem; padding: 16px 0; text-align: center; }
  #loading { display: none; color: #64748b; font-size: 0.85rem; padding: 10px 0; }
</style>
</head>
<body>

<header>
  <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#a78bfa" stroke-width="2">
    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
  </svg>
  <h1>Data Pipeline Dashboard</h1>
  <span>Local · Ollama · MCP</span>
</header>

<div class="container">
  <div class="session-bar">
    <label style="color:#64748b;font-size:0.85rem">Session:</label>
    <select id="sessionSelect" onchange="loadSession()">
      <option value="">— select session —</option>
    </select>
    <button class="btn btn-sm" onclick="refreshSessions()">↻ Refresh</button>
    <span id="loading">Loading...</span>
  </div>

  <div id="dashboard" style="display:none">

    <!-- Top KPI cards -->
    <div class="grid" id="kpiCards"></div>

    <!-- Pipeline steps -->
    <div class="section-title">Pipeline Steps</div>
    <div class="steps" id="stepsRow"></div>

    <!-- Warnings / Errors -->
    <div id="alertsRow"></div>

    <!-- Charts row -->
    <div class="grid grid-2">
      <div class="card">
        <h3>Null % per Column</h3>
        <div class="chart-wrap"><canvas id="nullChart"></canvas></div>
      </div>
      <div class="card">
        <h3>Outliers per Numeric Column</h3>
        <div class="chart-wrap"><canvas id="outlierChart"></canvas></div>
      </div>
    </div>

    <!-- Type map table -->
    <div class="section-title">Column Type Map</div>
    <div class="card" style="margin-bottom:18px">
      <table>
        <thead><tr><th>Column</th><th>Pandas dtype</th><th>Semantic Type</th><th>Null %</th></tr></thead>
        <tbody id="typeTable"></tbody>
      </table>
    </div>

    <!-- Feature engineering -->
    <div class="section-title">Feature Engineering</div>
    <div class="card" style="margin-bottom:18px" id="featureCard"></div>

    <!-- LLM Summary -->
    <div id="llmSection" style="display:none">
      <div class="section-title">LLM Pipeline Summary</div>
      <div class="llm-label">Generated by Ollama</div>
      <div class="llm-box" id="llmText"></div>
    </div>

  </div>

  <div id="empty" class="empty">Select a session to view results.</div>
</div>

<script>
let nullChartInst = null, outlierChartInst = null;

async function refreshSessions() {
  const res = await fetch('/api/sessions');
  const sessions = await res.json();
  const sel = document.getElementById('sessionSelect');
  sel.innerHTML = '<option value="">— select session —</option>';
  sessions.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.id; opt.textContent = `${s.id}  (${s.created || ''})`;
    sel.appendChild(opt);
  });
  if (sessions.length && !sel.value) sel.value = sessions[0].id;
  if (sel.value) loadSession();
}

async function loadSession() {
  const sid = document.getElementById('sessionSelect').value;
  if (!sid) { document.getElementById('dashboard').style.display='none'; document.getElementById('empty').style.display='block'; return; }
  document.getElementById('loading').style.display='inline';
  const res = await fetch(`/api/session/${sid}`);
  const d = await res.json();
  document.getElementById('loading').style.display='none';
  renderDashboard(d);
}

function renderDashboard(d) {
  document.getElementById('empty').style.display='none';
  document.getElementById('dashboard').style.display='block';

  // KPI cards
  const validation = d.validation || {};
  const quality = d.quality || {};
  const selector = d.selector || {};
  const stats = validation.stats || {};
  const shape = validation.final_shape || [quality.rows || '?', quality.columns || '?'];

  const pStatus = validation.pipeline_status || 'UNKNOWN';
  const badgeCls = pStatus.includes('PASS') && !pStatus.includes('WARN') ? 'badge-pass' : pStatus.includes('WARN') ? 'badge-warn' : pStatus === 'UNKNOWN' ? 'badge-unknown' : 'badge-fail';

  document.getElementById('kpiCards').innerHTML = `
    <div class="card"><h3>Pipeline Status</h3><div style="margin-top:8px"><span class="badge ${badgeCls}">${pStatus}</span></div></div>
    <div class="card"><h3>Final Shape</h3><div class="big">${shape[0]} <span style="font-size:1rem;color:#64748b">×</span> ${shape[1]}</div><div class="sub">rows × columns</div></div>
    <div class="card"><h3>Features Kept</h3><div class="big">${stats.features_kept ?? (selector.kept_features?.length ?? '?')}</div><div class="sub">${stats.features_removed ?? 0} removed</div></div>
    <div class="card"><h3>Quality</h3><div class="big" style="font-size:1.4rem">${quality.overall_quality || '?'}</div><div class="sub">${quality.duplicate_rows ?? 0} duplicate rows</div></div>
  `;

  // Steps
  const ALL_STEPS = ['intake','quality_report','type_report','missing_report','outlier_report','feature_report','selector_report','validation_report'];
  const STEP_LABELS = {'intake':'Intake','quality_report':'Quality Audit','type_report':'Type Inference','missing_report':'Missing Strategy','outlier_report':'Outlier Analysis','feature_report':'Feature Engineering','selector_report':'Feature Selection','validation_report':'Validation'};
  const done = d.meta?.steps?.map(s => s.step) || [];
  document.getElementById('stepsRow').innerHTML = ALL_STEPS.map(s =>
    `<div class="step-chip"><div class="dot ${done.includes(s)?'':'missing'}"></div>${STEP_LABELS[s]}</div>`
  ).join('');

  // Alerts
  const warns = validation.warnings || [];
  const errs = validation.validation_errors || [];
  document.getElementById('alertsRow').innerHTML =
    errs.map(e=>`<div class="alert alert-err">✗ ${e}</div>`).join('') +
    warns.map(w=>`<div class="alert alert-warn">⚠ ${w}</div>`).join('');

  // Null chart
  const nullPct = quality.null_percentage || {};
  const nullCols = Object.keys(nullPct).filter(c => nullPct[c] > 0);
  renderBarChart('nullChart', nullChartInst,
    nullCols, nullCols.map(c => nullPct[c]),
    'Null %', '#f87171',
    inst => nullChartInst = inst
  );

  // Outlier chart
  const outlierPer = (d.outlier || {}).per_column || {};
  const oCols = Object.keys(outlierPer);
  renderBarChart('outlierChart', outlierChartInst,
    oCols, oCols.map(c => outlierPer[c].outlier_count),
    'Outlier Count', '#fb923c',
    inst => outlierChartInst = inst
  );

  // Type table
  const typeMap = (d.types || {}).type_map || {};
  const nullPctAll = quality.null_percentage || {};
  document.getElementById('typeTable').innerHTML = Object.entries(typeMap).map(([col, info]) => {
    const np = nullPctAll[col] || 0;
    const npCls = np > 50 ? 'null-high' : np > 10 ? 'null-med' : 'null-low';
    return `<tr><td>${col}</td><td style="color:#94a3b8">${info.pandas_dtype}</td><td><span style="color:#a78bfa">${info.semantic_type}</span></td><td class="${npCls}">${np}%</td></tr>`;
  }).join('') || '<tr><td colspan="4" style="color:#475569;text-align:center">No type data</td></tr>';

  // Feature engineering
  const eng = (d.features || {}).engineered_features || [];
  document.getElementById('featureCard').innerHTML = eng.length
    ? eng.map(f => `<div style="padding:5px 0;border-bottom:1px solid #1e2235;font-size:0.82rem;color:#94a3b8">• ${f}</div>`).join('')
    : '<div class="empty">No features engineered yet</div>';

  // LLM summary
  const llmText = validation.llm_summary || d.features?.llm_suggestions;
  if (llmText) {
    document.getElementById('llmSection').style.display = 'block';
    document.getElementById('llmText').textContent = llmText;
  } else {
    document.getElementById('llmSection').style.display = 'none';
  }
}

function renderBarChart(canvasId, existing, labels, data, label, color, setter) {
  if (existing) existing.destroy();
  const ctx = document.getElementById(canvasId).getContext('2d');
  setter(new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label, data, backgroundColor: color + '99', borderColor: color, borderWidth: 1 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: '#64748b', font: { size: 10 } }, grid: { color: '#1e2235' } },
        y: { ticks: { color: '#64748b' }, grid: { color: '#2d3748' } }
      }
    }
  }));
}

// Boot
refreshSessions();
setInterval(refreshSessions, 15000);
</script>
</body>
</html>
"""


def get_sessions():
    if not STORE_DIR.exists():
        return []
    metas = sorted(STORE_DIR.glob("*_meta.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    result = []
    for m in metas:
        sid = m.name.replace("_meta.json", "")
        try:
            data = json.loads(m.read_text())
            result.append({"id": sid, "created": data.get("created_at", "")[:19]})
        except Exception:
            result.append({"id": sid, "created": ""})
    return result


def load_report(sid, label):
    path = STORE_DIR / f"{sid}_{label}.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/sessions")
def api_sessions():
    return jsonify(get_sessions())


@app.route("/api/session/<sid>")
def api_session(sid):
    return jsonify({
        "meta":       load_report(sid, "meta"),
        "quality":    load_report(sid, "quality_report"),
        "types":      load_report(sid, "type_report"),
        "missing":    load_report(sid, "missing_report"),
        "outlier":    load_report(sid, "outlier_report"),
        "features":   load_report(sid, "feature_report"),
        "selector":   load_report(sid, "selector_report"),
        "validation": load_report(sid, "validation_report"),
    })


if __name__ == "__main__":
    print("Starting Pipeline Dashboard...")
    print("Open: http://localhost:7860")
    app.run(host="0.0.0.0", port=7860, debug=False)
