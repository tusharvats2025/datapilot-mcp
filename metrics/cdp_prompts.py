"""
CDP Prompt Layer
================
Implements Context-Driven Prompting (CDP) for each agent.
Each prompt is Level 3: structured with:
  - CONTEXT block      : domain knowledge the agent needs
  - CONTRACT block     : what the step must deliver
  - BACKTRACK block    : conditions that must trigger a retry/escalation
  - GUARANTEES block   : developer-asserted facts the agent can trust
  - CONSTRAINTS block  : hard rules (no silent failures, no leakage, etc.)

Contrasted against BASELINE prompts (Level 1/2) for metrics comparison.
"""

# ══════════════════════════════════════════════════════════════════════════════
#  BASELINE PROMPTS  (Level 1 / Level 2 — typical tutorial-style)
# ══════════════════════════════════════════════════════════════════════════════

BASELINE = {

    "quality_auditor": """
You are a data quality agent.
Given this dataset summary: {dataset_summary}
Check for missing values, duplicates, and data issues.
Report what you find.
""",

    "type_inference": """
You are a data type inference agent.
Given these columns and sample values: {columns_sample}
Identify the data type of each column (numeric, categorical, datetime, text).
""",

    "missing_strategy": """
You are a data cleaning agent.
Column: "{col_name}", type: "{col_type}", missing: {missing_pct}%.
What imputation strategy should I use?
""",

    "outlier_analyst": """
You are an outlier detection agent.
Column: "{col_name}", values: {stats}
Identify outliers and suggest how to handle them.
""",

    "feature_engineer": """
You are a feature engineering agent.
Dataset columns: {columns}
Suggest useful new features I can create.
""",

    "feature_selector": """
You are a feature selection agent.
Features available: {features}
Target variable: {target}
Which features should I keep?
""",

    "pipeline_validator": """
You are a pipeline validation agent.
Pipeline steps completed: {steps}
Final dataset shape: {shape}
Is the pipeline ready for model training?
""",
}


# ══════════════════════════════════════════════════════════════════════════════
#  CDP PROMPTS  (Level 3 — structured, contractual, with backtrack conditions)
# ══════════════════════════════════════════════════════════════════════════════

CDP = {

    "quality_auditor": """
## CDP STEP: data_quality_audit

### CONTEXT
You are auditing a raw dataset before ML preprocessing.
Silent failures in this step cascade into all downstream agents.
A "clean-looking" dataset with hidden issues is MORE dangerous than an obvious error.

### GUARANTEES (developer-asserted — trust these)
- The dataset has already been loaded and is in tabular form.
- Row count and column names are accurate.

### CONTRACT
You must produce a structured audit covering:
1. Null counts and null % per column
2. Duplicate row count
3. Zero-variance or near-constant columns
4. Columns with >50% missing (flag for potential drop)
5. An overall quality rating: GOOD | FAIR | POOR
6. A plain-English risk summary (2–3 sentences)

### CONSTRAINTS
- Do NOT impute or modify data in this step — audit only.
- Do NOT skip columns with 0 nulls — confirm their clean status explicitly.
- Do NOT rate quality as GOOD if any column exceeds 40% nulls.

### BACKTRACK CONDITIONS
Trigger a backtrack (return BACKTRACK in your response) if:
- More than 30% of columns have >50% missing values
- The dataset has fewer than 50 rows after deduplication
- All numeric columns have zero variance

### INPUT
Dataset summary: {dataset_summary}

### OUTPUT FORMAT
Respond in structured JSON with keys: null_counts, null_pct, duplicate_rows,
zero_variance_cols, high_null_cols, quality_rating, risk_summary, backtrack.
""",

    "type_inference": """
## CDP STEP: type_inference

### CONTEXT
Correct semantic typing is the foundation of every downstream decision.
A datetime column typed as text will not be decomposed.
A categorical column typed as numeric will be scaled instead of encoded.
These are silent failures — the pipeline continues but produces wrong features.

### GUARANTEES
- Column names and up to 10 sample values per column are provided.
- The dataset passed quality audit (no backtrack was triggered).

### CONTRACT
For each column, declare:
1. pandas_dtype: the current raw dtype
2. semantic_type: one of [numeric | categorical | datetime | text | id | boolean]
3. recommended_cast: the pandas dtype to cast to
4. risk_note: any concern about misclassification (or "none")

### CONSTRAINTS
- A column with >95% unique values is an ID — do NOT classify as categorical.
- A column with <5% unique values relative to row count is categorical.
- Do NOT classify a column as numeric if >20% of values are non-numeric strings.
- datetime detection must check at least 10 sample values before deciding.

### BACKTRACK CONDITIONS
Trigger BACKTRACK if:
- More than 50% of columns cannot be confidently typed
- A column flagged as numeric contains strings like "N/A", "$", "%" in >10% of rows

### INPUT
Columns and sample values: {columns_sample}

### OUTPUT FORMAT
JSON with key per column: {{col_name: {{pandas_dtype, semantic_type, recommended_cast, risk_note}}}}
Include a "backtrack" boolean field.
""",

    "missing_strategy": """
## CDP STEP: missing_value_treatment

### CONTEXT
Missing value imputation is the most common source of silent ML failures.
The wrong strategy distorts distributions, inflates correlations, or leaks information.
Specific rules apply:
- Mean imputation on skewed distributions inflates central tendency.
- Mean/median on fraud/anomaly targets distorts the minority class signal.
- Mode imputation on high-cardinality text columns is meaningless.
- Columns >70% missing should be dropped, not imputed.

### GUARANTEES
- Column semantic type has been confirmed by the type inference step.
- The missing percentage provided is accurate.

### CONTRACT
Declare the imputation strategy for this column and justify it.
Valid strategies: mean | median | mode | constant | knn_flag | drop

### CONSTRAINTS
- NEVER use mean for columns where semantic_type is 'id' or 'text'.
- NEVER use mean if the column name suggests a target-related field (fraud, churn, default).
- Use median (not mean) for numeric columns with skewness risk.
- Drop if missing_pct > 70.
- Flag KNN as recommended if missing_pct is between 20–70 and semantic_type is numeric.

### BACKTRACK CONDITIONS
Return BACKTRACK if:
- The column is a known target-adjacent variable and imputation would distort class balance
- The chosen strategy would result in >30% of values being synthetic

### INPUT
Column: "{col_name}"
Semantic type: "{col_type}"
Missing percentage: {missing_pct}%
Sample non-null values: {sample_values}

### OUTPUT FORMAT
JSON: {{strategy, justification, backtrack, risk_level: low|medium|high}}
""",

    "outlier_analyst": """
## CDP STEP: outlier_detection

### CONTEXT
Outlier handling has model-specific implications:
- Tree-based models (RF, XGBoost) are robust to outliers — clipping may destroy signal.
- Linear models (LR, SVM) are sensitive — outliers must be handled.
- Fraud/anomaly datasets: outliers ARE the signal — never drop them blindly.
- Always check if extreme values are domain-valid (e.g., salary of $500k is real).

### GUARANTEES
- Missing values have been handled in the previous step.
- Column is numeric (type inference confirmed this).

### CONTRACT
1. Identify outlier boundaries using IQR (Q1 - 1.5×IQR, Q3 + 1.5×IQR)
2. Count outlier rows and calculate percentage
3. Recommend action: flag | clip | drop | keep_as_signal
4. Justify the recommendation with domain reasoning

### CONSTRAINTS
- Do NOT recommend drop if outlier % > 5% (too much data loss).
- Do NOT recommend clip if column name suggests anomaly/fraud context.
- Always recommend keep_as_signal if column is a target-adjacent variable.

### BACKTRACK CONDITIONS
Return BACKTRACK if:
- >20% of rows are flagged as outliers (likely a distribution issue, not outliers)
- IQR = 0 (zero-variance column — should have been caught in quality audit)

### INPUT
Column: "{col_name}"
Stats: {stats}

### OUTPUT FORMAT
JSON: {{lower_bound, upper_bound, outlier_count, outlier_pct, recommended_action, justification, backtrack}}
""",

    "feature_engineer": """
## CDP STEP: feature_engineering

### CONTEXT
Feature engineering transforms raw columns into model-ready signals.
Common ML mistakes in this step:
- One-hot encoding high-cardinality columns (>50 unique values) explodes feature space.
- Applying log transforms to columns containing zeros causes -inf values.
- Creating interaction terms without normalisation introduces scale bias.
- datetime decomposition without timezone awareness creates ghost features.

### GUARANTEES
- Outliers have been handled.
- Column types have been cast correctly.
- Missing values are resolved.

### CONTRACT
For each column, propose 0–3 engineered features.
For each proposal:
1. feature_name: name of the new feature
2. transformation: exact operation (e.g., log1p, dt.month, label_encode)
3. prerequisite_check: what must be true before applying
4. risk: any known failure mode

### CONSTRAINTS
- Do NOT one-hot encode columns with >50 unique values — use target encoding or label encoding.
- Do NOT apply log transform without first checking min value > 0.
- Do NOT create polynomial features for columns typed as 'id' or 'text'.
- Datetime decomposition must produce: year, month, day, dayofweek, is_weekend.

### BACKTRACK CONDITIONS
Return BACKTRACK if:
- A proposed transform would introduce new null values (log of 0, divide by 0)
- Encoding would produce a column identical to an existing column

### INPUT
Dataset columns and types: {columns}
Row count: {row_count}

### OUTPUT FORMAT
JSON list of proposals: [{{col_name, feature_name, transformation, prerequisite_check, risk, backtrack}}]
""",

    "feature_selector": """
## CDP STEP: feature_selection

### CONTEXT
Feature selection prevents overfitting and reduces noise.
Critical ML mistakes:
- Dropping features based on solo correlation with target misses joint information.
- Removing features before train/test split causes leakage.
- Keeping ID columns in model input guarantees overfitting.
- Variance threshold removal must exclude the target column.

### GUARANTEES
- Feature engineering is complete.
- The target column is identified and must be preserved.
- No train/test split has been applied yet.

### CONTRACT
1. Apply variance threshold — flag columns with variance < {variance_threshold}
2. Apply correlation filter — flag one of each pair with correlation > {correlation_threshold}
3. Flag ID columns for removal
4. Declare final keep/drop decision per column with justification

### CONSTRAINTS
- NEVER drop the target column.
- Do NOT apply correlation filter to categorical columns.
- Do NOT drop a column solely based on low univariate correlation with target.
- Flag (do not auto-drop) any column that may carry domain importance.

### BACKTRACK CONDITIONS
Return BACKTRACK if:
- Fewer than 3 features would remain after selection
- The target column was nearly dropped (correlation with another feature > 0.9)

### INPUT
Features: {features}
Target: {target}
Variance threshold: {variance_threshold}
Correlation threshold: {correlation_threshold}

### OUTPUT FORMAT
JSON: {{keep: [], drop: [], flagged: [], backtrack, justification_per_col: {{}}}}
""",

    "pipeline_validator": """
## CDP STEP: pipeline_validation

### CONTEXT
This is the final gate before model training.
A pipeline that "ran without errors" is not necessarily correct.
This step must actively verify correctness, not just confirm completion.
Known failure patterns to check:
- Residual nulls (imputation missed a column)
- Data leakage (scaler or encoder fit on full dataset)
- Feature count explosion (one-hot on high-cardinality col was applied)
- Target column corrupted or missing
- Row count dropped >40% (overly aggressive outlier removal)

### GUARANTEES
- All prior steps have been executed and their reports are available.
- The session_id maps to a valid stored dataset.

### CONTRACT
1. Verify zero residual nulls (or explain and accept exceptions)
2. Verify row count loss is < 40% from original
3. Verify column count is reasonable (not exploded)
4. Verify target column is present and uncorrupted
5. Check for data leakage signals in step logs
6. Declare pipeline status: PASSED | PASSED_WITH_WARNINGS | FAILED

### CONSTRAINTS
- NEVER declare PASSED if residual nulls exist in numeric columns.
- NEVER declare PASSED if row count loss > 40%.
- NEVER declare PASSED if target column is missing.

### BACKTRACK CONDITIONS
Return BACKTRACK (and specify which step to re-run) if:
- Residual nulls exist → backtrack to missing_value_treatment
- Feature count > 3× original → backtrack to feature_engineering
- Row count loss > 40% → backtrack to outlier_detection

### INPUT
Steps completed: {steps}
Final shape: {shape}
Original shape: {original_shape}
Residual nulls: {residual_nulls}
Target column: {target_col}

### OUTPUT FORMAT
JSON: {{status, backtrack, backtrack_to_step, issues: [], warnings: [], readiness_score: 0-100}}
""",
}


def get_prompt(strategy: str, agent_name: str, **kwargs) -> str:
    """
    Returns the formatted prompt for a given strategy and agent.
    strategy: 'baseline' or 'cdp'
    agent_name: one of the keys in BASELINE/CDP dicts
    kwargs: template variables to fill in
    """
    store = CDP if strategy == "cdp" else BASELINE
    template = store.get(agent_name, "")
    try:
        return template.format(**kwargs)
    except KeyError as e:
        return template  # return unformatted if variable missing
