"""
Generates a realistic messy CSV for pipeline testing.
Includes: nulls, duplicates, outliers, mixed types, datetime cols.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def generate(output_path: str = "/tmp/sample_messy_data.csv", rows: int = 500, seed: int = 42):
    np.random.seed(seed)
    n = rows

    df = pd.DataFrame({
        # Numeric - clean
        "age": np.random.randint(18, 80, n).astype(float),
        # Numeric with outliers
        "salary": np.random.normal(60000, 15000, n),
        # Numeric with lots of nulls
        "bonus": np.where(np.random.rand(n) < 0.45, np.nan, np.random.normal(5000, 2000, n)),
        # Categorical - low cardinality
        "department": np.random.choice(["Engineering", "Sales", "HR", "Finance", None], n,
                                        p=[0.35, 0.3, 0.2, 0.1, 0.05]),
        # Categorical - will be high correlation with department_encoded later
        "dept_code": np.random.choice(["ENG", "SAL", "HR", "FIN", None], n,
                                       p=[0.35, 0.3, 0.2, 0.1, 0.05]),
        # Datetime
        "hire_date": pd.date_range("2015-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
        # ID-like high cardinality
        "employee_id": [f"EMP{i:05d}" for i in range(n)],
        # Text
        "notes": np.random.choice(
            ["Good performer", "Needs improvement", "Excellent", None, "On probation"], n
        ),
        # Target variable
        "promoted": np.random.choice([0, 1], n, p=[0.7, 0.3]),
        # Nearly zero variance (should be dropped)
        "constant_col": np.ones(n),
        # Nearly all null (should be dropped)
        "mystery_col": np.where(np.random.rand(n) < 0.85, np.nan, np.random.rand(n)),
    })

    # Inject salary outliers
    df.loc[np.random.choice(n, 15, replace=False), "salary"] = np.random.choice(
        [500000, -10000, 999999], 15
    )

    # Inject age nulls
    df.loc[np.random.choice(n, 30, replace=False), "age"] = np.nan

    # Add ~20 duplicate rows
    dupes = df.sample(20, random_state=seed)
    df = pd.concat([df, dupes], ignore_index=True)

    df.to_csv(output_path, index=False)
    print(f"✓ Sample data written: {output_path}")
    print(f"  Rows: {len(df)} | Columns: {len(df.columns)}")
    print(f"  Columns: {df.columns.tolist()}")
    return output_path


if __name__ == "__main__":
    generate()
