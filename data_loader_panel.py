import os
import re
import pandas as pd
from tabulate import tabulate

# ============================================================
# FILE PATHS
# ============================================================
data_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)) 
    if "__file__" in globals() else os.getcwd(), 
    "data"
)
files = {
    "Active":   os.path.join(data_dir, "Active_companies.xlsx"),
    "Bankrupt": os.path.join(data_dir, "Bankrupt_companies.xlsx"),
}

# ============================================================
# HELPER — CLEAN RAW DATAFRAME ON LOAD
# ============================================================
# Applied once per file immediately after reading from disk.
# Handles: unnamed index columns, incorporation year parsing,
# NACE code zero-padding, financial column type conversion.

def clean_df(df):
    df = df.copy()

    # Drop unnamed index columns Orbis sometimes exports
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    # Incorporation year — handle both Orbis export formats:
    #   "Date of incorporation" → datetime string → extract year
    #   "Year of incorporation" → integer directly
    if "Date of incorporation" in df.columns:
        df["Year of incorporation"] = (
            pd.to_datetime(df["Date of incorporation"], errors="coerce")
            .dt.year.astype("Int16")
        )
        df = df.drop(columns=["Date of incorporation"])
    elif "Year of incorporation" in df.columns:
        df["Year of incorporation"] = (
            pd.to_numeric(df["Year of incorporation"], errors="coerce")
            .astype("Int16")
        )

    # NACE code — preserve as zero-padded 4-character string
    # e.g. integer 711 → "0711" to avoid losing leading zeros
    nace_col = "NACE Rev. 2, core code (4 digits)"
    if nace_col in df.columns:
        df[nace_col] = df[nace_col].astype(str).str.zfill(4)

    # Financial columns — any column ending with a 4-digit year
    # converted to Float64 (nullable float, supports NaN)
    fin_cols = [c for c in df.columns if re.search(r"\d{4}$", c)]
    df[fin_cols] = df[fin_cols].apply(pd.to_numeric, errors="coerce").astype("Float64")

    return df.copy()

# ============================================================
# HELPER — PRINT DATAFRAME STRUCTURE
# ============================================================
def print_df_structure(name, df):
    rows = []
    for i, col in enumerate(df.columns):
        non_null = df[col].notna().sum()
        null     = df[col].isna().sum()
        rows.append([i, col, str(df[col].dtype), f"{non_null:,}", f"{null:,}"])

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Rows: {len(df):,}    Columns: {len(df.columns)}")
    print()
    print(tabulate(
        rows,
        headers=["#", "Column", "Dtype", "Non-Null", "Null"],
        tablefmt="github",
        numalign="right",
        stralign="left",
    ))
    mem_mb = df.memory_usage(deep=True).sum() / 1024**2
    print(f"\n  Memory usage: {mem_mb:.2f} MB\n")

# ============================================================
# LOAD FILES
# ============================================================
df_active   = None
df_bankrupt = None

for label, path in files.items():
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
    try:
        df = pd.read_excel(path)
        df = clean_df(df)
        if label == "Active":
            df_active = df
            df_name   = "df_active"
        else:
            df_bankrupt = df
            df_name     = "df_bankrupt"
        print(f"Loaded: {df_name} ({os.path.basename(path)})")
        print_df_structure(df_name, df)
    except Exception as e:
        print(f"Error loading {path}: {e}")

# ============================================================
# PRESERVE RAW COPIES FOR IDEMPOTENT RE-RUNS
# ============================================================
# Processing blocks always reset from these _raw copies,
# meaning they can be re-run without reloading from disk
# and without accumulating side effects from previous runs.

if df_active is not None:
    df_active_raw = df_active.copy()

if df_bankrupt is not None:
    df_bankrupt_raw = df_bankrupt.copy()

print("✓ Raw copies saved: df_active_raw, df_bankrupt_raw")