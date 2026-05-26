import os
import re
import pandas as pd
from tabulate import tabulate

# File paths
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd(), "data")
files = {
    "Active":   os.path.join(data_dir, "Active_companies.xlsx"),
    "Bankrupt": os.path.join(data_dir, "Bankrupt_companies.xlsx"),
}

def clean_df(df):
    df = df.copy()  # defragment before any column assignments
    # Drop unnamed index column
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])

    # Year of incorporation: extract 4-digit year from datetime or string
    if "Date of incorporation" in df.columns:
        df["Year of incorporation"] = pd.to_datetime(df["Date of incorporation"], errors="coerce").dt.year.astype("Int16")
        df = df.drop(columns=["Date of incorporation"])

    # Convert all financial figure columns (those ending with a 4-digit year) to Float64
    fin_cols = [c for c in df.columns if re.search(r"\d{4}$", c)]
    df[fin_cols] = df[fin_cols].apply(pd.to_numeric, errors="coerce").astype("Float64")

    return df.copy()

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


# Load files
df_active   = None
df_bankrupt = None

for label, path in files.items():
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
    try:
        df = pd.read_excel(path)
        df = clean_df(df)
        print(f"Loaded: {os.path.basename(path)}")
        if label == "Active":
            df_active = df
        else:
            df_bankrupt = df
        print_df_structure(os.path.basename(path), df)
    except Exception as e:
        print(f"Error loading {path}: {e}")
