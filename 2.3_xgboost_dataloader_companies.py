import pandas as pd
from pathlib import Path
from tabulate import tabulate


def load_final_panel(csv_path: str | Path = Path("data") / "final_panel.csv") -> pd.DataFrame:
    """Load the final panel CSV file from the data folder into a pandas DataFrame."""
    csv_path = Path(csv_path)
    df_final_panel = pd.read_csv(csv_path)
    return df_final_panel


def print_df_structure(name: str, df: pd.DataFrame) -> None:
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


if __name__ == "__main__":
    csv_path = Path("data") / "final_panel.csv"
    df_final_panel = load_final_panel(csv_path)
    print(f"Loaded: {csv_path}")
    print_df_structure("final_panel", df_final_panel)
