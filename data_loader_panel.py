# Import necessary libraries
import os
import pandas as pd
import psutil
import inspect
from tabulate import tabulate

# Function to print DataFrame structure

def print_df_structure(df):
    callers_locals = inspect.currentframe().f_back.f_locals
    name = next((var_name for var_name, var_val in callers_locals.items() if var_val is df), "DataFrame")

    rows = []
    for i, col in enumerate(df.columns):
        non_null = f"{df[col].notna().sum():,}"  # formatted with thousand separators
        rows.append([
            i,
            col,
            non_null,
            str(df[col].dtype)
        ])

    print(f"\n{name} structure:\n")
    print(tabulate(
        rows,
        headers=["#", "Column", "Non‑Null Count", "Dtype"],
        tablefmt="github",
        numalign="right",
        stralign="left"
    ))

    mem_gb = df.memory_usage(deep=True).sum() / 1024**3
    print(f"\nMemory usage: {mem_gb:.2f} GB\n")


# Load and summarize the data

print("\n" + "=" * 50 + "\n")
print("Start of data summary")
print("\n" + "=" * 50 + "\n")

file_path = 'rd_panel_data.csv'

# Check if file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
else:
    try:
        # Load the data
        df_panel_data = pd.read_csv(file_path)

        print("File loaded successfully.")
        print("rd_panel_data.csv loaded successfully as df_panel_data.")

        # Print full tabulated structure
        print_df_structure(df_panel_data)

    except Exception as e:
        print(f"Error reading or processing the file: {e}")

print("\n" + "=" * 50 + "\n")
print("End of data loading")
print("\n" + "=" * 50 + "\n")
