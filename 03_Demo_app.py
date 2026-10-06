import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="Bankruptcy Prediction", layout="wide")

RESULTS = Path("results")
YEARS = [2022, 2023, 2024]
ITEMS = ["P/L before tax", "Total assets", "Current liabilities",
         "Current assets", "Shareholders funds"]

# ── Column descriptions ──────────────────────────────────────────────────────
RAW_DESC = {
    "Company name Latin alphabet": "Company name (identifier, not used as a feature)",
    "NACE Rev. 2, core code (4 digits)": "Industry classification, 4-digit NACE Rev. 2 code",
    "Year of incorporation": "Year the company was founded",
    "P/L before tax": "Profit or loss before tax for the financial year (EUR)",
    "Total assets": "Balance sheet total (EUR)",
    "Current liabilities": "Debts due within one year (EUR)",
    "Current assets": "Cash, receivables and inventory (EUR)",
    "Shareholders funds": "Equity: share capital plus accumulated profits (EUR)",
}

PANEL_INFO = {
    # column: (group, description)
    "Company name Latin alphabet": ("Identifier", "Company name, used only to group a company's years"),
    "Year of incorporation": ("Identifier", "Year the company was founded"),
    "Company Age": ("Company", "Years since incorporation at the financial year"),
    "Bankrupt": ("Label", "Target: 1 = bankrupt, 0 = active"),
    "NACE Section": ("Company", "Industry section (letter A–S) derived from the NACE code"),
    "NACE Rev. 2, core code (4 digits)": ("Company", "Detailed 4-digit industry code"),
    "Year": ("Identifier", "Financial year of the row"),
    "Current assets": ("Reported", "Cash, receivables and inventory (EUR)"),
    "Current liabilities": ("Reported", "Debts due within one year (EUR)"),
    "P/L before tax": ("Reported", "Profit or loss before tax (EUR)"),
    "Shareholders funds": ("Reported", "Equity (EUR)"),
    "Total assets": ("Reported", "Balance sheet total (EUR)"),
    "FE_total_liabilities": ("Engineered", "Total assets − shareholders' funds"),
    "FE_non_current_assets": ("Engineered", "Total assets − current assets"),
    "FE_debt_ratio": ("Engineered", "Total liabilities / total assets (leverage)"),
    "FE_current_ratio": ("Engineered", "Current assets / current liabilities (liquidity)"),
    "FE_roa": ("Engineered", "Profit before tax / total assets (profitability)"),
    "FE_negative_equity": ("Engineered", "1 if shareholders' funds < 0"),
    "FE_negative_pl": ("Engineered", "1 if the year ended in a loss"),
    "FE_assets_yoy_growth": ("Engineered", "Change in total assets vs prior year (%)"),
    "FE_equity_yoy_change": ("Engineered", "Change in shareholders' funds vs prior year (EUR)"),
}


@st.cache_data
def load(name):
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else None


def raw_base(col):
    """'Total assets EUR 2024' → ('Total assets', 2024); other columns → (col, None)."""
    for item in ITEMS:
        for y in YEARS:
            if col == f"{item} EUR {y}":
                return item, y
    return col, None


# ── Page 1: Data ─────────────────────────────────────────────────────────────
def data_page():
    st.title("Bankruptcy Prediction – Finnish SMEs")
    st.write(
        "Financial statement data for Finnish private companies from Orbis "
        "(Bureau van Dijk / Moody's). The raw data is licensed and not published; "
        "this page describes its structure using aggregate statistics only."
    )

    overview = load("dataset_overview.csv")
    raw = load("raw_columns_summary.csv")
    panel = load("panel_columns_summary.csv")
    if overview is None or raw is None or panel is None:
        st.error("Summary files missing from results/. Run the export cell in 01_Build_dataset.ipynb.")
        return

    o = dict(zip(overview["Metric"], overview["Value"]))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies", f"{int(o['Total companies']):,}")
    c2.metric("Active", f"{int(o['Active companies']):,}")
    c3.metric("Bankrupt", f"{int(o['Bankrupt companies']):,}",
              f"{o['Bankrupt companies'] / o['Total companies']:.1%} of total", delta_color="off")
    c4.metric("Company-year rows", f"{int(o['Panel rows (company-years)']):,}")

    st.divider()

    # ── Section 1: raw export ────────────────────────────────────────────────
    st.header("1. Raw Orbis export")
    st.write(
        f"Two Excel files (active and bankrupt companies) with identical structure: "
        f"**one row per company**, {len(raw)} columns. Each financial item appears once "
        f"per year ({YEARS[0]}–{YEARS[-1]}), in wide format."
    )

    raw = raw.copy()
    raw[["Item", "Year"]] = raw["Column"].apply(lambda c: pd.Series(raw_base(c)))
    raw["Description"] = raw["Item"].map(RAW_DESC)
    st.dataframe(
        raw[["Column", "Dtype", "Description", "Active missing %", "Bankrupt missing %"]],
        hide_index=True, width="stretch",
        column_config={
            "Active missing %": st.column_config.ProgressColumn(
                "Active missing %", format="%.1f%%", min_value=0, max_value=100),
            "Bankrupt missing %": st.column_config.ProgressColumn(
                "Bankrupt missing %", format="%.1f%%", min_value=0, max_value=100),
        },
    )

    st.subheader("Missing financial statements by year")
    st.write(
        "Bankrupt companies often stop publishing accounts, so their most recent years "
        "are frequently empty. This gap is itself informative and is kept in the data."
    )
    fin = raw[raw["Year"].notna()]
    missing_by_year = (
        fin.groupby("Year")[["Active missing %", "Bankrupt missing %"]].mean()
        .rename(columns={"Active missing %": "Active", "Bankrupt missing %": "Bankrupt"})
    )
    missing_by_year.index = missing_by_year.index.astype(int).astype(str)
    st.bar_chart(missing_by_year, stack=False, y_label="Missing (%)", x_label="Financial year")

    st.divider()

    # ── Section 2: processing ────────────────────────────────────────────────
    st.header("2. From export to model-ready panel")
    st.markdown(
        """
1. **Merge and label:** the two files are combined; `Bankrupt` = 1 for the bankrupt file, 0 for the active file.
2. **Industry section:** each 4-digit NACE code is mapped to its section letter (A–S).
3. **Reshape wide → long:** one row per company **per year** instead of one row per company.
4. **Company age:** financial year minus year of incorporation.
5. **Feature engineering:** nine financial ratios, distress flags and year-over-year trends (prefix `FE_`).
        """
    )

    st.divider()

    # ── Section 3: panel ─────────────────────────────────────────────────────
    st.header("3. Model-ready panel")
    st.write(
        f"**One row per company and financial year**, {len(panel)} columns. Medians are "
        "shown separately for active and bankrupt companies to show how the groups differ."
    )

    panel = panel.copy()
    panel["Group"] = panel["Column"].map(lambda c: PANEL_INFO.get(c, ("", ""))[0])
    panel["Description"] = panel["Column"].map(lambda c: PANEL_INFO.get(c, ("", ""))[1])

    groups = ["All", "Reported", "Engineered", "Company", "Label", "Identifier"]
    choice = st.radio("Show columns", groups, horizontal=True)
    view = panel if choice == "All" else panel[panel["Group"] == choice]

    st.dataframe(
        view[["Column", "Group", "Dtype", "Description", "Missing %",
              "Median active", "Median bankrupt"]],
        hide_index=True, width="stretch",
        column_config={
            "Missing %": st.column_config.ProgressColumn(
                "Missing %", format="%.1f%%", min_value=0, max_value=100),
            "Median active": st.column_config.NumberColumn(format="%.3g"),
            "Median bankrupt": st.column_config.NumberColumn(format="%.3g"),
        },
    )
    st.caption(
        "Year-over-year features are missing for each company's first observed year, "
        "since there is no prior year to compare with."
    )


# ── Page 2: About ────────────────────────────────────────────────────────────
def about_page():
    st.title("About this project")
    st.write(
        "This project predicts bankruptcy among Finnish private companies, comparing "
        "a logistic regression benchmark with an XGBoost classifier trained on "
        "financial statement data from Orbis."
    )
    st.info("Model results and feature importance will be added here.")


pg = st.navigation([
    st.Page(data_page, title="Data", default=True),
    st.Page(about_page, title="About"),
])
pg.run()