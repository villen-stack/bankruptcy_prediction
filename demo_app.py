import streamlit as st
import pandas as pd

st.set_page_config(page_title="Bankruptcy Prediction", layout="wide")

NACE_COL = "NACE Rev. 2, core code (4 digits)"
YEARS = [2024, 2023, 2022]


@st.cache_data
def load_sample():
    return pd.read_csv("results/sample_synthetic.csv", dtype={NACE_COL: str})


# ── Page 1: Data overview (long, scrollable) ──────────────────────────────────
def data_page():
    df = load_sample()

    st.title("Bankruptcy Prediction – Finnish SMEs")
    st.caption("All rows on this page are synthetic simulation data, not real companies.")

    # Key figures
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies (sample)", len(df))
    c2.metric("Columns", df.shape[1])
    c3.metric("NACE codes", df[NACE_COL].nunique())
    c4.metric("Missing 2024 figures", f"{df['Total assets EUR 2024'].isna().mean():.0%}")

    st.divider()

    # Section 1 – raw table
    st.header("1. Sample data")
    st.dataframe(df, width="stretch", hide_index=True)

    st.divider()

    # Section 2 – missing values per year
    st.header("2. Missing financial statements by year")
    st.write("Distressed companies often stop filing, so the most recent years are emptiest.")
    missing = pd.DataFrame(
        {"Missing rows": [df[f"Total assets EUR {y}"].isna().sum() for y in YEARS]},
        index=[str(y) for y in YEARS],
    )
    st.bar_chart(missing)

    st.divider()

    # Section 3 – size of companies
    st.header("3. Total assets 2022 by company (EUR)")
    assets = df.set_index("Company name Latin alphabet")["Total assets EUR 2022"]
    assets.index = assets.index.str.replace(r"SIMULATED COMPANY (\d+).*", r"Co \1", regex=True)
    st.bar_chart(assets)

    st.divider()

    # Section 4 – companies per sector
    st.header("4. Companies per NACE code")
    st.bar_chart(df[NACE_COL].value_counts())

    st.divider()

    # Section 5 – column reference
    st.header("5. Column reference")
    with st.expander("Show column names and types"):
        st.dataframe(
            pd.DataFrame({"Column": df.columns, "Type": df.dtypes.astype(str).values}),
            width="stretch", hide_index=True,
        )


# ── Page 2: About ─────────────────────────────────────────────────────────────
def about_page():
    st.title("About this project")
    st.write(
        "This project predicts bankruptcy among Finnish private companies with an "
        "XGBoost classifier trained on financial statement data from Orbis."
    )
    st.info("Model results and feature importance will be added here.")


pg = st.navigation([
    st.Page(data_page, title="Data overview", default=True),
    st.Page(about_page, title="About"),
])
pg.run()