"""
Finnish Corporate Bankruptcy Prediction — Streamlit app.

Reads only the aggregate result files in results/, written by
01_Build_dataset.ipynb and 02_Train_and_evaluate.ipynb. No
company-level data is used or shown.

Run locally from the project folder:
    streamlit run 03_Demo_app.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

st.set_page_config(page_title="Bankruptcy Prediction", page_icon="📉", layout="wide")

RESULTS = Path(__file__).parent / "results"
GITHUB_URL = "https://github.com/villen-stack/bankruptcy_prediction"   # ← check the repo name

ACTIVE_COLOR   = "#189AB4"
BANKRUPT_COLOR = "#E4572E"
CLASS_COLORS   = {"Active": ACTIVE_COLOR, "Bankrupt": BANKRUPT_COLOR}

# ── Column descriptions for the data tables ──────────────────────────────────
RAW_ITEMS = {
    "P/L before tax":      "Profit or loss before tax for the financial year (EUR)",
    "Total assets":        "Balance sheet total (EUR)",
    "Current liabilities": "Debts due within one year (EUR)",
    "Current assets":      "Cash, receivables and inventory (EUR)",
    "Shareholders funds":  "Equity: share capital plus accumulated profits (EUR)",
}
RAW_OTHER = {
    "Company name Latin alphabet":       "Company name (identifier only)",
    "NACE Rev. 2, core code (4 digits)": "Industry classification, 4-digit NACE Rev. 2 code",
    "Year of incorporation":             "Year the company was founded",
}

MODEL_COLUMNS = {
    # column: (group, description)
    "company_id":                  ("Identifier", "Unique company key (names are not unique)"),
    "Company name Latin alphabet": ("Identifier", "Company name, for reference only"),
    "Bankrupt":                    ("Label",      "Target: 1 = bankrupt, 0 = active"),
    "Year":                        ("Reference",  "Year of the latest accounts (not a model input)"),
    "n_years_reported":            ("Reference",  "Years of accounts available (not a model input)"),
    "NACE Section":                ("Company",    "Industry section derived from the NACE code"),
    "Company Age":                 ("Company",    "Years since incorporation, at the latest accounts"),
    "Total assets":                ("Reported",   "Balance sheet total (EUR)"),
    "Current assets":              ("Reported",   "Cash, receivables and inventory (EUR)"),
    "Current liabilities":         ("Reported",   "Debts due within one year (EUR)"),
    "Shareholders funds":          ("Reported",   "Equity (EUR)"),
    "P/L before tax":              ("Reported",   "Profit or loss before tax (EUR)"),
    "FE_total_liabilities":        ("Engineered", "Total assets − shareholders' funds"),
    "FE_non_current_assets":       ("Engineered", "Total assets − current assets"),
    "FE_debt_ratio":               ("Engineered", "Total liabilities / total assets (leverage)"),
    "FE_current_ratio":            ("Engineered", "Current assets / current liabilities (liquidity)"),
    "FE_roa":                      ("Engineered", "Profit before tax / total assets (profitability)"),
    "FE_negative_equity":          ("Engineered", "1 if shareholders' funds are negative"),
    "FE_negative_pl":              ("Engineered", "1 if the year ended in a loss"),
    "FE_assets_yoy_growth":        ("Engineered", "Change in total assets vs the previous year, relative"),
    "FE_equity_yoy_change":        ("Engineered", "Change in equity vs the previous year, relative to total assets"),
    "FE_debt_ratio_change":        ("Engineered", "Debt ratio this year minus last year"),
    "FE_roa_change":               ("Engineered", "Return on assets this year minus last year"),
}


# ── Helpers ──────────────────────────────────────────────────────────────────
FEATURE_NAMES = {
    "FE_debt_ratio": "Debt ratio", "FE_current_ratio": "Current ratio",
    "FE_roa": "Return on assets", "FE_negative_equity": "Negative equity",
    "FE_negative_pl": "Loss-making year", "FE_assets_yoy_growth": "Asset growth (YoY)",
    "FE_equity_yoy_change": "Equity change (YoY)", "FE_debt_ratio_change": "Debt ratio change",
    "FE_roa_change": "ROA change", "FE_total_liabilities": "Total liabilities",
    "FE_non_current_assets": "Non-current assets",
}


def feature_name(col):
    """Readable name for a model feature."""
    return FEATURE_NAMES.get(col, col)


@st.cache_data
def load(name):
    path = RESULTS / name
    return pd.read_csv(path) if path.exists() else None


def require(*names):
    """Load result files; stop the page with a clear message if any is missing."""
    frames = [load(n) for n in names]
    missing = [n for n, f in zip(names, frames) if f is None]
    if missing:
        st.error(
            "Missing result files in `results/`: " + ", ".join(f"`{m}`" for m in missing)
            + ". Run `01_Build_dataset.ipynb` and `02_Train_and_evaluate.ipynb` first."
        )
        st.stop()
    return frames


def split_raw_column(col):
    """'Total assets EUR 2024' → ('Total assets', 2024); other columns → (col, None)."""
    for item in RAW_ITEMS:
        prefix = f"{item} EUR "
        if col.startswith(prefix) and col[len(prefix):].isdigit():
            return item, int(col[len(prefix):])
    return col, None


def style(fig, height=380):
    has_title = bool(fig.layout.title.text)
    fig.update_layout(
        height=height,
        # Title on top, legend in its own row just above the plot area
        margin=dict(l=10, r=10, t=85 if has_title else 40, b=10),
        title=dict(y=0.98, yanchor="top", x=0, xanchor="left", pad=dict(l=5)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        plot_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="#EEEEEE")
    return fig


def rate_chart(table, group_col, level, title, horizontal=False):
    """Bar chart of bankruptcy rate per group (sample or population level)."""
    col = "rate_sample" if level == "Sample" else "rate_population"
    t = table.dropna(subset=[col]).copy()
    t["pct"] = t[col] * 100
    text = [f"{p:.1f}%" if level == "Sample" else f"{p:.2f}%" for p in t["pct"]]
    hover = [f"{g}<br>{p:.2f}% bankrupt<br>{n:,} companies"
             for g, p, n in zip(t[group_col], t["pct"], t["companies"])]
    if horizontal:
        t = t.iloc[::-1]
        text, hover = text[::-1], hover[::-1]
        bar = go.Bar(x=t["pct"], y=t[group_col], orientation="h", text=text,
                     hovertext=hover, hoverinfo="text", marker_color=BANKRUPT_COLOR)
    else:
        bar = go.Bar(x=t[group_col].astype(str), y=t["pct"], text=text,
                     hovertext=hover, hoverinfo="text", marker_color=BANKRUPT_COLOR)
    fig = go.Figure(bar)
    fig.update_traces(textposition="outside", cliponaxis=False)
    axis_title = "% of companies bankrupt"
    if horizontal:
        fig.update_layout(xaxis_title=axis_title)
    else:
        fig.update_layout(yaxis_title=axis_title)
    fig.update_layout(title=title)
    return style(fig, height=420 if horizontal else 360)


# ── Page 1: Data ─────────────────────────────────────────────────────────────
def data_page():
    st.title("Bankruptcy Prediction — Finnish SMEs")
    st.write(
        "How do companies that went bankrupt differ from those that stayed solvent? "
        "This page describes the data behind the model, using aggregate statistics only."
    )
    st.info(
        "The company data comes from Moody's (Orbis database), accessed under an academic "
        "licence through Tampere University of Applied Sciences (TAMK). It cannot be "
        "redistributed, so the app shows summaries, never individual companies.",
        icon="ℹ️",
    )

    overview, raw, model_cols, summary = require(
        "dataset_overview.csv", "raw_columns_summary.csv", "model_columns_summary.csv",
        "results_summary.csv")
    o = dict(zip(overview["Metric"], overview["Value"]))
    v = dict(zip(summary["item"], summary["value"]))

    pop_total    = int(float(v["Companies in population"]))
    pop_bankrupt = int(float(v["Bankrupt companies in population"]))
    pop_active   = pop_total - pop_bankrupt
    pop_rate     = pop_bankrupt / pop_total

    n_total    = int(o["Total companies (model data)"])
    n_active   = int(o["Active companies (model data)"])
    n_bankrupt = int(o["Bankrupt companies (model data)"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies", f"{n_total:,}")
    c2.metric("Active", f"{n_active:,}")
    c3.metric("Bankrupt", f"{n_bankrupt:,}")
    c3.caption(f"{n_bankrupt / n_total:.1%} of the sample")
    c4.metric("Bankrupt in the population", f"{pop_rate:.2%}")
    c4.caption(f"of {pop_total:,} Finnish companies")

    # ── 1. Sample design ─────────────────────────────────────────────────────
    st.header("1. Where the data comes from")
    st.markdown(
        f"""
The population is **{pop_total:,} Finnish limited companies** of SME size: revenue ≥ €200k
and total assets between €100k and €43M (the EU SME balance-sheet limit), with finance, public
sector, education, health and energy excluded. Of these, **{pop_rate:.2%} are bankrupt**.

Bankruptcy is rare, and a single export is capped in size. The sample therefore keeps
**every bankrupt company** and a **random 1 in 7** of the active ones. This gives the model
enough failure cases to learn from, but makes bankruptcies about 6× more common in the
sample than in reality. The model's probabilities are corrected back to the population
rate before they are read as real-world risk.

| | Population | Exported | In model data |
|---|---:|---:|---:|
| Active companies | {pop_active:,} | {int(o['Active companies (export)']):,} | {n_active:,} |
| Bankrupt companies | {pop_bankrupt:,} | {int(o['Bankrupt companies (export)']):,} | {n_bankrupt:,} |
| Bankruptcy share | {pop_rate:.2%} | {int(o['Bankrupt companies (export)']) / (int(o['Active companies (export)']) + int(o['Bankrupt companies (export)'])):.1%} | {n_bankrupt / n_total:.1%} |
"""
    )
    st.caption(
        "Exported vs model data: duplicate exports are removed, and companies with total "
        "assets above €43M are excluded. The export had no upper size limit, so the population "
        "counts for the SME scope are estimated from the share of sample companies kept."
    )

    # ── 2. Raw export ────────────────────────────────────────────────────────
    st.header("2. The raw export")
    raw = raw.copy()
    split = raw["Column"].apply(split_raw_column)
    raw["Item"] = split.str[0]
    raw["Year"] = split.str[1]
    years = sorted(int(y) for y in raw["Year"].dropna().unique())
    st.write(
        f"Two Excel files, one for active and one for bankrupt companies, with the same "
        f"structure: **one row per company** and each financial item repeated for every "
        f"year ({years[0]}–{years[-1]})."
    )

    st.subheader("Missing accounts by year")
    st.write(
        "Bankrupt companies stop publishing accounts after they fail, so their latest years "
        "are often empty. That is why the model never uses the calendar year or the number "
        "of years reported: they would reveal *when the records end*, not how healthy the "
        "company was."
    )
    fin = raw[raw["Year"].notna()]
    by_year = fin.groupby("Year")[["Active missing %", "Bankrupt missing %"]].mean()
    fig = go.Figure()
    for cls in ["Active", "Bankrupt"]:
        fig.add_bar(x=[str(int(y)) for y in by_year.index], y=by_year[f"{cls} missing %"],
                    name=cls, marker_color=CLASS_COLORS[cls],
                    text=[f"{v:.0f}%" for v in by_year[f"{cls} missing %"]],
                    textposition="outside", cliponaxis=False)
    fig.update_layout(barmode="group", yaxis_title="Companies with no accounts (%)",
                      xaxis_title="Financial year")
    st.plotly_chart(style(fig, 340), width="stretch")

    with st.expander("All columns in the raw export"):
        raw["Description"] = raw["Item"].map({**RAW_ITEMS, **RAW_OTHER})
        st.dataframe(
            raw[["Column", "Dtype", "Description", "Active missing %", "Bankrupt missing %"]],
            hide_index=True, width="stretch",
            column_config={
                c: st.column_config.ProgressColumn(c, format="%.1f%%", min_value=0, max_value=100)
                for c in ["Active missing %", "Bankrupt missing %"]
            },
        )

    # ── 3. Model data ────────────────────────────────────────────────────────
    st.header("3. From export to model data")
    st.markdown(
        """
1. **Merge and label:** the two files are combined; `Bankrupt` = 1 for the bankrupt file, 0 for the active file.
2. **Clean:** duplicate exports are removed and each company gets a unique ID (several companies share a name).
3. **Reshape:** one row per company per year, so features can be calculated year by year.
4. **Feature engineering:** financial ratios, distress flags and year-over-year changes (prefix `FE_`).
5. **One row per company:** each company is reduced to its **latest year of accounts**, with the
   year-over-year changes carrying its recent history.
        """
    )

    mc = model_cols.copy()
    mc["Group"]       = mc["Column"].map(lambda c: MODEL_COLUMNS.get(c, ("", ""))[0])
    mc["Description"] = mc["Column"].map(lambda c: MODEL_COLUMNS.get(c, ("", ""))[1])
    # Identifiers: medians are meaningless, and the name column has none
    mc = mc[mc["Column"] != "Company name Latin alphabet"]

    groups = ["All", "Reported", "Engineered", "Company", "Reference", "Label", "Identifier"]
    choice = st.radio("Show columns", groups, horizontal=True)
    view = mc if choice == "All" else mc[mc["Group"] == choice]
    st.dataframe(
        view[["Column", "Group", "Description", "Missing %", "Median active", "Median bankrupt"]],
        hide_index=True, width="stretch",
        column_config={
            "Missing %": st.column_config.ProgressColumn(
                "Missing %", format="%.1f%%", min_value=0, max_value=100),
            "Median active":   st.column_config.NumberColumn(format="%.3g"),
            "Median bankrupt": st.column_config.NumberColumn(format="%.3g"),
        },
    )
    st.caption(
        "Year-over-year features are missing for companies with only one year of accounts."
    )

    # ── 4. What the data shows ───────────────────────────────────────────────
    st.header("4. Bankrupt vs active companies")
    quartiles, flags, by_industry, by_age, by_size = require(
        "desc_ratio_quartiles.csv", "desc_distress_flags.csv",
        "desc_rate_by_industry.csv", "desc_rate_by_age.csv", "desc_rate_by_size.csv")

    st.subheader("Key ratios")
    st.write(
        "Each box spans the middle 50% of companies, with the line at the median. "
        "Whiskers reach the furthest values within 1.5× the box height; extreme values are not shown."
    )
    labels = quartiles.drop_duplicates("feature")[["feature", "label"]]
    picked = st.multiselect(
        "Ratios", labels["label"].tolist(),
        default=[l for l in ["Debt ratio", "Current ratio", "Return on assets",
                             "Equity change / assets (YoY)"] if l in labels["label"].tolist()],
    )
    if picked:
        cols = st.columns(min(len(picked), 4))
        for i, label in enumerate(picked):
            q = quartiles[quartiles["label"] == label]
            fig = go.Figure()
            for _, r in q.iterrows():
                fig.add_trace(go.Box(
                    name=r["class"], x=[r["class"]],
                    q1=[r["q1"]], median=[r["median"]], q3=[r["q3"]],
                    lowerfence=[r["whisker_low"]], upperfence=[r["whisker_high"]],
                    marker_color=CLASS_COLORS[r["class"]], boxpoints=False,
                    hoverinfo="y",
                ))
            fig.update_layout(title=label, showlegend=False)
            cols[i % len(cols)].plotly_chart(style(fig, 340), width="stretch")

    st.subheader("Distress flags")
    fig = go.Figure()
    for cls in ["Active", "Bankrupt"]:
        f = flags[flags["class"] == cls]
        fig.add_bar(x=f["flag"], y=f["pct_of_companies"], name=cls,
                    marker_color=CLASS_COLORS[cls],
                    text=[f"{v:.0f}%" for v in f["pct_of_companies"]],
                    textposition="outside", cliponaxis=False)
    fig.update_layout(barmode="group", yaxis_title="% of companies", yaxis_range=[0, 100])
    st.plotly_chart(style(fig, 340), width="stretch")

    st.subheader("Where bankruptcy is more common")
    level = st.radio(
        "Show rates as", ["Sample", "Population"], horizontal=True,
        help="Sample: share of companies in this dataset (bankruptcies over-represented). "
             "Population: corrected to the real-world base rate.",
    )
    st.plotly_chart(rate_chart(by_industry, "Industry", level, "By industry", horizontal=True),
                    width="stretch")
    c1, c2 = st.columns(2)
    c1.plotly_chart(rate_chart(by_age, "Age band", level, "By company age (years)"),
                    width="stretch")
    c2.plotly_chart(rate_chart(by_size, "Size band", level, "By company size (total assets)"),
                    width="stretch")
    st.caption("Groups with fewer than 20 companies are not shown.")


# ── Page: Overview ───────────────────────────────────────────────────────────
def overview_page():
    st.title("Bankruptcy Prediction — Finnish SMEs")
    st.markdown(
        """
When a private company runs into trouble, there is no share price to signal it. Lenders,
suppliers and auditors have to judge financial health from annual accounts alone.

This project asks **how well a company's latest financial statements can tell apart firms
that went bankrupt from those that stayed solvent**. It trains an **XGBoost** model on
about 10,000 Finnish companies and uses **SHAP** to show which financial signals drive
each prediction.
        """
    )

    summary = load("results_summary.csv")
    if summary is not None:
        v = dict(zip(summary["item"], summary["value"]))
        c1, c2, c3 = st.columns(3)
        c1.metric("Companies", f"{int(float(v['Companies in model data'])):,}")
        c2.metric("Test ROC-AUC", f"{float(v['Test ROC-AUC']):.3f}")
        c2.caption("on companies the model never saw")
        c3.metric("Strongest signal", feature_name(v["Top feature 1"]))
        c3.caption("largest average SHAP contribution")

    st.subheader("In this app")
    st.markdown(
        """
1. **Overview:** this page.
2. **Data:** where the data comes from, how it was prepared, and how bankrupt and active
   companies differ.
3. **Model results:** how well the model performs and what drives its predictions.
4. **Risk calculator:** set a company profile with sliders and get a risk estimate.
        """
    )

    st.link_button("Code, notebooks and full write-up on GitHub", GITHUB_URL)
    st.caption(
        "Portfolio and technology demonstration. Company data: Moody's (Orbis), academic "
        "licence via TAMK; only aggregate results are shown."
    )


# ── Page: Model results ──────────────────────────────────────────────────────
def results_page():
    st.title("Model results")
    st.write(
        "How well the model separates bankrupt from active companies, how stable that is, "
        "and which financial signals drive its predictions."
    )

    (summary, folds, learning, metrics, roc, pr, cm, calib,
     importance, profile, dependence) = require(
        "results_summary.csv", "cv_folds.csv", "learning_curve.csv", "test_metrics.csv",
        "roc_curve.csv", "pr_curve.csv", "confusion_matrix.csv", "calibration.csv",
        "shap_importance.csv", "shap_profile_by_class.csv", "shap_dependence.csv")
    v = dict(zip(summary["item"], summary["value"]))
    m = metrics.set_index("metric")

    auc      = m.loc["ROC-AUC", "sample"]
    cv_auc   = float(v["CV ROC-AUC (validation, mean)"])
    cv_std   = float(v["CV ROC-AUC (validation, std)"])
    recall   = m.loc["Recall at threshold", "sample"]
    prec_s   = m.loc["Precision at threshold", "sample"]
    prec_p   = m.loc["Precision at threshold", "population"]
    ap_s     = m.loc["Average Precision", "sample"]
    rate_s   = m.loc["Bankruptcy rate", "sample"]
    rate_p   = m.loc["Bankruptcy rate", "population"]

    # Learning curve summary for the explanation box
    lc_first, lc_last = learning.iloc[0], learning.iloc[-1]
    lc_gain = lc_last["val_roc_auc"] - lc_first["val_roc_auc"]
    lc_prev_gain = lc_last["val_roc_auc"] - learning.iloc[-2]["val_roc_auc"]
    lc_verdict = (
        "it is still rising at the end, so more data would likely improve the model further"
        if lc_prev_gain > 0.005 else
        "it has levelled off at the end, so more companies alone would add little; better "
        "information (more features or history) matters more"
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test ROC-AUC", f"{auc:.3f}",
              help="How well the model ranks risk: the chance that a randomly chosen bankrupt "
                   "company gets a higher score than a randomly chosen active one.")
    c1.caption("0.5 = coin flip, 1.0 = perfect")
    c2.metric("Cross-validation ROC-AUC", f"{cv_auc:.3f}",
              help="The same measure, averaged over 5 rounds of training and checking on "
                   "different parts of the training data. Shows how stable the result is.")
    c2.caption(f"± {cv_std:.3f} across 5 folds")
    c3.metric("Bankruptcies caught", f"{recall:.0%}",
              help="Recall: the share of companies that went bankrupt which the model flags "
                   "at the chosen cut-off.")
    c3.caption("recall at the chosen threshold")
    c4.metric("Average Precision", f"{ap_s:.3f}",
              help="How clean the list of flagged companies is, averaged over all cut-offs. "
                   "Random guessing scores the bankruptcy rate.")
    c4.caption(f"random guessing: {rate_s:.3f}")

    with st.expander("How to read these numbers"):
        st.markdown(
            f"""
**ROC-AUC (Area Under the Receiver Operating Characteristic curve).** Take one company that
went bankrupt and one that stayed active, at random. ROC-AUC is the probability that the
model gives the bankrupt one the higher risk score. 0.5 is a coin flip, 1.0 is perfect. This
model scores **{auc:.2f}**: it puts the bankrupt company ahead in about
{auc * 100:.0f} out of 100 such pairs. It measures *ranking* only, so it does not depend on
how common bankruptcy is.

**Cross-validation.** The training companies are split into 5 parts. The model is trained on
four and tested on the fifth, five times over, so every company is tested once by a model
that never saw it. The average ({cv_auc:.2f}) estimates performance; the spread (±{cv_std:.3f})
shows whether it depends on which companies happen to be in the test part. The final test set
is kept separate and used only once, at the end.

**Learning curve.** The model is retrained on growing shares of the training companies
({int(lc_first['train_size']):,} up to {int(lc_last['train_size']):,}) and scored with
cross-validation each time. Two lines are shown: the score on the companies it trained on, and
on companies it did not see (validation). Validation ROC-AUC goes from
{lc_first['val_roc_auc']:.2f} to {lc_last['val_roc_auc']:.2f} ({lc_gain:+.2f}); {lc_verdict}.
A persistent gap between the two lines is overfitting: the model fits the companies it has seen
better than new ones. Validation is the honest number.

**Threshold.** The model gives every company a probability. To get a yes/no answer, companies
above a cut-off are flagged. A low cut-off catches more bankruptcies but also flags more healthy
companies; a high one gives fewer false alarms but misses more failures. The usual default of
0.5 does not suit rare events: with only about {rate_s:.0%} of companies bankrupt, the model
rarely gives anyone over 50%, so it would miss most bankruptcies. Here the cut-off is the one
that best balances catching bankruptcies against false alarms (maximum F1), chosen on the
cross-validation results, never on the test set. In practice the right cut-off depends on the
user: a lender who loses money on every missed bankruptcy would choose a lower one than a
supplier screening many small customers. The cut-off only affects precision, recall and F1;
ROC-AUC and Average Precision cover all cut-offs.

**Recall (bankruptcies caught).** Of the companies that went bankrupt, the share the model
flags: **{recall:.0%}** here.

**Precision.** Of the companies the model flags, the share that really went bankrupt:
**{prec_s:.0%}** in the sample. Precision depends on how common bankruptcy is. In the sample
about {rate_s:.0%} of companies are bankrupt; in reality only {rate_p:.1%}, so the same flags
contain many more healthy companies and precision drops to about **{prec_p:.0%}**. Use the
Sample / Population switch below to see both.

**Average Precision (PR-AUC).** Precision averaged over every possible cut-off. Random
guessing scores the bankruptcy rate ({rate_s:.2f} in the sample), so the gap to that number
shows how much the model adds.

**Why not accuracy?** Calling every company "active" would already be right about
{1 - rate_s:.0%} of the time while catching no bankruptcies at all.

**SHAP.** A method that splits each prediction into a contribution from every input, showing
*why* the model scored a company as it did. See *How to read the SHAP charts* in section 5.
            """
        )

    # ── 1. How the model was evaluated ───────────────────────────────────────
    st.header("1. How the model was evaluated")
    st.markdown(
        """
- **Held-out test set:** 20% of companies are set aside before training and scored once
  at the end, so the results describe companies the model has never seen.
- **Cross-validation:** the other 80% are split into 5 parts; the model is trained on four
  and checked on the fifth, five times over. This shows how stable the performance is.
- **Threshold:** the cut-off for flagging a company is chosen on the cross-validation
  results, not on the test set, so the test score is not tuned to the test data.
        """
    )

    # ── 2. Stability and overfitting ─────────────────────────────────────────
    st.header("2. Stability and overfitting")
    c1, c2 = st.columns(2)

    metric_choice = c1.radio("Metric", ["ROC-AUC", "Average Precision"], horizontal=True)
    key = "roc_auc" if metric_choice == "ROC-AUC" else "avg_precision"
    fig = go.Figure()
    fig.add_bar(x=[f"Fold {f}" for f in folds["fold"]], y=folds[f"train_{key}"],
                name="Training", marker_color="#B0BEC5")
    fig.add_bar(x=[f"Fold {f}" for f in folds["fold"]], y=folds[f"val_{key}"],
                name="Validation", marker_color=ACTIVE_COLOR,
                text=[f"{x:.3f}" for x in folds[f"val_{key}"]],
                textposition="outside", cliponaxis=False)
    fig.update_layout(title=f"{metric_choice} per fold", barmode="group",
                      yaxis_range=[0, 1.1])
    c1.plotly_chart(style(fig, 360), width="stretch")
    gap = folds[f"train_{key}"].mean() - folds[f"val_{key}"].mean()
    c1.caption(
        f"Average gap between training and validation: {gap:.3f}. The model fits the "
        "companies it learned from better than new ones; validation is the honest number."
    )

    fig = go.Figure()
    for col, name, color in [("train_roc_auc", "Training", "#B0BEC5"),
                             ("val_roc_auc", "Validation", ACTIVE_COLOR)]:
        std = col.replace("roc_auc", "std")
        fig.add_scatter(x=learning["train_size"], y=learning[col], name=name,
                        mode="lines+markers", line_color=color,
                        error_y=dict(type="data", array=learning[std], visible=True))
    fig.update_layout(title="Learning curve (ROC-AUC)",
                      xaxis_title="Companies used for training")
    c2.plotly_chart(style(fig, 360), width="stretch")
    c2.caption(
        "If validation still rises at the right edge, more data would help; if it has "
        "levelled off, the model is limited by the information in the features."
    )

    # ── 3. Test set performance ──────────────────────────────────────────────
    st.header("3. Performance on unseen companies")
    level = st.radio(
        "Base rate", ["Sample", "Population"], horizontal=True,
        help="Sample: bankruptcies are over-represented (about 11%). Population: re-estimated "
             "for the real-world rate. Ranking (ROC-AUC) and recall do not change; precision does.",
    )
    lvl = level.lower()

    c1, c2 = st.columns(2)
    fig = go.Figure()
    fig.add_scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="XGBoost",
                    line=dict(color=ACTIVE_COLOR, width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                    line=dict(color="#999999", dash="dash"))
    fig.update_layout(title=f"ROC curve (AUC {m.loc['ROC-AUC', 'sample']:.3f})",
                      xaxis_title="Share of active companies flagged (false positive rate)",
                      yaxis_title="Share of bankruptcies caught")
    c1.plotly_chart(style(fig, 400), width="stretch")
    c1.caption("Each point is one possible cut-off. The further the curve bows to the top-left, "
               "the better the model ranks bankrupt companies above active ones.")

    p = pr[pr["level"] == lvl]
    base = m.loc["Bankruptcy rate", lvl]
    fig = go.Figure()
    fig.add_scatter(x=p["recall"], y=p["precision"], mode="lines", name="XGBoost",
                    line=dict(color=BANKRUPT_COLOR, width=3))
    fig.add_hline(y=base, line_dash="dash", line_color="#999999",
                  annotation_text=f"random ({base:.1%})", annotation_position="top left")
    fig.add_scatter(x=[m.loc["Recall at threshold", lvl]],
                    y=[m.loc["Precision at threshold", lvl]], mode="markers",
                    name="Chosen threshold", marker=dict(size=12, color="#333333"))
    fig.update_layout(title=f"Precision–recall curve (AP {m.loc['Average Precision', lvl]:.3f})",
                      xaxis_title="Share of bankruptcies caught (recall)",
                      yaxis_title="Share of flagged companies that fail (precision)",
                      yaxis_range=[0, 1.05])
    c2.plotly_chart(style(fig, 400), width="stretch")
    c2.caption("Catching more bankruptcies means flagging more companies, and a larger share "
               "of them turn out to be healthy. At population level, healthy companies are far "
               "more common, so precision is lower.")

    c1, c2 = st.columns([1, 1.2])
    grid = cm.pivot(index="actual", columns="predicted", values="companies").loc[
        ["Bankrupt", "Active"], ["Active", "Bankrupt"]]
    fig = go.Figure(go.Heatmap(
        z=grid.values, x=["Predicted active", "Predicted bankrupt"],
        y=["Actually bankrupt", "Actually active"], colorscale="Blues", showscale=False,
        text=[[f"{x:,}" for x in row] for row in grid.values],
        texttemplate="%{text}", textfont=dict(size=18),
    ))
    fig.update_layout(title="Test companies at the chosen threshold (sample)")
    c1.plotly_chart(style(fig, 340), width="stretch")

    tbl = metrics.copy()
    tbl.columns = ["Metric", "Sample", "Population"]
    c2.dataframe(tbl, hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.3f")
                                for c in ["Sample", "Population"]})
    c2.caption(f"The threshold is a sample-level probability of "
               f"{m.loc['Threshold (probability)', 'sample']:.3f}, which equals "
               f"{m.loc['Threshold (probability)', 'population']:.3f} at population level.")

    # ── Reading the chosen cut-off ───────────────────────────────────────────
    tn = int(grid.loc["Active", "Active"]);   fp = int(grid.loc["Active", "Bankrupt"])
    fn = int(grid.loc["Bankrupt", "Active"]); tp = int(grid.loc["Bankrupt", "Bankrupt"])
    st.subheader("What the cut-off means")
    st.markdown(
        f"""
The model gives every company a probability; companies above the cut-off are **flagged**.
At the chosen cut-off, the model flags **{tp + fp:,}** of the {tp + fp + tn + fn:,} test
companies. **{tp:,}** of them really went bankrupt, so it catches **{tp:,} of {tp + fn:,}
bankruptcies** ({tp / (tp + fn):.0%}), while **{fp:,}** flagged companies stayed active and
**{fn:,}** bankruptcies were missed.

Where to put the cut-off is a trade-off: lower it and more bankruptcies are caught, but more
healthy companies are flagged too. The cut-off used here is the one that best balances the two
(maximum F1), chosen on the training data. The usual default of 0.5 would miss most
bankruptcies, because the model rarely gives a company more than a 50% probability when only
about {rate_s:.0%} of companies are bankrupt.
        """
    )

    # ── How the cut-off was chosen ───────────────────────────────────────────
    curve = load("threshold_curve.csv")
    if curve is not None:
        st.markdown(
            """
**How the cut-off was chosen.** Every training company was scored by a model that had not
seen it (cross-validation). For each possible cut-off, the notebook counted how many bankrupt
companies would be caught (recall) and how many of the flagged companies really failed
(precision). F1 combines the two and is only high when *both* are reasonably high:

`F1 = 2 × precision × recall / (precision + recall)`

The cut-off with the highest F1 was chosen, then applied unchanged to the test companies.
            """
        )
        chosen = curve[curve["chosen"]].iloc[0]
        c1, c2 = st.columns([1.4, 1])
        fig = go.Figure()
        for col_name, name, color, width in [("recall", "Recall (bankruptcies caught)", BANKRUPT_COLOR, 2),
                                             ("precision", "Precision (flagged that fail)", ACTIVE_COLOR, 2),
                                             ("f1", "F1 (balance of the two)", "#333333", 4)]:
            fig.add_scatter(x=curve["cutoff"], y=curve[col_name], mode="lines", name=name,
                            line=dict(color=color, width=width))
        fig.add_vline(x=chosen["cutoff"], line_dash="dash", line_color="#999999",
                      annotation_text=f"chosen {chosen['cutoff']:.3f}",
                      annotation_position="top")
        fig.add_scatter(x=[chosen["cutoff"]], y=[chosen["f1"]], mode="markers",
                        marker=dict(size=12, color="#333333"), showlegend=False,
                        hovertemplate="F1 %{y:.3f}<extra></extra>")
        fig.update_layout(title="Precision, recall and F1 by cut-off (training data, cross-validated)",
                          xaxis_title="Cut-off (sample probability)", yaxis_range=[0, 1.05])
        c1.plotly_chart(style(fig, 400), width="stretch")

        # A few cut-offs around the chosen one, as a table
        picks = sorted([0.2, 0.4, chosen["cutoff"], 0.8, 0.95])
        rows = []
        for pck in picks:
            r = curve.iloc[(curve["cutoff"] - pck).abs().argsort().iloc[0]]
            rows.append({
                "Cut-off": f"{r['cutoff']:.3f}",
                "Recall": r["recall"], "Precision": r["precision"], "F1": r["f1"],
                "": "← chosen" if r["chosen"] else "",
            })
        c2.dataframe(pd.DataFrame(rows).drop_duplicates("Cut-off"), hide_index=True,
                     width="stretch",
                     column_config={c: st.column_config.NumberColumn(format="%.2f")
                                    for c in ["Recall", "Precision", "F1"]})
        c2.caption(
            "Lowering the cut-off catches more bankruptcies, but precision falls faster than "
            "recall rises. Raising it gives cleaner flags but misses more failures. The chosen "
            "cut-off is the balance point; the F1 peak is usually broad, so nearby cut-offs "
            "perform almost as well."
        )

    # ── 4. Calibration ───────────────────────────────────────────────────────
    st.header("4. Can the probabilities be trusted?")
    c1, c2 = st.columns([1.2, 1])
    lim = float(max(calib["mean_predicted"].max(), calib["observed_rate"].max())) * 1.05
    fig = go.Figure()
    fig.add_scatter(x=[0, lim], y=[0, lim], mode="lines", name="Perfect calibration",
                    line=dict(color="#999999", dash="dash"))
    fig.add_scatter(x=calib["mean_predicted"], y=calib["observed_rate"], name="XGBoost",
                    mode="lines+markers", line=dict(color=ACTIVE_COLOR, width=3),
                    hovertemplate="predicted %{x:.1%}<br>observed %{y:.1%}<extra></extra>")
    fig.update_layout(title="Calibration (sample level)",
                      xaxis_title="Average predicted probability",
                      yaxis_title="Observed bankruptcy rate",
                      xaxis_tickformat=".0%", yaxis_tickformat=".0%")
    c1.plotly_chart(style(fig, 380), width="stretch")
    c2.markdown(
        """
Test companies are sorted into ten equal groups by predicted risk. If the points follow the
diagonal, a predicted 20% really means about 1 in 5 of such companies went bankrupt.

Because the model was trained without reweighting, its probabilities match the sample. They
are then corrected to the real-world rate before being shown as risk, as in the risk calculator.
        """
    )

    # ── 5. What drives the predictions ───────────────────────────────────────
    st.header("5. What drives the predictions")
    st.write(
        "SHAP splits each prediction into a contribution from every feature. "
        "All charts show averages over the test companies."
    )
    top = feature_name(importance["feature"].iloc[0])
    with st.expander("How to read the SHAP charts"):
        st.markdown(
            f"""
**What SHAP does.** Every prediction starts from the same baseline: the model's average score.
Each feature then pushes the score up (towards bankruptcy) or down (towards active), and the
pushes add up exactly to the company's final score. SHAP (SHapley Additive exPlanations) works
out how big each push is, using a method from game theory that shares credit fairly between
features that act together.

**The scale.** Contributions are in *log-odds*, the scale the model works on internally. As a
rule of thumb, +0.7 roughly doubles the odds of bankruptcy, −0.7 roughly halves them, and 0 means
no effect. Read the bars as relative sizes rather than exact percentages.

**Feature importance.** For each feature, the size of its push is averaged over all test
companies, ignoring direction. A long bar means the feature moves predictions a lot, either up
or down. Here **{top}** matters most. The chart does not say *which way* a feature pushes; the
two charts below do.

**Typical profile.** The average push of each feature, separately for bankrupt (red) and active
(blue) companies. Red bars to the right show what drives a typical bankrupt company's risk up;
blue bars to the left show what keeps a typical active company's risk down.

**Effect of one feature.** How a feature's push changes across its range. Companies are grouped
into bands of similar values (at least 20 per band), and each point is the band's average push.
Above the zero line raises risk, below lowers it. Where the line crosses zero is roughly where a
value stops being a warning sign.

**Things to keep in mind.**
- SHAP shows what the *model* learned from the data: associations, not causes.
- Related features share credit. Debt ratio, negative equity and total liabilities all describe
  leverage, so the importance of leverage as a whole is spread across them.
- Importance depends on the data: a feature can matter a lot for a few companies but little on
  average.
            """
        )

    imp = importance.head(12).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=imp["mean_abs_shap"], y=[feature_name(f) for f in imp["feature"]],
        orientation="h", marker_color=ACTIVE_COLOR,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(title="Feature importance (average absolute SHAP value)",
                      xaxis_title="Average impact on the prediction (log-odds)")
    st.plotly_chart(style(fig, 440), width="stretch")

    c1, c2 = st.columns(2)
    prof = profile.head(10).iloc[::-1]
    fig = go.Figure()
    for col, name, color in [("mean_shap_active", "Active", ACTIVE_COLOR),
                             ("mean_shap_bankrupt", "Bankrupt", BANKRUPT_COLOR)]:
        fig.add_bar(x=prof[col], y=[feature_name(f) for f in prof["feature"]],
                    orientation="h", name=name, marker_color=color)
    fig.add_vline(x=0, line_color="#999999")
    fig.update_layout(title="Typical profile: average contribution by class",
                      barmode="group", xaxis_title="← lowers risk      raises risk →")
    c1.plotly_chart(style(fig, 460), width="stretch")
    c1.caption("For a typical bankrupt company, the red bars show which features push its "
               "risk up; for a typical active company, the blue bars show what keeps it down.")

    feats = dependence["feature"].unique().tolist()
    chosen = c2.selectbox("Feature", feats, format_func=feature_name)
    d = dependence[dependence["feature"] == chosen]
    known, missing = d[d["band"] != "missing"], d[d["band"] == "missing"]
    fig = go.Figure()
    fig.add_scatter(x=known["feature_median"], y=known["mean_shap"], mode="lines+markers",
                    name="Value band", line=dict(color=ACTIVE_COLOR, width=3),
                    customdata=known[["band", "companies"]],
                    hovertemplate="%{customdata[0]}<br>effect %{y:+.2f}"
                                  "<br>%{customdata[1]} companies<extra></extra>")
    fig.add_hline(y=0, line_color="#999999")
    fig.update_layout(title=f"How {feature_name(chosen)} affects the prediction",
                      xaxis_title=f"{feature_name(chosen)} (median of each band)",
                      yaxis_title="Average effect (log-odds)", showlegend=False)
    c2.plotly_chart(style(fig, 400), width="stretch")
    if not missing.empty:
        c2.caption(f"Missing values ({int(missing['companies'].iloc[0])} companies): "
                   f"average effect {missing['mean_shap'].iloc[0]:+.2f}.")
    c2.caption("Each point averages at least 20 companies. Above zero raises risk, "
               "below zero lowers it.")


# ── Page: Risk calculator ────────────────────────────────────────────────────
@st.cache_resource
def load_demo_model():
    model_path, meta_path = RESULTS / "demo_model.json", RESULTS / "demo_model_meta.json"
    if not (model_path.exists() and meta_path.exists()):
        return None, None
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def to_population(p, factor):
    """Sample-level probability → population-level (real-world) probability."""
    odds = p / (1 - p) * factor
    return odds / (1 + odds)


def pct(p):
    """Probability as a readable percentage, keeping small values visible."""
    if p < 0.0001:
        return "< 0.01%"
    if p < 0.01:
        return f"{p:.2%}"
    return f"{p:.1%}"


def times_average(ratio):
    if ratio < 0.1:
        return "well below the average company"
    if ratio < 0.95:
        return f"{ratio:.1f}× the average company (lower)"
    if ratio <= 1.05:
        return "about the same as the average company"
    return f"{ratio:.1f}× the average company"


def euro(x):
    if x >= 1e6:
        return f"€{x / 1e6:.1f}M"
    return f"€{x / 1e3:.0f}k"


def calculator_page():
    st.title("Risk calculator")
    st.write(
        "Set a company profile and see the model's estimate of its bankruptcy risk, "
        "and which inputs drive it."
    )

    model, meta = load_demo_model()
    if model is None:
        st.error("Missing `results/demo_model.json` or `demo_model_meta.json`. "
                 "Run `02_Train_and_evaluate.ipynb` first.")
        st.stop()
    sl = meta["sliders"]

    # ── Inputs ───────────────────────────────────────────────────────────────
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.subheader("Company profile")
        industries = {i.split(" - ", 1)[1]: i for i in meta["industries"]}
        industry = st.selectbox("Industry", list(industries))

        def ratio_slider(key, fmt, help_text):
            s = sl[key]
            return st.slider(s["label"], min_value=round(s["min"], 2), max_value=round(s["max"], 2),
                             value=round(s["default"], 2), step=s["step"], format=fmt,
                             help=help_text)

        debt = ratio_slider("FE_debt_ratio", "%.2f",
                            "Total liabilities / total assets. Above 1 means liabilities "
                            "exceed assets (negative equity).")
        current = ratio_slider("FE_current_ratio", "%.2f",
                               "Current assets / current liabilities. Below 1 means short-term "
                               "debts exceed liquid assets.")
        roa = ratio_slider("FE_roa", "%.2f",
                           "Profit before tax / total assets. Below 0 means a loss-making year.")

        s = sl["Company Age"]
        age = st.slider(s["label"], int(s["min"]), int(s["max"]), int(round(s["default"])))

        # Size on a log scale: evenly spaced steps from €100k to the data's upper range
        s = sl["Total assets"]
        lo, hi = max(s["min"], 100_000), max(s["max"], 1_000_000)
        size_options = sorted({float(f"{v:.2g}") for v in np.geomspace(lo, hi, 40)})
        default_size = min(size_options, key=lambda v: abs(np.log(v) - np.log(s["default"])))
        assets = st.select_slider("Total assets (company size)", options=size_options,
                                  value=default_size, format_func=euro)

        neg_equity, loss = int(debt > 1), int(roa < 0)
        flags = []
        if neg_equity:
            flags.append("negative equity (debt ratio above 1)")
        if loss:
            flags.append("a loss-making year (ROA below 0)")
        if flags:
            st.caption("This profile has " + " and ".join(flags) + ".")

    # ── Score ────────────────────────────────────────────────────────────────
    row = pd.DataFrame([{c: 0.0 for c in meta["features"]}])
    row.loc[0, ["FE_debt_ratio", "FE_current_ratio", "FE_roa", "Company Age", "Total assets",
                "FE_negative_equity", "FE_negative_pl"]] = [
        debt, current, roa, age, assets, neg_equity, loss]
    row.loc[0, f"Industry: {industry}"] = 1.0
    row = row[meta["features"]]

    p_sample = float(model.predict_proba(row)[0, 1])
    p_pop    = float(to_population(p_sample, meta["odds_factor"]))
    base     = meta["population_rate"]

    deciles = np.array(meta["population_score_deciles"])
    percentile = float(np.interp(p_pop, deciles, np.linspace(0, 100, len(deciles))))
    if percentile < 50:
        band, color = "Low", "#2E8B57"
    elif percentile < 80:
        band, color = "Moderate", "#D4A017"
    elif percentile < 95:
        band, color = "Elevated", "#E07B39"
    else:
        band, color = "High", BANKRUPT_COLOR

    with right:
        st.subheader("Estimated risk")
        c1, c2 = st.columns(2)
        c1.metric("Bankruptcy probability", pct(p_pop))
        c1.caption(times_average(p_pop / base))
        c2.metric("Risk level", band)
        c2.caption(f"riskier than {percentile:.0f}% of companies")
        st.markdown(
            f"<div style='height:10px;border-radius:5px;background:linear-gradient(to right,"
            f"{color} {percentile:.0f}%, #EEEEEE {percentile:.0f}%);'></div>",
            unsafe_allow_html=True,
        )

        # ── What drives this estimate ────────────────────────────────────────
        # Per-input contributions on the log-odds scale. Industry columns are
        # combined into one, and each automatic flag is added to the slider it
        # comes from, so there is one bar per input the user controls.
        contribs = model.get_booster().predict(xgb.DMatrix(row), pred_contribs=True)[0]
        c = dict(zip(meta["features"] + ["bias"], contribs))
        parts = {
            "Debt ratio":       c["FE_debt_ratio"] + c["FE_negative_equity"],
            "Current ratio":    c["FE_current_ratio"],
            "Return on assets": c["FE_roa"] + c["FE_negative_pl"],
            "Company age":      c["Company Age"],
            "Company size":     c["Total assets"],
            "Industry":         sum(v for k, v in c.items() if k.startswith("Industry: ")),
        }
        parts = dict(sorted(parts.items(), key=lambda kv: abs(kv[1])))

        st.subheader("What drives this estimate")
        fig = go.Figure(go.Bar(
            x=list(parts.values()), y=list(parts), orientation="h",
            marker_color=[BANKRUPT_COLOR if v > 0 else ACTIVE_COLOR for v in parts.values()],
            hovertemplate="%{y}: %{x:+.2f}<extra></extra>",
        ))
        fig.update_layout(xaxis_title="← lowers risk      raises risk →", showlegend=False)
        fig.add_vline(x=0, line_color="#999999")
        st.plotly_chart(style(fig, 320), width="stretch")
        st.caption(
            "Bar length shows how much each input moves the estimate compared with an "
            "average company (log-odds scale)."
        )

    st.divider()
    st.caption(
        f"An illustrative estimate from a simplified model using only these inputs "
        f"(test ROC-AUC {meta['test_roc_auc']:.2f}), not a credit rating. Probabilities are "
        f"corrected to the real-world bankruptcy rate of {base:.2%}. Risk level compares the "
        f"profile with the companies in the test set."
    )


pg = st.navigation([
    st.Page(overview_page,   title="1. Overview",        icon="🏠", default=True),
    st.Page(data_page,       title="2. Data",            icon="📊"),
    st.Page(results_page,    title="3. Model results",   icon="📈"),
    st.Page(calculator_page, title="4. Risk calculator", icon="🎚️"),
])
pg.run()
