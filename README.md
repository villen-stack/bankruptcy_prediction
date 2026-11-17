# Finnish Corporate Bankruptcy Prediction

**Predicting insolvency risk in Finnish private companies from their financial statements, using machine learning.**

**Author:** [Ville Nurminen](https://www.linkedin.com/in/villenurminen/)
🔗 **[Live demo](https://YOUR-APP.streamlit.app)**

> **About this project:** This is a portfolio and technology demonstration.
> The company data comes from Moody's, accessed under an academic licence
> through Tampere University of Applied Sciences (TAMK). Licence terms do not
> allow the data to be redistributed, so this repository contains the code,
> methodology and aggregated results only, not the underlying company data.

## Contents

1. [The problem](#the-problem)
2. [Data](#data)
3. [Why XGBoost?](#why-xgboost)
4. [Approach](#approach)
5. [Results](#results)
6. [What drives the predictions](#what-drives-the-predictions)
7. [Limitations](#limitations)
8. [Project structure](#project-structure)
9. [Tech Stack](#tech-stack)
10. [Data and licensing](#data-and-licensing)
11. [License](#license)

## The problem

When a private company runs into trouble, there is no share price to signal it. Lenders, suppliers, investors and auditors have to judge financial health from annual financial statements alone — often with limited time and only a few years of data.

Classic bankruptcy models such as Altman's Z-score were built on large, listed US manufacturing firms decades ago, and they are a poor fit for Finnish SMEs. The Z-score needs variables such as the market value of equity, retained earnings, EBIT and sales. A private company has no market value at all, and the other items are often missing from the accounts that smaller companies publish.

This project asks a practical question: **how well can a company's latest financial statements distinguish firms that went bankrupt from those that stayed solvent?** It focuses on Finnish private companies in 2022–2024, uses a gradient-boosted model (XGBoost) that captures non-linear relationships and interactions between financial ratios, and uses SHAP to show which financial signals drive each prediction.

---

## Data

Financial statement data was obtained from Moody's (Orbis database). The target population is Finnish limited companies matching these search criteria:

- Status: active, bankrupt, or dissolved through bankruptcy
- Revenue ≥ €200,000 and total assets ≥ €100,000
- Excluded industries (NACE sections): K (finance), O (public administration), P (education), Q (health), D (energy), T (households), U (extraterritorial)

This gives a population of **64,052 companies, of which 1,136 (1.77%) are bankrupt**.

### Sample design

Bankruptcy is rare, and the database caps the size of a single export. The sample was therefore built from two exports:

| Group | In population | Exported | In model data |
|---|---:|---:|---:|
| Active companies | 62,916 | 9,100 (random sample) | _(update)_ |
| Bankrupt companies | 1,136 | 1,136 (all) | _(update)_ |
| **Bankruptcy share** | **1.77%** | 11.1% | _(update)_ |

<!-- "In model data": take from results/dataset_overview.csv after re-running 01_Build_dataset (duplicate exports removed; companies sharing a name are kept separate). -->

Every bankrupt company is included, while active companies are a random sample of about 1 in 7. Keeping all bankruptcies gives the model as many failure cases to learn from as possible. A purely random sample of the same size would contain only around 180 bankrupt companies.

The trade-off is that bankruptcies are about 6× more common in the sample than in reality, and a model learns the bankruptcy rate it is trained on. Its raw probabilities therefore overstate real-world risk: a company scored at 30% in sample terms has a real-world probability of about 6%. Because the sampling rates are known exactly, the probabilities are corrected back to the 1.77% population rate after training (see *Evaluation*). The correction does not change which companies are ranked as riskier, only how the probabilities read.

For each company the data contains five financial statement items for 2022–2024 (profit/loss before tax, total assets, current assets, current liabilities and shareholders' funds), plus its industry (NACE Rev. 2) and year of incorporation. The model uses **one row per company: its latest year of accounts**, with changes from the previous year as additional features (see *Approach*).

### Class imbalance

Even in the sample, active companies outnumber bankrupt ones about 8 to 1. This is handled in two ways:

- **Stratified train/test split.** Each company is a single row, so no company can appear in both sets, and both sets keep the same share of bankruptcies.
- **No class reweighting.** An 8:1 ratio is mild, and upweighting bankruptcies would exaggerate a class that is already over-represented. The sampling bias is corrected on the probabilities instead.

### Missing values and data availability

Bankrupt companies stop publishing financial statements after they fail, so they have fewer years of data, and their latest accounts are often from 2022 or 2023 while active companies' are from 2024. Remaining gaps, such as year-over-year changes for a company with only one year of data, are left as missing values, which XGBoost handles natively.

This creates patterns the model could exploit without learning anything about finances: the year of the latest accounts, or how many years a company reported, would reveal *when the records end* rather than how healthy the company was. Neither is used as a feature, so the model has to rely on the financial figures themselves.

### Bankruptcy timing

The export records each company's bankruptcy status at the time of extraction, not the date of the bankruptcy. The model therefore classifies companies by the financial profile in their **latest accounts** (for a bankrupt company, the last accounts before failure) rather than forecasting bankruptcy a fixed number of years ahead. With bankruptcy dates, each year could instead be labelled "fails within 12 months", which would allow a true forecasting model.

The data itself is not included in this repository. See [*Data and licensing*](#data-and-licensing).

---

## Why XGBoost?

### What it is

XGBoost (eXtreme Gradient Boosting) builds an ensemble of small decision trees, one after another. Each new tree focuses on the cases the previous trees got wrong, and the final prediction combines all of them. A single tree might learn a rule such as:

    debt ratio > 0.9  AND  loss-making year  →  higher risk

Hundreds of such trees, each correcting the last, add up to a model that can capture complex patterns while each individual step stays simple.

### Why it fits this problem

| Property of the problem | How XGBoost handles it |
|---|---|
| **Tabular data** — financial statement items and ratios | Gradient-boosted trees are consistently among the strongest methods for structured, tabular data |
| **Non-linear effects** — risk jumps sharply once equity turns negative or debt exceeds assets | Trees split at thresholds, so they learn such cut-offs directly instead of assuming a straight-line relationship |
| **Interactions** — high debt matters more for a loss-making company | Each tree combines several features in sequence, capturing combinations automatically |
| **Missing values** — e.g. no year-over-year change in a company's first year | Handled natively: each split learns the best direction for missing values, without imputation |
| **Skewed figures and outliers** — company sizes range from €100k to billions | Splits depend only on the order of values, so no scaling or log-transforms are needed and extreme values do not dominate |
| **Moderate data size** — about 10,000 companies | Regularisation, shallow trees and row/feature subsampling keep the model from memorising individual companies |
| **Need for explanations** — a risk score is only useful if you can say *why* | SHAP's TreeExplainer computes exact feature contributions for tree models, so every prediction can be broken down feature by feature |

### Trade-offs

- **Less transparent than a simple scorecard.** A single equation is easier to read than hundreds of trees. SHAP closes most of this gap by explaining both the model as a whole and each individual prediction.
- **Can overfit.** Boosting fits the training data closely, so performance is judged only on companies the model has never seen (grouped cross-validation and a held-out test set), and the train/validation gap is reported openly.
- **Raw probabilities reflect the training sample.** Because bankruptcies are over-represented in the sample, predicted probabilities are corrected to the real population rate before they are interpreted as risk levels.

---

## Approach

### From raw data to features

The two exports are merged, labelled (0 = active, 1 = bankrupt), given a unique company ID (several companies share a name) and reshaped into a **company-year panel**: one row per company per financial year. Features are computed on the panel, then each company is reduced to **its latest year**, so every company appears exactly once. History is carried in the change features, which compare the latest year with the year before:

| Feature              | What it measures                             | Why it matters                                                            |
| -------------------- | -------------------------------------------- | ------------------------------------------------------------------------- |
| Debt ratio           | Total liabilities / total assets             | Leverage. Values above 1 mean liabilities exceed assets                   |
| Current ratio        | Current assets / current liabilities         | Short-term liquidity. Below 1 means short-term debts exceed liquid assets |
| Return on assets     | Profit before tax / total assets             | Profitability relative to the asset base                                  |
| Negative equity flag | Shareholders' funds < 0                      | Accumulated losses have wiped out the equity                              |
| Loss-year flag       | Profit before tax < 0                        | The company made a loss that year                                         |
| Asset growth         | Year-over-year change in total assets, relative to the prior year | A shrinking balance sheet can signal distress        |
| Equity change        | Year-over-year change in shareholders' funds, relative to prior-year total assets | Eroding equity is a strong warning sign |
| Debt ratio change    | Debt ratio this year minus last year         | Rising leverage shows a weakening balance sheet                           |
| ROA change           | Return on assets this year minus last year   | Falling profitability is an early sign of trouble                         |
| Total liabilities    | Total assets − shareholders' funds          | Derived, as the data does not report it directly                          |
| Non-current assets   | Total assets − current assets               | Derived balance sheet structure                                           |

Each company row also includes the five reported figures, company age and industry section. Ratios are set to missing rather than extreme values when the denominator is effectively zero (below €1,000), so a near-empty balance sheet does not produce misleading outliers. Change features compare only consecutive calendar years of the same company, and are missing for companies with a single year of data.

### Evaluation

Each company is one row, so the same company never appears in both the training and test sets. Performance is estimated with stratified 5-fold cross-validation on the training set and confirmed once on the held-out test set.

- **Metrics:** ROC-AUC and Average Precision (PR-AUC), plus precision, recall and F1.
- **Decision threshold:** chosen to maximise F1 on cross-validated training predictions, then applied unchanged to the test set, so the test set is used only for scoring.
- **Population-level results:** predicted probabilities are corrected from the sample's 11% bankruptcy rate to the population's 1.77%, and precision and Average Precision are re-estimated at the population rate.
- **Explanations:** **SHAP values** show which features drive the predictions, both overall and for individual companies.

---

## Results

*(to do)*

---

## What drives the predictions

*(to do)*

---

## Limitations

*(to do)*

---

## Project structure

| File                                                          | What it does                                     |
| ------------------------------------------------------------- | ------------------------------------------------ |
| [`01_Build_dataset.ipynb`](01_Build_dataset.ipynb)           | Loading, cleaning, feature engineering           |
| [`02_Train_and_evaluate.ipynb`](02_Train_and_evaluate.ipynb) | XGBoost training, cross-validation, evaluation, probability correction, SHAP |
| [`03_Demo_app.py`](03_Demo_app.py)                           | Streamlit dashboard                              |
| [`results/`](results/)                                       | Saved outputs used by the app                    |

---

## Tech Stack

| Tool/Library                                                                                                             | Role in this project                                                                       | Homepage                                  |
| ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ | ----------------------------------------- |
| ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)                    | Core programming language                                                                  | [Python](https://www.python.org/)          |
| ![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)                    | Panel construction (~28,000 company-years), feature engineering, per-company model dataset | [pandas](https://pandas.pydata.org/)       |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)                       | Numerical computations and array operations                                                | [NumPy](https://numpy.org/)                |
| ![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge&logoColor=white)                              | Bankruptcy prediction model — gradient boosting on financial statement data                | [XGBoost](https://xgboost.readthedocs.io/) |
| ![SHAP](https://img.shields.io/badge/SHAP-FF6B6B?style=for-the-badge&logoColor=white)                                    | Model interpretability — which financial ratios drive insolvency predictions              | [SHAP](https://shap.readthedocs.io/)       |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) | Company-grouped cross-validation, train/test split, evaluation metrics, calibration check   | [scikit-learn](https://scikit-learn.org/)  |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)        | Model performance visualization, ROC curves, feature importance plots                      | [Matplotlib](https://matplotlib.org/)      |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logoColor=white)                              | Statistical visualizations — correlation matrices, class distributions                    | [Seaborn](https://seaborn.pydata.org/)     |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)                 | Interactive development environment                                                        | [Jupyter](https://jupyter.org/)            |

---

## Data and licensing

The financial statement data was obtained from Moody's under an academic licence held by Tampere University of Applied Sciences (TAMK), and is used here for non-commercial, educational purposes only.

- **Not included in this repository:** the raw exports, the processed datasets (`final_panel.csv`, `model_data.csv`) and any company-level records. These are excluded via `.gitignore`.
- **Included:** all code, the methodology, and aggregated results (metrics, charts and summary statistics that do not identify individual companies).
- **Reproducing the results** requires your own access to Moody's company data. The search criteria used are listed under [*Data*](#data).

---

## License

The code in this project is licensed under the [MIT License](LICENSE) — free to use and adapt with attribution. The licence does not cover the underlying company data, which is not included (see [*Data and licensing*](#data-and-licensing)).
