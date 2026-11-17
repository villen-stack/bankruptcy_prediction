# Finnish Corporate Bankruptcy Prediction

**Predicting insolvency risk in Finnish private companies from their financial statements, using machine learning.**

**Author:** [Ville Nurminen](https://www.linkedin.com/in/villenurminen/)
🔗 **[Live demo](https://YOUR-APP.streamlit.app)**

**About this project:** This is a portfolio and technology demonstration.

> The company data comes from Moody's, accessed under an academic licence
> through Tampere University of Applied Sciences (TAMK). Licence terms do not
> allow the data to be redistributed, so this repository contains the code,
> methodology and aggregated results only, not the underlying company data.

## The problem

When a private company runs into trouble, there is no share price to signal it. Lenders, suppliers, investors and auditors have to judge financial health from annual financial statements alone — often with limited time and only a few years of data.

Classic bankruptcy models such as Altman's Z-score were built on large, listed US manufacturing firms decades ago, and they are a poor fit for Finnish SMEs. The Z-score needs variables such as the market value of equity, retained earnings, EBIT and sales. A private company has no market value at all, and the other items are often missing from the accounts that smaller companies publish.

In practice, credit risk for private companies is typically scored with **logistic regression** on financial ratios — the industry-standard approach, valued for being simple and transparent.

This project asks a practical question: **how well can a company's latest financial statements distinguish firms that went bankrupt from those that stayed solvent?** It focuses on Finnish private companies in the 2022–2024 period and tests whether a gradient-boosted model (XGBoost) can improve on the logistic regression benchmark, using SHAP to show which financial signals drive its predictions.

---

## Data

Financial statement data was obtained from **Orbis (Bureau van Dijk / Moody's)**. The dataset combines two separate exports:

| Group                      |        Companies | Share |
| -------------------------- | ---------------: | ----: |
| Active (solvent) companies |            9,100 |   89% |
| Bankrupt companies         |            1,136 |   11% |
| **Total**            | **10,236** |  100% |

For each company the data contains five key financial statement items for **2022, 2023 and 2024** (profit/loss before tax, total assets, current assets, current liabilities and shareholders' funds), plus its industry code (NACE Rev. 2) and year of incorporation.

**Why two exports?** Bankruptcy is rare: in any given year only a small fraction of Finnish companies go bankrupt. A random sample of companies would contain too few bankruptcies for a model to learn from. Bankrupt companies were therefore extracted separately to ensure a large enough number of failure cases. As a result, the 11% bankruptcy share in this dataset is much higher than in the real company population.

**Class imbalance.** Even after this enrichment, solvent companies outnumber bankrupt ones roughly 8 to 1. A model could reach 89% accuracy by simply predicting "active" for everyone, so the imbalance is handled explicitly:

- **Class weighting:** errors on bankrupt companies are weighted more heavily during training (`scale_pos_weight` in XGBoost, `class_weight="balanced"` in logistic regression).
- **Stratified train/test split**, so both sets keep the same share of bankruptcies.
- **Imbalance-aware metrics:** models are compared on PR AUC, precision, recall and F1 rather than accuracy.

**Missing values.** Bankrupt companies often stop publishing financial statements, so their most recent years are frequently empty. These gaps are kept rather than filled in blindly, because the absence of a filing is itself a warning sign.

**Bankruptcy timing.** The Orbis export records each company's bankruptcy status at the time of extraction but does not include the date of the bankruptcy. Because the dataset does not specify when each bankruptcy occurred, the model is designed to classify companies by their financial profile rather than to forecast bankruptcy a fixed number of years ahead.

> ⚠️ Raw data is not included in this repository due to Orbis licensing restrictions.

---

## Approach

### From raw data to features

The two exports are merged, labelled (0 = active, 1 = bankrupt) and reshaped into a **company-year panel**: one row per company per financial year, giving 27,965 observations. Each row combines the five reported figures with company age, industry section and nine engineered features:

| Feature              | What it measures                             | Why it matters                                                            |
| -------------------- | -------------------------------------------- | ------------------------------------------------------------------------- |
| Debt ratio           | Total liabilities / total assets             | Leverage. Values above 1 mean liabilities exceed assets                   |
| Current ratio        | Current assets / current liabilities         | Short-term liquidity. Below 1 means short-term debts exceed liquid assets |
| Return on assets     | Profit before tax / total assets             | Profitability relative to the asset base                                  |
| Negative equity flag | Shareholders' funds < 0                      | Accumulated losses have wiped out the equity                              |
| Loss-year flag       | Profit before tax < 0                        | The company made a loss that year                                         |
| Asset growth         | Year-over-year change in total assets        | A shrinking balance sheet can signal distress                             |
| Equity change        | Year-over-year change in shareholders' funds | Eroding equity is a strong warning sign                                   |
| Total liabilities    | Total assets − shareholders' funds          | Derived, as Orbis does not report it directly                             |
| Non-current assets   | Total assets − current assets               | Derived balance sheet structure                                           |

Ratios are set to missing rather than extreme values when the denominator is effectively zero, so a near-empty balance sheet does not produce misleading outliers.

### Models

**Logistic regression (benchmark).** The standard credit-scoring approach. It estimates how each feature raises or lowers the odds of bankruptcy, assuming each effect is a straight line.

**XGBoost (challenger).** XGBoost builds hundreds of small decision trees one after another, with each new tree focusing on the cases the previous trees got wrong. The final prediction combines all the trees. Three properties make it a good fit for this problem:

- **Non-linear effects:** it can learn that negative equity matters far more than slightly low equity, or that high leverage is only dangerous when combined with losses.
- **Missing values:** it handles gaps natively, which matters because missing filings are common among distressed companies.
- **Imbalanced classes:** it supports weighting the rare bankrupt class more heavily during training.

### Evaluation

Both models are trained and tested on the same split. The split is made **by company**, so the same company never appears in both the training and test sets; otherwise a model could recognise a company from its other years instead of learning general patterns. Results are compared on imbalance-aware metrics (PR AUC, precision, recall, F1). **SHAP values** are used to explain which features drive XGBoost's predictions, both overall and for individual companies.

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
| [`02_Train_and_evaluate.ipynb`](02_Train_and_evaluate.ipynb) | Logistic regression vs XGBoost, evaluation, SHAP |
| [`03_Demo_app.py`](03_Demo_app.py)                           | Streamlit dashboard                              |
| [`results/`](results/)                                       | Saved outputs used by the app                    |

---

## Tech Stack

| Tool/Library                                                                                                             | Role in this project                                                                       | Homepage                                  |
| ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ | ----------------------------------------- |
| ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)                    | Core programming language                                                                  | [Python](https://www.python.org/)          |
| ![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)                    | Panel data construction and feature engineering across 10,000+ company-year observations   | [pandas](https://pandas.pydata.org/)       |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)                       | Numerical computations and array operations                                                | [NumPy](https://numpy.org/)                |
| ![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge&logoColor=white)                              | Primary bankruptcy prediction classifier — gradient boosting on imbalanced financial data | [XGBoost](https://xgboost.readthedocs.io/) |
| ![SHAP](https://img.shields.io/badge/SHAP-FF6B6B?style=for-the-badge&logoColor=white)                                    | Model interpretability — which financial ratios drive insolvency predictions              | [SHAP](https://shap.readthedocs.io/)       |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) | Logistic regression baseline, cross-validation, evaluation metrics                         | [scikit-learn](https://scikit-learn.org/)  |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)        | Model performance visualization, ROC curves, feature importance plots                      | [Matplotlib](https://matplotlib.org/)      |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logoColor=white)                              | Statistical visualizations — correlation matrices, class distributions                    | [Seaborn](https://seaborn.pydata.org/)     |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)                 | Interactive development environment                                                        | [Jupyter](https://jupyter.org/)            |

---

## License

This project is licensed under the [MIT License](LICENSE) — free to use and adapt with attribution.
