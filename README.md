# Finnish Corporate Bankruptcy Prediction
### Machine Learning on Private Company Financial Data
**Author:** [Ville Nurminen](https://www.linkedin.com/in/villenurminen/) · CPA · CFA Level II Candidate

---

## Overview

This project builds a **binary bankruptcy prediction model** for Finnish private companies using gradient boosting (XGBoost) on real financial statement data sourced from the Orbis database. The model predicts the probability of corporate insolvency based on balance sheet and P&L indicators, benchmarked against the classical Altman Z-score.

**Key highlights:**
- 📊 **10,000+ Finnish private companies** across 14 NACE industry sectors
- 📅 **20 years of financial history** (2004–2024) from Moody's
- 🏦 **Real-world class imbalance** — ~11% bankruptcy rate, handled via XGBoost class weighting
- 🔍 **SHAP interpretability** — identifying which financial ratios drive insolvency risk
- 📈 **Benchmarked against Altman Z-score** (1968) to demonstrate ML improvement over classical methods
- 🇫🇮 **Finnish private company focus** — addresses a gap in the literature dominated by US public company data

---

## Project Structure

The analysis is organized into sequential Jupyter notebooks:

| Notebook | Description |
|----------|-------------|
| `01_data_preparation.ipynb` | Data loading, merging, panel construction |
| `02_feature_engineering.ipynb` | Financial ratio computation, Altman variables |
| `03_eda.ipynb` | Exploratory analysis, class distribution, missing data |
| `04_modeling.ipynb` | XGBoost, logistic regression, cross-validation |
| `05_evaluation.ipynb` | AUC-ROC, precision-recall, confusion matrix |
| `06_interpretability.ipynb` | SHAP feature importance, waterfall plots |

Install dependencies: `pip install -r requirements.txt`

---

## Tech Stack

| Tool/Library | Role in this project | Homepage |
|--------------|----------------------|----------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | Core programming language | [Python](https://www.python.org/) |
| ![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) | Panel data construction and feature engineering across 10,000+ company-year observations | [pandas](https://pandas.pydata.org/) |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) | Numerical computations and array operations | [NumPy](https://numpy.org/) |
| ![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge&logoColor=white) | Primary bankruptcy prediction classifier — gradient boosting on imbalanced financial data | [XGBoost](https://xgboost.readthedocs.io/) |
| ![SHAP](https://img.shields.io/badge/SHAP-FF6B6B?style=for-the-badge&logoColor=white) | Model interpretability — which financial ratios drive insolvency predictions | [SHAP](https://shap.readthedocs.io/) |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) | Logistic regression baseline, cross-validation, evaluation metrics | [scikit-learn](https://scikit-learn.org/) |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white) | Model performance visualization, ROC curves, feature importance plots | [Matplotlib](https://matplotlib.org/) |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logoColor=white) | Statistical visualizations — correlation matrices, class distributions | [Seaborn](https://seaborn.pydata.org/) |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white) | Interactive development environment | [Jupyter](https://jupyter.org/) |

---

## Data Source

| Source | Description | Homepage |
|--------|-------------|----------|
| ![Orbis](https://img.shields.io/badge/Orbis-Bureau%20van%20Dijk-003366?style=for-the-badge&logoColor=white) | Finnish private company financial statements 2004–2024 (10,000+ companies). Active and bankrupt firms across 14 NACE industry sectors. Accessed via TAMK institutional license. | [Orbis / Bureau van Dijk](https://www.bvdinfo.com/en-gb/our-products/data/international/orbis) |

> ⚠️ Raw data is not included in this repository due to Orbis licensing restrictions. The dataset can be reproduced by applying the documented filters to an institutional Orbis subscription.

---

## License

This project is licensed under the [MIT License](LICENSE) — free to use and adapt with attribution.