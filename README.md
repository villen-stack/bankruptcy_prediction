Data Analytics Demo by Ville Nurminen
======================

[Ville Nurminen's LinkedIn Profile](https://www.linkedin.com/in/villenurminen/)


This section provides an overview of the purpose and scope of the Data Analytics Demo. The repository showcases Python-based data analytics techniques, with a focus on financial statement analysis. Authored by Ville Nurminen, the project is intended solely for demonstration purposes, including all data, analyses, and conclusions. This project is covered under the [MIT License](LICENSE), allowing for reuse with proper attribution.



This file lists all the Python libraries and dependencies required to run the project. Install them using `pip install -r requirements.txt`. [requirements.txt](requirements.txt):

The project is structured into several Jupyter notebooks, each focusing on different aspects of the data analytics process, from data collection to model implementation. The key components include: data visualization, statistical modeling, and machine learning techniques such as XGBoost.


### Summary of key technologies used:
| Tool/Library | Description | Homepage |
|--------------|-------------|----------|
| <div align="center"><img src="htmlsupport/numpy.svg" alt="NumPy Logo" height="60"></div> | NumPy: A library for numerical computations in Python, supporting arrays and matrices. | [NumPy Homepage](https://numpy.org/) |
| <div align="center"><img src="htmlsupport/python.svg" alt="Python Logo" height="60"></div> | Python: The programming language used for this project. | [Python Homepage](https://www.python.org/) |
| <div align="center"><img src="htmlsupport/seaborn.svg" alt="Seaborn Logo" height="60"></div> | Seaborn: A Python visualization library based on Matplotlib. | [Seaborn Homepage](https://seaborn.pydata.org/) |
| <div align="center"><img src="htmlsupport/matplotlib.svg" alt="Matplotlib Logo" height="60"></div> | Matplotlib: A comprehensive library for creating static, animated, and interactive visualizations in Python. | [Matplotlib Homepage](https://matplotlib.org/) |
| <div align="center"><img src="htmlsupport/scikit.svg" alt="Scikit Logo" height="60"></div> | Scikit-learn: A machine learning library for Python. | [Scikit-learn Homepage](https://scikit-learn.org/) |
| <div align="center"><img src="htmlsupport/pandas.svg" alt="Pandas Logo" height="60"></div> | Pandas: A data manipulation and analysis library for Python. | [Pandas Homepage](https://pandas.pydata.org/) |
| <div align="center"><img src="htmlsupport/jupyter.svg" alt="Jupyter Logo" height="60"></div> | Jupyter Notebook: An interactive environment for running Python code. | [Jupyter Homepage](https://jupyter.org/) |
| <div align="center"><img src="htmlsupport/xgboost.png" alt="XGBoost Logo" height="60"></div> | XGBoost: An optimized gradient boosting library for machine learning. | [XGBoost Homepage](https://xgboost.readthedocs.io/) |


### Summary of key technologies used:

| Tool/Library | Description | Homepage |
|--------------|-------------|----------|
| ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) | Python: The programming language used for this project. | [Python Homepage](https://www.python.org/) |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) | NumPy: Used for numerical computations and array operations throughout the analysis pipeline. | [NumPy Homepage](https://numpy.org/) |
| ![pandas](https://img.shields.io/badge/pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) | pandas: Used for panel data construction, feature engineering, and preprocessing of 10,000+ Finnish company financial statements. | [pandas Homepage](https://pandas.pydata.org/) |
| ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white) | scikit-learn: Used for model evaluation, cross-validation, and baseline logistic regression classifier. | [scikit-learn Homepage](https://scikit-learn.org/) |
| ![XGBoost](https://img.shields.io/badge/XGBoost-189AB4?style=for-the-badge&logoColor=white) | XGBoost: Primary gradient boosting classifier for bankruptcy prediction on imbalanced Finnish private company data. | [XGBoost Homepage](https://xgboost.readthedocs.io/) |
| ![SHAP](https://img.shields.io/badge/SHAP-FF6B6B?style=for-the-badge&logoColor=white) | SHAP: Used for model interpretability — identifying which financial ratios drive bankruptcy predictions. | [SHAP Homepage](https://shap.readthedocs.io/) |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white) | Matplotlib: Used for visualizing model performance, feature importance, and financial data distributions. | [Matplotlib Homepage](https://matplotlib.org/) |
| ![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge&logoColor=white) | Seaborn: Used for statistical visualizations including correlation matrices and class distribution plots. | [Seaborn Homepage](https://seaborn.pydata.org/) |
| ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white) | Jupyter Notebook: Interactive development environment used for exploratory analysis and model development. | [Jupyter Homepage](https://jupyter.org/) |

### Data Source

| Source | Description | Homepage |
|--------|-------------|----------|
| ![Orbis](https://img.shields.io/badge/Orbis-Bureau%20van%20Dijk-003366?style=for-the-badge&logoColor=white) | Orbis (Bureau van Dijk / Moody's Analytics): Source of Finnish private company financial statements 2004–2024, accessed via TAMK institutional license. | [Orbis Homepage](https://www.bvdinfo.com/en-gb/our-products/data/international/orbis) |