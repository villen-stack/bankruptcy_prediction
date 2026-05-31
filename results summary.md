### Model Evaluation Metrics

Binary classification models require different evaluation metrics
than regression models. Unlike R² or RMSE which measure deviation
from a continuous target, classification performance must capture
both the model's ability to distinguish classes and its behaviour
under class imbalance.

**Why not accuracy?**

Accuracy is the most intuitive metric but is fundamentally
misleading for imbalanced datasets. In our sample, 88% of
companies are active. A trivial model that classifies every
company as active — identifying zero bankruptcies — would
achieve 88% accuracy while being completely useless for the
intended purpose. Accuracy conflates true positive and true
negative performance, masking complete failure on the minority
class.

**ROC-AUC — Receiver Operating Characteristic, Area Under Curve**

ROC-AUC answers the question: *if we randomly select one bankrupt
company and one active company, what is the probability that the
model assigns a higher bankruptcy probability to the bankrupt one?*

The ROC curve plots the **True Positive Rate** (recall — fraction
of bankruptcies correctly identified) against the **False Positive
Rate** (fraction of active companies incorrectly flagged) at every
possible classification threshold from 0 to 1. The area under
this curve summarises discrimination ability across all thresholds:

ROC-AUC = 1.00   Perfect classifier
ROC-AUC = 0.86   Current model — strong discrimination
ROC-AUC = 0.72   Altman Z-score benchmark (private firms, based on academic literature)
ROC-AUC = 0.50   Random classifier — no discrimination

ROC-AUC is threshold-independent — it does not require choosing
a specific cutoff probability above which a company is classified
as bankrupt. This makes it suitable for comparing models
regardless of the operating point chosen for deployment.

A limitation of ROC-AUC is that it evaluates performance across
all thresholds equally, including very low thresholds that would
flag almost every company as bankrupt. In practice, only a narrow
range of thresholds is operationally meaningful. For severely
imbalanced datasets, ROC-AUC can appear optimistic because the
large number of true negatives (active companies) inflates the
denominator of the False Positive Rate, making the curve look
better than it is at the thresholds that matter.

**Average Precision — Area Under the Precision-Recall Curve**

Average Precision addresses the limitation of ROC-AUC directly.
It summarises the **Precision-Recall curve**, which plots:

- **Precision** — of all companies the model flags as bankrupt,
  what fraction are genuinely bankrupt?
- **Recall** — of all genuinely bankrupt companies, what fraction
  does the model correctly identify?

These two metrics are in tension — increasing recall (catching
more bankruptcies) typically reduces precision (flagging more
false positives). The precision-recall curve traces this tradeoff
across all thresholds and Average Precision computes the weighted
mean precision at each recall level.

Critically, the precision-recall curve does not involve true
negatives — it focuses entirely on the minority class performance.
This makes it far more informative than ROC-AUC when the positive
class (bankruptcy) is rare:

An Average Precision of 0.387 means that across all operating
thresholds the model achieves an average precision of 38.7% —
approximately 4.8 times better than randomly flagging companies.
In practical terms, a credit analyst using the model's ranked
output would find a genuine bankruptcy in roughly 2 out of every
5 companies flagged at moderate recall levels, compared to 1 in
12 by random selection.

**Interpreting the two metrics together**

| Metric | Our Model | Altman Z-score | Random |
|---|---|---|---|
| ROC-AUC | 0.8556 | ~0.72 | 0.50 |
| Average Precision | 0.3867 | ~0.15 | 0.081 |

ROC-AUC of 0.8556 confirms the model has strong overall
discriminative power. Average Precision of 0.3867 confirms this
performance holds specifically for the bankrupt minority class
and is not an artefact of the model's ability to correctly
classify the abundant active majority. The gap between the two
metrics is expected — Average Precision is always more demanding
than ROC-AUC under class imbalance — and the magnitude of both
relative to their respective baselines indicates genuine
predictive ability.

**The train-validation gap**

The difference between training ROC-AUC (0.9635) and validation
ROC-AUC (0.8556) indicates the model fits training data more
closely than unseen data — a standard manifestation of
overfitting in gradient boosting. This is partially controlled
through regularisation (`reg_alpha`, `reg_lambda`) and shallow
tree depth (`max_depth = 4`). Final overfitting assessment is
performed on the held-out test set, which the model has not
seen at any point during training or cross-validation.

### XGBoost Hyperparameters

The primary bankruptcy prediction model is an XGBoost gradient
boosting classifier (Chen & Guestrin, 2016) with the following
hyperparameter configuration:

| Parameter | Value | Rationale |
|---|---|---|
| `n_estimators` | 500 | Number of sequential boosting rounds — higher values reduce bias at the cost of training time |
| `learning_rate` | 0.05 | Step size shrinkage per round — low value requires more trees but generalises better |
| `max_depth` | 4 | Maximum tree depth — shallow trees reduce overfitting on the minority bankrupt class |
| `subsample` | 0.8 | Fraction of training rows sampled per tree — introduces randomness, reduces variance |
| `colsample_bytree` | 0.8 | Fraction of features sampled per tree — prevents over-reliance on any single predictor |
| `reg_alpha` | 0.1 | L1 regularisation — encourages sparse feature weights, penalises irrelevant features |
| `reg_lambda` | 1.0 | L2 regularisation — smooths feature weights, standard ridge penalty |
| `scale_pos_weight` | 55.4 | Universe-calibrated class weight — see Class Imbalance section |

### Cross-Validation Methodology

Model performance was estimated using **5-fold
StratifiedGroupKFold cross-validation** on the training set.
This approach ensures:

- **Stratified** — each fold maintains the ~8% bankruptcy rate
  of the training set, preventing folds with no bankrupt cases
- **Grouped by company** — all observation years for a given
  company appear in the same fold, preventing the same company
  from appearing in both training and validation within a fold
  — consistent with the company-level train/test split design

### Cross-Validation Results

| Metric | Train | Validation | Std |
|---|---|---|---|
| ROC-AUC | 0.9635 | 0.8556 | 0.0172 |
| Average Precision | 0.6464 | 0.3867 | 0.0394 |

**Per-fold stability:**

| Fold | ROC-AUC | Avg Precision |
|---|---|---|
| 1 | 0.8426 | 0.3361 |
| 2 | 0.8817 | 0.4570 |
| 3 | 0.8325 | 0.3715 |
| 4 | 0.8562 | 0.3810 |
| 5 | 0.8652 | 0.3878 |
| **Mean** | **0.8556** | **0.3867** |
| **Std** | **0.0172** | **0.0394** |

Fold-level variation is moderate and consistent — no single
fold produces anomalous results, indicating the model is
stable across different subsamples of the training data.