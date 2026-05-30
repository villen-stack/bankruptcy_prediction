## Modeling Design

### Class Imbalance

The bankruptcy prediction task involves inherent class imbalance
reflecting the rarity of corporate insolvency in the Finnish SME
population. Our Moody's universe search — applying identical NACE
section exclusions and size filters as the study sample — returned
64,052 Finnish companies of which 1,136 (1.77%) held bankruptcy
or dissolved-by-bankruptcy status at the time of extraction.

Our study sample deliberately overrepresents the bankrupt class
to ensure sufficient minority class observations for model
training — the sample bankruptcy rate of 11.1% is approximately
6.3× the universe rate. To counteract this oversampling and
calibrate the model toward the true Finnish SME bankruptcy
prevalence, we set XGBoost's `scale_pos_weight` parameter to
the universe-derived class ratio:

$$\text{scale\_pos\_weight} = \frac{N_{\text{active}}^{\text{universe}}}{N_{\text{bankrupt}}^{\text{universe}}} = \frac{62{,}916}{1{,}136} \approx 55.4$$

This approach is preferred over sample-derived weighting because
the universe counts represent real observed prevalence within the
same filtered company population — not an external population
estimate — making the calibration directly applicable to the
study segment.

| | Universe (Moody's) | Sample |
|---|---|---|
| Total companies | 64,052 | 10,236 |
| Active | 62,916 | 9,100 |
| Bankrupt | 1,136 | 1,136 |
| Bankruptcy rate | 1.77% | 11.1% |
| scale_pos_weight | **55.4** | 8.0 |

### Train / Test Split

To prevent data leakage in the panel structure — where the same
company appears across multiple observation years — the
train/test split is performed at the **company level** rather
than the row level. Each company's observations appear entirely
within either the training set or the test set, never both.

An 80/20 split stratified on the company-level bankruptcy label
was applied, yielding:

| | Train | Test |
|---|---|---|
| Companies | ~8,200 | ~2,050 |
| Rows | ~22,400 | ~5,600 |
| Bankruptcy rate | ~11.1% | ~11.1% |

Stratification ensures the bankruptcy prevalence in the sample
is preserved across both splits, preventing train/test imbalance
from affecting model evaluation.