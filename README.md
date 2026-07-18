# Predicting Content Virality on YouTube/TikTok

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-yellow)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-green)

## Project Overview
This project addresses a **Supervised Binary Classification** task aimed at predicting whether content published on social media platforms like TikTok or YouTube will become **viral** (labeled as "rising").

Rather than focusing solely on maximizing performance metrics, this work emphasizes **methodological rigor** by applying data science best practices: from *data leakage* prevention to feature engineering, and model interpretability using Explainable AI (XAI).

## Objectives and Methodology
The goal is to demonstrate a robust machine learning process through the following steps:

1. **Data Leakage Prevention:** Early split into Train/Test sets and critical removal of "future-looking" variables (e.g., `trend_duration_days`, `engagement_velocity`) that would otherwise invalidate the model.
2. **Exploratory Data Analysis (EDA) & Data Quality:** Systematic audit of data quality, verification of algebraic redundancies (linearly correlated metrics), and target distribution analysis conducted strictly on the Training Set.
3. **Feature Engineering & Selection:** Targeted selection of 12 heterogeneous features (metadata and early-stage metrics) to avoid perfect collinearity, discarding high-cardinality variables to prevent overfitting.
4. **Robust Modeling:** Training of an ensemble model (`RandomForestClassifier`), preceded by a univariate predictive power audit to establish the actual existence of signal within the dataset.
5. **Critical Evaluation:** Out-of-sample evaluation focused on appropriate classification metrics, correctly interpreting output based on the real signal available compared to a dummy classifier.
6. **Explainable AI (XAI):** Integration of **SHAP** values to provide transparent and interpretable insights into the impact and marginal contribution of each feature on model decisions.

## The Dataset
The dataset contains social media trend information for the year 2025, consisting of **48,079 observations and 58 original columns**.
*   **Target:** Binary variable (`target = 1` if `trend_label == "rising"`, `0` otherwise).
*   **Key Selected Features:** `platform`, `category`, `traffic_source`, `duration_sec`, `log_views`, `views_vs_creator_baseline`, `quota_comment`, `log_creator_avg_views`,`engagement_rate`, `title_length`, `quota_share`, `completion_rate`.

Note: The dataset is treated as imbalanced, with a positive class prevalence of approximately 25%.

## Technologies Used
*   **Data Manipulation:** `pandas`, `numpy`
*   **Data Visualization:** `matplotlib`, `seaborn`
*   **Machine Learning & Preprocessing:** `scikit-learn` (Pipeline, ColumnTransformer, RandomForest, DummyClassifier, GridSearchCV/RandomizedSearchCV)
*   **Statistical Testing:** `scipy.stats` (Chi-Square, Mann-Whitney U)
*   **Explainability:** `shap`


## Conclusion

The central finding of this project is stated plainly, because it shapes how every result
above should be read: **the dataset contains no usable predictive signal with respect to the
target.** The final Random Forest achieves a test-set AUROC of approximately 0.50, i.e. the value
expected from random guessing.

This is a *result*, not a failure of the pipeline. It is supported by four independent and
converging lines of evidence:

1. **Univariate audit**: no numeric feature departs meaningfully from an AUROC of 0.50, and
   the number of categorical variables flagged as significant is consistent with what
   multiple testing produces by chance alone.
2. **Independence tests**: chi-square tests provide no basis for rejecting independence
   between the predictors and the target.
3. **Validation curves**: as model capacity increases, the training score rises while the
   validation score stays flat at the chance level: the canonical signature of a model
   memorising noise rather than learning structure.
4. **Out-of-sample test**: the ROC curve lies on the diagonal, and precision remains flat at
   the base rate across every threshold.


### The likely reason: label integrity

The temporal-consistency audit reframes the absence of signal as part of a broader pattern.
At least two derived columns carry labels assigned independently of the reality they claim to
describe: `season` is random with respect to the publication date, and
`event_season` implements a correct *geographic* constraint while omitting the *temporal* one. That `trend_label`, and hence the target, is also
independent of every predictor fits the same pattern rather than standing out as an anomaly.
The GIGO principle applies without exception: no modelling sophistication recovers information that is not present in the data.

### Methodological value

Although performance is not the deliverable, the pipeline demonstrates the techniques the
project is meant to assess: an early, stratified train/test split with all learning confined
to the training set; algebraic verification and removal of redundant, perfectly collinear
engagement metrics; principled exclusion of high-cardinality columns backed by a quantitative
demonstration of the overfitting they induce; leakage-safe preprocessing inside a
`ColumnTransformer`/`Pipeline` fitted independently within each cross-validation fold; a
single, final look at the test set; and full reproducibility via a fixed random seed.

### What a real-world next step would be

The operational conclusion is that the available features cannot predict virality, and that
effort should be redirected toward collecting *different* data such as early temporal trajectories, or intrinsic content characteristics and network effects, rather than toward further tuning of a model that has no raw material to work with.

## Future Improvements
- **Alternative Modeling:** Testing Gradient Boosting models (e.g., XGBoost, LightGBM) to compare with the current Random Forest performance.
- **Advanced Feature Engineering:** Implementing NLP techniques (e.g., TF-IDF or BERT embeddings) on the `title` and `tags` columns to capture deeper semantic signals.
- **External Data:** Incorporating external APIs to fetch real-time creator metrics to enrich the dataset.


## Author
**Elia Dallanoce**
* [Portfolio](https://eliadallanoce-portfolio.vercel.app)
* [GitHub](https://github.com/eliaxyzz)

*Academic Project - Applied Machine Learning 2026*