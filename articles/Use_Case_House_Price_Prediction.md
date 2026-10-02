## **HOUSE PRICE PREDICTION USING MACHINE LEARNING**


*Complete Use-Case Report and Explanation*

---
- Dataset: California Housing Prices
- Target Variable: median_house_value
- Final Model: HistGradientBoostingRegressor
- Implementation: Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- Author: Ravikiran K C
---


## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Overview](#2-project-overview)
3. [Real-World Use Case](#3-real-world-use-case)
4. [Problem Statement](#4-problem-statement)
5. [Project Objectives](#5-project-objectives)
6. [Dataset Description](#6-dataset-description)
7. [Feature Description](#7-feature-description)
8. [End-to-End Machine Learning Methodology](#8-end-to-end-machine-learning-methodology)
9. [Exploratory Data Analysis](#9-exploratory-data-analysis)
10. [Data Preprocessing](#10-data-preprocessing)
11. [Why This Is a Regression Problem](#11-why-this-is-a-regression-problem)
12. [Model Selection](#12-model-selection)
13. [Why HistGradientBoostingRegressor Was Chosen](#13-why-histgradientboostingregressor-was-chosen)
14. [Hyperparameter Tuning](#14-hyperparameter-tuning)
15. [Model Evaluation](#15-model-evaluation)
16. [Inference / Prediction Function](#16-inference--prediction-function)
17. [Limitations and Academic Observations](#17-limitations-and-academic-observations)
18. [Conclusion](#18-conclusion)
19. [Viva / Presentation Questions and Answers](#19-viva--presentation-questions-and-answers)
20. [Appendix A. Cell-by-Cell and Line-by-Line Code Explanation](#appendix-a-cell-by-cell-and-line-by-line-code-explanation)

## 1. Executive Summary

This project develops a supervised machine-learning regression system for predicting the median value of houses using demographic, geographic, housing-stock, income, and ocean-proximity information. The notebook
uses the California Housing Prices dataset and follows a structured machine-learning workflow: exploratory data analysis (EDA), data preprocessing, a baseline model, cross-validation-based model selection,
hyperparameter tuning, final evaluation, residual analysis, and inference.

The dataset contains 20,640 observations and 10 columns. The target variable is median_house_value. There are eight numerical predictors and one categorical predictor, ocean_proximity. The notebook identifies 207
missing values in total_bedrooms and no duplicate rows. The project uses median imputation for numerical missing values, most-frequent imputation for categorical values, one-hot encoding for ocean_proximity, and a
pipeline-based preprocessing design.

Five regression approaches are compared using 5-fold cross-validation: Linear Regression, Ridge, Lasso, Random Forest, and HistGradientBoosting. HistGradientBoosting gives the lowest cross-validated RMSE of approximately 48,098.67 and an RÂ² of approximately 0.827, so it is selected for tuning.

After GridSearchCV, the notebook reports a best cross-validation RMSE of approximately 47,327.97. The subsequently trained final model reports test RMSE = 46,957.65, MAE = 30,865.66, and RÂ² = 0.832. This means the
final model explains approximately 83.2% of the variation in the target on the test set.

An important observation is that the GridSearchCV output identifies l2_regularization = 0.0 as the best parameter combination, while the manually reconstructed final pipeline uses l2_regularization =
0.1. Therefore, the reported final test metrics belong to the manually reconstructed model rather than an exact reproduction of grid.best_estimator\_. This should be corrected before a formal academic
submission if exact reproducibility is required.

## 2. Project Overview

| Item | Description |
| --- | --- |
| **Project type** | Supervised machine learning -- regression |
| **Business domain** | Real estate / property valuation |
| **Input** | House/block-level geographic, demographic, housing and location attributes |
| **Output** | Predicted median house value in US dollars |
| **Dataset size** | 20,640 rows Ã— 10 columns |
| **Numerical predictors** | 8 |
| **Categorical predictors** | 1 |
| **Target** | median_house_value |
| **Primary metric** | RMSE |
| **Secondary metrics** | MAE and RÂ² |
| **Selected algorithm** | HistGradientBoostingRegressor |

## 3. Real-World Use Case

The practical use case is automated property valuation. A real-estate platform, mortgage team, property analyst, or investment organization can use historical property observations to estimate the expected median
value of a housing area.

### 3.1 Business Scenario

- A property analyst receives information about a housing area, such as location, median income, number of rooms, bedrooms, population, households, age, and ocean proximity.
- The system transforms the input into the same representation used during model training.
- The trained regression model estimates the median house value.
- The estimate can support initial valuation, market analysis, portfolio screening, or further human review.
- The prediction should be treated as a statistical estimate, not as a legally binding property appraisal.

### 3.2 Example

The notebook provides an inference example with longitude = -122.230, latitude = 37.880, housing_median_age = 41, total_rooms = 880, total_bedrooms = 129, population = 322, households = 126, median_income = 8.3252 and ocean_proximity = NEAR BAY. The reported prediction is approximately \$429,507.24.

## 4. Problem Statement

Given a set of housing and geographic characteristics for a housing block, predict the median house value. Because the output is a continuous numerical amount, the problem is formulated as a regression task.

Mathematically, the objective is to learn a function:

**Å· = f(X)**

where X represents the input features and Å· is the predicted median house value. The learning algorithm attempts to minimize prediction error on unseen data.

## 5. Project Objectives

- Understand the California housing dataset and identify the target variable.
- Perform exploratory data analysis to understand distributions, missing values, categorical balance, outliers, and feature relationships.
- Prepare numerical and categorical variables using reproducible preprocessing pipelines.
- Build Linear Regression as a baseline.
- Compare multiple regression algorithms using 5-fold cross-validation.
- Select the strongest model using RMSE as the primary selection metric.
- Tune the selected model using GridSearchCV.
- Evaluate the final model on a previously unseen test set.
- Inspect residuals to understand prediction errors.
- Provide a reusable prediction function for a single new observation.

## 6. Dataset Description

The notebook identifies the source as the California Housing Prices dataset and references the Kaggle dataset page. The loaded DataFrame has 20,640 observations and 10 columns. Nine columns are numerical according
to the notebook\'s numeric selection, with median_house_value being the target, and ocean_proximity is categorical.

### 6.1 Data Quality Findings

- 20,640 records were loaded.
- total_bedrooms contains 20,433 non-null values, meaning 207 values are missing.
- The other displayed fields contain 20,640 non-null values.
- The notebook reports zero duplicate rows.
- ocean_proximity contains five categories: <1H OCEAN, INLAND, NEAR OCEAN, NEAR BAY, and ISLAND.
- The target median_house_value has a maximum of 500,001, and the notebook explicitly notes that the target is capped/right-skewed.

### 6.2 Descriptive Statistics

| **Variable** | **Observed summary** |
| --- | --- |
| **median_income** | Mean 3.871; median 3.535; min 0.5; max 15.0 |
| **median_house_value** | Mean $206,855.82; median $179,700; min $14,999; max $500,001 |
| **total_rooms** | Mean 2,635.76; median 2,127; max 39,320 |
| **population** | Mean 1,425.48; median 1,166; max 35,682 |
| **households** | Mean 499.54; median 409; max 6,082 |

## 7. Feature Description

| Feature | Meaning |
| --- | --- |
| **longitude** | Geographic longitude of the housing block. |
| **latitude** | Geographic latitude of the housing block. |
| **housing_median_age** | Median age of houses in the block. |
| **total_rooms** | Total number of rooms in the block. |
| **total_bedrooms** | Total number of bedrooms in the block. |
| **population** | Total population in the block. |
| **households** | Total number of households in the block. |
| **median_income** | Median household income, represented in tens of thousands of US dollars. |
| **ocean_proximity** | Categorical location relationship to the ocean. |
| **median_house_value** | Target variable: median house value in US dollars. |

## 8. End-to-End Machine Learning Methodology

The notebook explicitly follows this sequence: EDA â†’ preprocessing â†’ baseline model â†’ model selection (cross-validation) â†’ tuning(GridSearchCV) â†’ final evaluation â†’ inference.

### 8.1 Pipeline

- Load the CSV dataset.
- Inspect structure and data quality.
- Separate X (features) and y (target).
- Split the dataset into 80% training and 20% testing data.
- Create separate preprocessing paths for numerical and categorical features.
- Build a Linear Regression baseline.
- Evaluate several candidate regressors using 5-fold cross-validation.
- Select HistGradientBoosting based on lowest CV RMSE.
- Tune HistGradientBoosting hyperparameters using GridSearchCV.
- Retrain the selected configuration on training data.
- Evaluate only once on the held-out test set for final performance.
- Use the fitted pipeline for new-house inference.

## 9. Exploratory Data Analysis

EDA is used before modeling to understand the dataset and identify
problems that could affect model training. The notebook performs column
inspection, data-type inspection, missing-value analysis, duplicate
detection, descriptive statistics, categorical counts, target
distribution analysis, feature distributions, boxplots, correlation
analysis, and target-correlation analysis.

### 9.1 Important EDA Findings

- The dataset contains both numerical and categorical information.
- total_bedrooms has 207 missing values.
- There are no duplicate rows.
- Several numerical variables are highly skewed and contain outliers.
- median_house_value is right-skewed and capped at 500,001.
- median_income has the strongest numerical correlation with the target among the displayed correlations: approximately 0.688.
- The notebook notes high multicollinearity among several room/population-related variables.
- The ocean_proximity categories are imbalanced, with <1H OCEAN and INLAND much more common than ISLAND.

The correlation coefficient is useful for initial understanding, but it is not sufficient by itself to decide which variables should be retained. Tree-based models can learn nonlinear relationships and interactions that a simple correlation coefficient does not capture.

## 10. Data Preprocessing

Preprocessing is designed around the actual structure of the data. Numerical features need missing-value handling and scaling for linear models; the categorical feature needs encoding. The notebook places
these operations inside scikit-learn pipelines so the transformations are learned only from the relevant training fold during cross-validation.

### 10.1 Numerical preprocessing

- SimpleImputer(strategy='median') replaces missing numerical values with the median calculated from the training data.
- StandardScaler() standardizes numerical variables by centering and scaling them.
- Scaling is especially important for Linear Regression, Ridge and Lasso because their coefficients and regularization operate on feature magnitudes.
- The same preprocessing pipeline is retained for the final model, giving a consistent inference path.

### 10.2 Categorical preprocessing

- SimpleImputer(strategy='most_frequent') fills missing categorical values with the most frequent category.
- OneHotEncoder(handle_unknown='ignore') converts categories into machine-readable indicator variables.
- handle_unknown='ignore' makes inference safer when a future observation contains a category not observed during training.

### 10.3 Why Pipeline and ColumnTransformer?

Pipeline chains preprocessing and modeling into one object. ColumnTransformer applies different transformations to different column groups. Together they reduce the risk of inconsistent preprocessing and
data leakage during cross-validation. This is particularly important academically because preprocessing should be fitted using training data rather than the complete dataset before model validation.

## 11. Why This Is a Regression Problem

Classification predicts discrete classes such as low/medium/high. Regression predicts a continuous numerical quantity. median_house_value is a monetary amount, so regression is the correct problem formulation.

## 11.1 Regression Metrics

RMSE (Root Mean Squared Error):

**RMSE = âˆš\[(1/n) Î£(yáµ¢ âˆ’ Å·áµ¢)Â²\]**

RMSE penalizes larger errors more strongly because the errors are squared before averaging. The notebook uses RMSE as the primary model-selection metric, which is reasonable for property-value prediction because large valuation errors should receive greater penalty.

MAE (Mean Absolute Error):

**MAE = (1/n) Î£\|yáµ¢ âˆ’ Å·áµ¢\|**

MAE gives the average absolute prediction error in the same unit as the target. It is easier to interpret than RMSE.

RÂ² (Coefficient of Determination):

**RÂ² = 1 âˆ’ \[Î£(yáµ¢ âˆ’ Å·áµ¢)Â² / Î£(yáµ¢ âˆ’ È³)Â²\]**

RÂ² indicates how much variation in the target is explained by the model relative to a mean-prediction baseline. An RÂ² of 0.832 means the final test predictions explain about 83.2% of the observed target variation in this dataset.

## 12. Model Selection

The project deliberately compares several models rather than assuming one algorithm will always be best. The candidate models are Linear Regression, Ridge, Lasso, Random Forest, and HistGradientBoosting.

| Model | CV RMSE | CV MAE | CV RÂ² |
| --- | --- | --- | --- |
| **HistGB** | 48,098.666 | 32,208.412 | 0.827 |
| **RandomForest** | 49,519.130 | 32,295.725 | 0.817 |
| **Ridge** | 68,595.617 | 49,664.331 | 0.648 |
| **Lasso** | 68,603.233 | 49,667.263 | 0.648 |
| **LinearRegression** | 68,604.163 | 49,667.159 | 0.648 |

HistGradientBoosting has the lowest CV RMSE and the highest CV RÂ² in the notebook. It improves substantially over the linear models, indicating that the relationship between the housing variables and house value is
not well represented by a simple linear function.

## 13. Why HistGradientBoostingRegressor Was Chosen

### 13.1 Conceptual reason

House prices depend on nonlinear combinations of location, income, rooms, population, households and proximity to the ocean. For example, an increase in income may have a different effect in a coastal area than in an inland area. A linear model assumes a fixed additive relationship unless additional nonlinear features are explicitly created. A gradient-boosting tree model can learn nonlinear decision boundaries and interactions automatically.

### 13.2 How gradient boosting works

Gradient boosting builds an additive sequence of weak decision-tree learners. The first learner makes an initial prediction. Later trees focus on reducing the remaining prediction error. Each new tree is added
in a direction that improves the chosen loss function. The final prediction is the combined contribution of the trees.

### 13.3 Why it fits this project

- The dataset has nonlinear relationships and interactions.
- Tree-based boosting does not require the same linearity assumption as Linear Regression.
- It can model threshold-like effects, such as different price behavior at different income or geographic levels.
- The cross-validation results in the notebook empirically support this choice.
- It performs better than Random Forest and all three linear approaches on the notebook's primary CV metric.

The choice is therefore supported by both domain reasoning and measured validation performance, which is stronger than choosing an algorithm only because it is popular.

## 14. Hyperparameter Tuning

After selecting HistGradientBoosting, the notebook searches over five hyperparameters using GridSearchCV and 5-fold cross-validation.

| Parameter | Search values | Purpose |
| --- | --- | --- |
| **learning_rate** | 0.03, 0.05, 0.1 | Controls the contribution of each boosting stage. |
| **max_depth** | None, 3, 6 | Controls tree depth and therefore model complexity. |
| **max_leaf_nodes** | 15, 31, 63 | Controls the maximum number of leaves per tree. |
| **min_samples_leaf** | 20, 50, 100 | Controls the minimum number of samples allowed in a leaf. |
| **l2_regularization** | 0.0, 0.1, 1.0 | Adds L2 regularization to reduce excessive model complexity. |

There are 3 Ã— 3 Ã— 3 Ã— 3 Ã— 3 = 243 hyperparameter combinations. With 5-fold cross-validation, the notebook reports 1,215 fits (243 Ã— 5).

GridSearchCV reports the best CV RMSE as approximately 47,327.97 with the parameter combination: learning_rate = 0.1, max_depth = None, max_leaf_nodes = 63, min_samples_leaf = 20, l2_regularization = 0.0.

### 14.1 Important reproducibility issue

The notebook later constructs hgb_best manually with l2_regularization = 0.1 instead of the GridSearchCV-reported best value 0.0. This is a small but important implementation inconsistency. For a formal academic project, the safest approach is to use grid.best_estimator\_ directly or set every parameter exactly equal to grid.best_params\_.

hgb_best = grid.best_estimator\_

Using grid.best_estimator\_ avoids accidentally changing one hyperparameter between tuning and final evaluation.

## 15. Model Evaluation

The notebook reports the following final performance for the manually reconstructed tuned HGB model:

| Evaluation | RMSE | MAE | RÂ² |
| --- | --- | --- | --- |
| **Baseline Linear Regression -- Test** | 70,059.193 | 50,670.489 | 0.625 |
| **Final Tuned HGB -- Train** | 36,094.823 | 24,370.386 | 0.903 |
| **Final Tuned HGB -- Test** | 46,957.652 | 30,865.657 | 0.832 |

Compared with the baseline test RMSE of 70,059.19, the final test RMSE of 46,957.65 is substantially lower. The final test RÂ² also increases from 0.625 to 0.832. This demonstrates that the selected nonlinear
boosting approach captures substantially more of the target relationship than the baseline linear model.

The difference between train RÂ² (0.903) and test RÂ² (0.832) indicates some generalization gap, but the model still retains strong predictive performance on unseen data. The test metrics are the more important numbers for reporting real predictive capability.

## 16. Inference / Prediction Function

The notebook defines predict_house_price(), which accepts the model and the nine input features required by the trained pipeline. It creates a one-row DataFrame and sends it through model.predict().

```python

def predict_house_price(\
model,\
longitude: float,\
latitude: float,\
housing_median_age: float,\
total_rooms: float,\
total_bedrooms: float,\
population: float,\
households: float,\
median_income: float,\
ocean_proximity: str\
) -\> float:

```

The function is useful because it provides a simple interface between the machine-learning model and a future application such as a web form, API, dashboard, or property-analysis tool. Because preprocessing is embedded in the pipeline, the caller does not need to manually perform imputation or one-hot encoding.

## 17. Limitations and Academic Observations

- The target is capped at 500,001 in the supplied dataset, so predictions near the upper end may be affected by target censoring.
- The dataset represents historical housing conditions and may not represent current real-estate markets.
- The dataset is California-specific; the model should not be assumed to generalize to other regions without retraining and validation.
- The notebook does not perform explicit feature engineering such as rooms-per-household or bedrooms-per-room ratios.
- Correlation does not prove causation; a high correlation between income and price does not mean income alone causes the observed prices.
- The project does not report confidence intervals or prediction intervals, so uncertainty around individual predictions is not quantified.
- The final manually constructed model does not exactly match the GridSearchCV-reported best l2_regularization value.
- For production use, model monitoring, drift detection, data validation, versioning, and retraining procedures would be required.

### 17.1 Recommended academic correction

Replace the manually reconstructed final model with grid.best_estimator\_, evaluate that exact object on X_test/y_test, and report the resulting metrics. This makes the tuning and final evaluation logically consistent and reproducible.

## 18. Conclusion

This project demonstrates a complete supervised machine-learning workflow for a real-world regression problem. The project starts with data understanding and EDA, addresses missing values and mixed feature types, creates a baseline, compares several algorithms using cross-validation, tunes the strongest model, evaluates it on unseen data, and exposes a reusable prediction function.

HistGradientBoostingRegressor is the most appropriate model among the tested approaches because it achieves the lowest cross-validation RMSE and highest cross-validation RÂ² in the notebook. The reported final test
performance (RMSE 46,957.65, MAE 30,865.66, RÂ² 0.832) shows good predictive capability for this dataset.

From an academic perspective, the strongest part of the project is the model-selection methodology: the final algorithm is not chosen arbitrarily; it is supported by cross-validation results. The main improvement needed is to ensure the final trained model exactly matches the hyperparameters selected by GridSearchCV.

## 19. Viva / Presentation Questions and Answers

**Q:** Why is this a regression problem?

**A:** Because median_house_value is a continuous numerical target rather than a discrete class.

**Q:** Why did you use Linear Regression first?

**A:** It is simple, interpretable and provides a baseline against which more complex models can be compared.

**Q:** Why did HistGradientBoosting outperform Linear Regression?

**A:** The housing-price relationship is nonlinear and contains interactions. Boosted trees can learn these patterns without manually specifying nonlinear equations.

**Q:** Why use cross-validation?

**A:** It gives a more reliable estimate of model performance than relying on one training/validation split and helps compare models using multiple folds.

**Q:** Why is RMSE the primary metric?

**A:** RMSE strongly penalizes large errors, which is appropriate when large house-price prediction errors are especially undesirable.

**Q:** Why use MAE as a secondary metric?

**A:** MAE gives an easy-to-understand average absolute error in dollars.

**Q:** What does RÂ² = 0.832 mean?

**A:** Approximately 83.2% of the variation in the test target is explained by the model relative to a mean-prediction baseline.

**Q:** Why use median imputation?

**A:** It is robust to extreme values and allows the 207 missing total_bedrooms records to be retained.

**Q:** Why one-hot encode ocean_proximity?

**A:** The model needs numerical input; one-hot encoding represents each category without imposing an artificial numeric ordering.

**Q:** Why use handle_unknown=\'ignore\'?

**A:** It prevents prediction from failing if an unseen category appears during inference.

**Q:** What is data leakage?

**A:** It occurs when information from validation/test data influences training or preprocessing. The pipeline helps prevent this during CV.

**Q:** What is the main issue in the current final model implementation?

**A:** GridSearchCV reports l2_regularization=0.0 as best, but the manually rebuilt hgb_best uses 0.1. The exact best estimator should be used for reproducibility.

## Appendix A. Cell-by-Cell and Line-by-Line Code Explanation

The following explanation is based directly on the code cells in the supplied notebook. Output-heavy HTML/CSS generated by Jupyter and scikit-learn is not treated as project logic; the explanations focus on executable Python statements written by the project.

### Appendix A.1 -- Install Required Libraries

```python
%pip install -q numpy pandas matplotlib seaborn scikit-learn
```

**Purpose**: Installs all necessary Python libraries for data processing, visualization, and machine learning. The `-q` flag suppresses verbose installation output.

### Appendix A.2 -- Import All Required Libraries

```python
import numpy as np\
import pandas as pd\
import matplotlib.pyplot as plt\
import seaborn as sns\
\
from sklearn.model_selection import train_test_split, KFold,
cross_validate, GridSearchCV\
from sklearn.pipeline import Pipeline\
from sklearn.compose import ColumnTransformer\
from sklearn.preprocessing import StandardScaler, OneHotEncoder\
from sklearn.impute import SimpleImputer\
\
from sklearn.linear_model import LinearRegression, Ridge, Lasso\
from sklearn.ensemble import RandomForestRegressor,
HistGradientBoostingRegressor\
\
from sklearn.metrics import (\
mean_absolute_error,\
root_mean_squared_error,\
r2_score\
)
```

**Purpose**: Imports all required libraries for data processing (NumPy, Pandas), visualization (Matplotlib, Seaborn), and machine learning (scikit-learn). scikit-learn tools are imported for model training, preprocessing, evaluation, and hyperparameter tuning.

### Appendix A.3 -- Configure Pandas, Seaborn, and Matplotlib Settings

```python

\# configurations\
pd.set_option(\"display.max_columns\", None)\
pd.set_option(\"display.float_format\", lambda x: f\"{x:.3f}\")\
sns.set_theme(style=\"darkgrid\")\
\
plt.rcParams.update({\
\"axes.titlesize\": 10,\
\"axes.labelsize\": 9,\
\"xtick.labelsize\": 8,\
\"ytick.labelsize\": 8\
})\
\
RANDOM_STATE = 42\
CSV_PATH = \"housing.csv\" \# update path for a different dataset\
TARGET_COL = \"median_house_value\" \# target column name

```

**Purpose**: Configures display settings for better readability (show all columns, format decimals to 3 places). Sets up consistent plot themes. Fixes random seed to 42 for reproducible results. Defines key constants (CSV file path and target column name) used throughout the analysis.

### Appendix A.4 -- Load the Dataset

```python
df = pd.read_csv(CSV_PATH)
```

**Purpose**: Reads the housing dataset from CSV into a Pandas DataFrame for analysis.

### Appendix A.5 -- Check DataFrame Dimensions

```python
print("DataFrame shape:", df.shape)
```

**Purpose**: Prints the dataset dimensions (rows and columns) to verify data was loaded correctly.

### Appendix A.6 -- Display First Five Rows

```python
df.head()
```

**Purpose**: Shows the first 5 rows to verify the data loaded correctly and examine the structure.

### Appendix A.7 -- Display Column Names

```python
df.columns
```

**Purpose**: Lists all column names to confirm which features and target are available.

### Appendix A.8 -- Inspect Data Types and Missing Values

```python
# basic dataset overview
df.info()
```

**Purpose**: Shows data types, non-null value counts, and memory usage to quickly identify data quality issues (missing values, incorrect data types).

### Appendix A.9 -- Identify Numerical and Categorical Columns

```python
num_cols = df.select_dtypes(include=\[np.number\]).columns.tolist()\
cat_cols = df.select_dtypes(include=\[\"object\"\]).columns.tolist()\
\
print(\"Target column:\", TARGET_COL)\
print(\"Numerical columns:\", num_cols)\
print(\"Categorical columns:\", cat_cols)
```

**Purpose**: Automatically separates columns into numerical and categorical types. Confirms the target column and displays which features will be used for each data preprocessing path.

### Appendix A.10 -- Analyze Missing Values

```python
# missing values analysis
print("\nMissing values per column:")
print(df.isna().sum())
```

**Purpose**: Counts missing values in each column. The dataset has 207 missing values in `total_bedrooms` that will require imputation.

### Appendix A.11 -- Examine Value Distributions

```python
# check presence of encoded missing values
for col in df.columns:
    print(df[col].value_counts().head(20))
```

**Purpose**: Shows the 20 most frequent values in each column to identify categories, data patterns, and any unusual encoding of missing values (like -999 or 'N/A').

### Appendix A.12 -- Detect Duplicate Rows

```python
# duplicates
duplicate_mask = df.duplicated()
num_duplicates = duplicate_mask.sum()
print("Number of duplicate rows:", num_duplicates)
```

**Purpose**: Identifies and counts duplicate rows in the dataset. The results show zero duplicates, meaning each row is unique.
### Appendix A.13 -- Summary Statistics for Numerical Features

```python
# descriptive stat
df[num_cols].describe()
```

**Purpose**: Displays mean, median, std deviation, min, max, and quartiles for all numerical columns to understand data distributions.

### Appendix A.14 -- Summary Statistics (Transposed View)

```python
# descriptive stat
df[num_cols].describe().T
```

**Purpose**: Transposes the statistics table so each feature appears as a row, making it easier to compare statistics across columns.

### Appendix A.15 -- Visualize Categorical Feature Distributions

```python
# countplot for categorical columns
for col in cat_cols:
    plt.figure(figsize=(10, 3))
    sns.countplot(x=col, data=df)
    plt.title(f"Distribution of {col}")
    plt.show()
```

**Purpose**: Creates count plots for each categorical column to visualize the frequency distribution of categories.
### Appendix A.16 -- Display Categorical Value Counts

```python
for col in cat_cols:
    print(df[col].value_counts())
```

**Purpose**: Prints frequency counts for each categorical column to see distribution and identify rare categories.
### Appendix A.17 -- Visualize Target Distribution

```python
# target column distribution
plt.figure(figsize=(6,4))
sns.histplot(df[TARGET_COL], bins=40, kde=True)
plt.title("Target Distribution: Median House Value")
plt.xlabel("Median House Value")
plt.show()
```

**Purpose**: Creates a histogram with kernel density estimation to visualize the distribution of the target variable (median house value).

### Appendix A.18 -- Display Target Value Frequencies

```python
df[TARGET_COL].value_counts()
# higher cap
```

**Purpose**: Shows frequency counts of target values to identify how many houses fall into each price range.
### Appendix A.19 -- Visualize Numerical Feature Distributions

```python
# histogram plot - distribution
fig, axes = plt.subplots(3, 3, figsize=(8, 6))
axes = axes.flatten()

for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, ax=axes[i])
    axes[i].set_title(col, fontsize=8)

plt.tight_layout()
plt.show()
```

**Purpose**: Creates a 3x3 grid of histograms showing the distribution of each numerical feature with kernel density estimation curves.
### Appendix A.20 -- Detect Outliers in Numerical Features

```python
# outliers analysis - boxplot
fig, axes = plt.subplots(3, 3, figsize=(8, 6))
axes = axes.flatten()

for i, col in enumerate(num_cols):
sns.boxplot(x=df\[col\], ax=axes\[i\])
axes\[i\].set_title(col, fontsize=8)
axes\[i\].set_xlabel("")

plt.tight_layout()
plt.show()
```

**Purpose**: Creates a 3x3 grid of boxplots to visualize the distribution and detect outliers in each numerical feature.

### Appendix A.21 -- Visualize Correlation Between Features

```python
# identify presence of highly correlated columns & feature relationships
plt.figure(figsize=(10, 5))
sns.heatmap(
df\[num_cols\].corr(),
annot=True,
cmap="coolwarm",
center=0
)
plt.title("Correlation Heatmap")
plt.show()
```

**Purpose**: Creates a heatmap showing correlation coefficients between all numerical features to identify multicollinearity.

### Appendix A.22 -- Analyze Feature Correlations with Target

```python
# Correlation with target
corr_with_target = df\[num_cols\].corr()\[TARGET_COL\].sort_values(ascending=False)
print("\nCorrelation with target:")
print(corr_with_target)
```

**Purpose**: Computes correlation coefficients between each numerical feature and the target variable, sorted in descending order to identify which features have the strongest relationship with house prices.

### Appendix A.23 -- Separate Features from Target Variable

```python
# separate features and target
X = df.drop(columns=\[TARGET_COL\])
y = df\[TARGET_COL\]
```

**Purpose**: Separates the feature matrix (X) from the target variable (y). X contains all columns except the median house value, and y contains only the target variable for model training.

### Appendix A.24 -- Preview Training Features

```python
X.head()
```

**Purpose**: Displays the first 5 rows of features to verify the data structure before model training.
### Appendix A.25 -- Preview Target Variable

```python
y.head()
```

**Purpose**: Displays the first 5 values of the target variable (`median_house_value`) to verify extraction.
### Appendix A.26 -- Split Data into Training and Testing Sets

```python
# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE
)
```

**Purpose**: Divides the data into 80% training and 20% testing. This ensures model evaluation on unseen data. `random_state` ensures reproducible splits.

### Appendix A.27 -- Verify Data Split Dimensions

```python
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
```

**Purpose**: Confirms the size of training and test sets to ensure proper data split.

### Appendix A.28 -- Identify Numerical and Categorical Features

```python
numerical_features = X_train.select_dtypes(include=\[np.number\]).columns.tolist()
categorical_features = X_train.select_dtypes(exclude=\[np.number\]).columns.tolist()

print("Numerical features:", numerical_features)
print("Categorical features:", categorical_features)

# numerical features - preprocessing steps
numerical_transformer = Pipeline(
steps=\[
("imputer", SimpleImputer(strategy="median")),
("scaler", StandardScaler())
\]
)

# categorical features - preprocessing steps
categorical_transformer = Pipeline(
steps=\[
("imputer", SimpleImputer(strategy="most_frequent")),
("onehot", OneHotEncoder(handle_unknown="ignore"))
\]
)

# preprocessing pipeline
preprocess = ColumnTransformer(
transformers=\[
("num", numerical_transformer, numerical_features),
("cat", categorical_transformer, categorical_features)
\]
)
```

**Purpose**: Identifies numerical and categorical features from training data. Builds preprocessing pipelines: numerical features get median imputation + scaling, categorical features get most-frequent imputation + one-hot encoding. Combines both pipelines into a single ColumnTransformer for coordinated preprocessing.

### Appendix A.29 -- Create Baseline Pipeline with Linear Regression

```python
baseline_pipe = Pipeline(
steps=\[
("preprocess", preprocess),
("model", LinearRegression())
\]
)
```

**Purpose**: Combines the preprocessing pipeline and Linear Regression model into a single pipeline that will automatically preprocess data before training the baseline model.

### Appendix A.30 -- Train Baseline Linear Regression Model

```python
# preprocess the data and train the baseline model
baseline_pipe.fit(X_train, y_train)
```

**Purpose**: Fits the preprocessing pipeline and trains the baseline Linear Regression model on training data.

### Appendix A.31 -- Generate Baseline Predictions

```python
train_baseline_pred = baseline_pipe.predict(X_train)
test_baseline_pred = baseline_pipe.predict(X_test)
```

**Purpose**: Uses the trained baseline model to generate predictions on both training and test datasets.

### Appendix A.32 -- Display Sample Predictions

```python
train_baseline_pred[:5]
```

**Purpose**: Shows the first 5 predictions from the baseline Linear Regression model for inspection.

### Appendix A.33 -- Display Sample Actual Values

```python
y_train[:5]
```

**Purpose**: Shows the first 5 actual target values for side-by-side comparison with predictions.

### Appendix A.34 -- Calculate Baseline Training Metrics

```python
train_baseline_rmse = root_mean_squared_error(y_train, train_baseline_pred)
train_baseline_mae = mean_absolute_error(y_train, train_baseline_pred)
train_baseline_r2 = r2_score(y_train, train_baseline_pred)

print("\n=== TRAIN BASELINE METRICS (LinearRegression) ===")
print(f"RMSE: {train_baseline_rmse:.3f}")
print(f"MAE : {train_baseline_mae:.3f}")
print(f"R2 : {train_baseline_r2:.3f}")
```

**Purpose**: Computes three evaluation metrics (RMSE, MAE, R²) on training predictions to assess baseline Linear Regression model performance.

### Appendix A.35 -- Calculate Baseline Test Metrics

```python
test_baseline_rmse = root_mean_squared_error(y_test, test_baseline_pred)
test_baseline_mae = mean_absolute_error(y_test, test_baseline_pred)
test_baseline_r2 = r2_score(y_test, test_baseline_pred)

print("\n=== TEST BASELINE METRICS (LinearRegression) ===")
print(f"RMSE: {test_baseline_rmse:.3f}")
print(f"MAE : {test_baseline_mae:.3f}")
print(f"R2 : {test_baseline_r2:.3f}")
```

**Purpose**: Computes three evaluation metrics (RMSE, MAE, R²) on test predictions to assess baseline Linear Regression model generalization to unseen data.

### Appendix A.36 -- Define Models for Comparison

```python
# models to try
models = {
"LinearRegression": LinearRegression(),
"Ridge": Ridge(random_state=RANDOM_STATE),
"Lasso": Lasso(random_state=RANDOM_STATE, max_iter=10000),
"RandomForest": RandomForestRegressor(),
"HistGB": HistGradientBoostingRegressor()
}
```

**Purpose**: Creates a dictionary of five regression models to compare: three linear variants (Linear, Ridge, Lasso) and two ensemble methods (Random Forest, Histogram Gradient Boosting) for nonlinear relationships.
### Appendix A.37 -- Configure Cross-Validation

```python
k = 5
cv = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
```

**Purpose**: Sets up 5-fold cross-validation with shuffling. Data is divided into 5 groups; each is used once as a test set while the other 4 are used for training.

### Appendix A.38 -- Define Scoring Metrics for Cross-Validation

```python
scoring = {
"rmse": "neg_root_mean_squared_error",
"mae": "neg_mean_absolute_error",
"r2": "r2"
}
```

**Purpose**: Configures three evaluation metrics to use during cross-validation: RMSE and MAE (negated for compatibility with scikit-learn's maximization convention) and R² (higher is better).
### Appendix A.39 -- Perform Cross-Validation Model Comparison

```python
rows = \[\]

for name, model in models.items():
    pipe = Pipeline(
        steps=\[
            ("preprocess", preprocess),
            ("model", model)
        \]
    )
    scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
    rows.append({
        "model": name,
        "cv_rmse": -scores\["test_rmse"\].mean(),
        "cv_mae": -scores\["test_mae"\].mean(),
        "cv_r2": scores\["test_r2"\].mean()
    })

# sort based on lowest rmse value
cv_results = pd.DataFrame(rows).sort_values("cv_rmse")
print("=== CV Model Comparison ===")
print(cv_results)
```

**Purpose**: Performs 5-fold cross-validation for each of the five candidate models. Collects average RMSE, MAE, and R² for each model and sorts them by RMSE to identify the best-performing model.
### Appendix A.40 -- Identify Best Model from Cross-Validation

```python
best_row = cv_results.sort_values("cv_rmse").iloc\[0\]

best_model_name = best_row\["model"\]
best_rmse = best_row\["cv_rmse"\]

print("Best model based on CV RMSE:")
print("Model :", best_model_name)
print("CV RMSE:", best_rmse)
```

**Purpose**: Extracts the top-performing model (lowest CV RMSE) from the comparison results and displays its name and cross-validation RMSE score.
### Appendix A.41 -- Create Pipeline with Histogram Gradient Boosting Model

```python
hgb_pipe = Pipeline(
steps=\[
("preprocess", preprocess),
("model", HistGradientBoostingRegressor(random_state=RANDOM_STATE))
\]
)
```

**Purpose**: Builds a pipeline combining the preprocessing pipeline with a Histogram Gradient Boosting Regressor, the best-performing model identified from cross-validation.

### Appendix A.42 -- Define Hyperparameter Grid for Tuning

```python
# hyperparameters combination
param_grid = {
"model\_\_learning_rate": \[0.03, 0.05, 0.1\],
"model\_\_max_depth": \[None, 3, 6\],
"model\_\_max_leaf_nodes": \[15, 31, 63\],
"model\_\_min_samples_leaf": \[20, 50, 100\],
"model\_\_l2_regularization": \[0.0, 0.1, 1.0\]
}
```

**Purpose**: Defines a grid of hyperparameter combinations to test: learning rate, tree depth, leaf nodes, minimum samples per leaf, and L2 regularization strength. The double underscore notation (model__) indicates these parameters belong to the model step in the pipeline.

### Appendix A.43 -- Configure GridSearchCV for Hyperparameter Tuning

```python
grid = GridSearchCV(
    estimator=hgb_pipe,
    param_grid=param_grid,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)
```

**Purpose**: Sets up GridSearchCV to systematically search all hyperparameter combinations using 5-fold cross-validation. Uses RMSE as the scoring metric and parallelizes across all available processors.

### Appendix A.44 -- Execute Grid Search

```python
# perform grid search
grid.fit(X_train, y_train)
```

**Purpose**: Executes the grid search on training data, fitting each hyperparameter combination and tracking cross-validation performance.
### Appendix A.45 -- Display Best Hyperparameters from Grid Search

```python
print("\\n=== TUNED HistGB (CV) ===")
print("Best CV RMSE:", -grid.best_score\_)
print("Best params:", grid.best_params\_)
```

**Purpose**: Reports the best cross-validation RMSE score achieved and displays the optimal hyperparameters found by grid search.
### Appendix A.46 -- Reconstruct Final Pipeline with Best Hyperparameters

```python
hgb_best = Pipeline(
steps=\[
("preprocess", preprocess),
("model", HistGradientBoostingRegressor(
l2_regularization=0.1,
learning_rate=0.1,
max_depth=None,
max_leaf_nodes=63,
min_samples_leaf=20
))
\]
)
```

**Purpose**: Manually recreates the best pipeline found by grid search, combining preprocessing with Histogram Gradient Boosting using the optimal hyperparameters for training on the full dataset.
### Appendix A.47 -- Train Final Model on Full Training Data

```python
# train best model on entire training data (can also be done with refit=True in grid search)
hgb_best.fit(X_train, y_train)
```

**Purpose**: Fits the final pipeline with best hyperparameters on the complete training dataset to maximize learning before evaluation.
### Appendix A.48 -- Evaluate Final Model on Training Data

```python
train_final_pred = hgb_best.predict(X_train)

train_final_rmse = root_mean_squared_error(y_train, train_final_pred)
train_final_mae = mean_absolute_error(y_train, train_final_pred)
train_final_r2 = r2_score(y_train, train_final_pred)

print("\n=== FINAL MODEL (Tuned HGB) Train Performance ===")
print(f"RMSE: {train_final_rmse:.3f}")
print(f"MAE : {train_final_mae:.3f}")
print(f"R2 : {train_final_r2:.3f}")
```

**Purpose**: Generates predictions on training data and computes RMSE, MAE, and R² to assess how well the final model fits the training set.
### Appendix A.49 -- Evaluate Final Model on Test Data

```python
test_final_pred = hgb_best.predict(X_test)

test_final_rmse = root_mean_squared_error(y_test, test_final_pred)
test_final_mae = mean_absolute_error(y_test, test_final_pred)
test_final_r2 = r2_score(y_test, test_final_pred)

print("\n=== FINAL MODEL (Tuned HGB) Test Performance ===")
print(f"RMSE: {test_final_rmse:.3f}")
print(f"MAE : {test_final_mae:.3f}")
print(f"R2 : {test_final_r2:.3f}")
```

**Purpose**: Generates predictions on unseen test data and computes RMSE, MAE, and R² to evaluate final model generalization and real-world performance.
### Appendix A.50 -- Visualize Residuals and Model Errors

```python
# residual plot
residuals = y_test - test_final_pred

plt.figure(figsize=(6, 4))
plt.scatter(test_final_pred, residuals, s=10)
plt.axhline(0)
plt.title("Residuals vs Predictions")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(residuals, bins=40, kde=True)
plt.title("Residual Distribution")
plt.xlabel("Residual")
plt.ylabel("Count")
plt.show()
```

**Purpose**: Creates two diagnostic plots: a scatter plot of residuals versus predictions (to check for systematic errors) and a histogram of residuals (to verify normality assumption).
### Appendix A.51 -- Define Inference Function for Single House Prediction

```python
def predict_house_price(
    model,
    longitude: float,
    latitude: float,
    housing_median_age: float,
    total_rooms: float,
    total_bedrooms: float,
    population: float,
    households: float,
    median_income: float,
    ocean_proximity: str
) -\> float:
    """
    Predict median_house_value for one new house.
    total_bedrooms can be np.nan (pipeline will impute).
    """
    new_row = pd.DataFrame(\[{
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "ocean_proximity": ocean_proximity
    }\])
    
    return float(model.predict(new_row)\[0\])
```

**Purpose**: Defines a reusable function that takes house characteristics as individual parameters, constructs a DataFrame, and returns a single predicted price. Handles missing bedroom values through the pipeline's imputation.

### Appendix A.52 -- Test Inference Function with Example Data

```python
# Example inference
example_pred = predict_house_price(
    model=hgb_best,
    longitude=-122.230,
    latitude=37.880,
    housing_median_age=41,
    total_rooms=880,
    total_bedrooms=129,
    population=322,
    households=126,
    median_income=8.3252,
    ocean_proximity="NEAR BAY"
)

print("\nExample prediction:", round(example_pred, 2))
```

**Purpose**: Demonstrates the predict_house_price function by passing an example house with specific characteristics. Outputs the predicted median house value for that example observation.

