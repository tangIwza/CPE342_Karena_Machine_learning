# CPE342 Karena Task 3 (V6): Future Spending Prediction

This repository contains the solution for **CPE342 Task 3 (Karena)**, aimed at predicting the 30-day future spending of players (`spending_30d`). The solution implements a sophisticated **Two-Stage Full Stacking Ensemble** approach, optimized with **Optuna** and enhanced by **MICE Imputation**.

---

## 1. EDA Findings (Exploratory Data Analysis)
Analysis of the game data revealed distinct patterns necessitating a specific modeling approach:

* **Zero-Inflated Data:** A vast majority of players do not spend money ($0 value), while a small segment ("Whales") spends significantly. This implies a single regression model would struggle.
    * *Solution:* Split the problem into **Classification** (Will spend?) and **Regression** (How much?).
* **Skewed Distributions:** Key features like `historical_spending` and `total_playtime_hours` are heavily right-skewed.
* **Correlated Behaviors:** High spenders often have distinct behavioral markers, such as high `purchases_on_discount` ratios and specific `platform` preferences.

## ⚙️ 2. Data Preprocessing
A robust pipeline was built to handle missing values and outliers effectively:

* **Advanced Imputation (MICE):** Switched from simple mean/median filling to **IterativeImputer (MICE)**. This models each feature as a function of others, providing much more accurate estimates for missing player stats.
* **Feature Engineering:**
    * **Behavioral Ratios:** Created `spending_per_day`, `playtime_per_session`, and `interaction_per_friend` to normalize data across account ages.
    * **Whale Flags:** Added `is_whale` (Top 5% spenders) and `high_activity` (Top 10% playtime) to help the model identify VIPs explicitly.
* **Transformations:**
    * **Log Transformation:** Applied `np.log1p` to skewed columns to normalize distributions for linear models.
    * **Robust Scaling:** Used `RobustScaler` instead of StandardScaler to minimize the impact of extreme outliers (Whales).

## 3. Model Design
The core architecture is a **Hurdle Model (Two-Stage Prediction)** implemented via Stacking:

### Stage 1: Classification (Will the player spend?)
* **Objective:** Predict probability of `spending_30d > 0`.
* **Stacking Ensemble:**
    * **Base Learners:** `LGBMClassifier` (Tuned), `XGBClassifier` (Tuned), `LogisticRegression`.
    * **Meta Learner:** `LogisticRegression`.
    * **Optimization:** Hyperparameters tuned via **Optuna** to maximize ROC-AUC.

### Stage 2: Regression (If they spend, how much?)
* **Objective:** Predict `log1p(spending_30d)` for positive cases.
* **Stacking Ensemble:**
    * **Base Learners:** `LGBMRegressor`, `XGBRegressor`, `GradientBoostingRegressor`, `LassoCV`, `ElasticNetCV`.
    * **Meta Learner:** `Lasso` (L1 Regularization) to robustly blend predictions.
    * **Optimization:** Hyperparameters tuned via **Optuna** to minimize RMSE.

## 4. Evaluation & Results
The model uses **5-Fold Cross-Validation** to ensure generalization.

* **Metric:** The final output combines the Probability from Stage 1 and the Amount from Stage 2.
    * *Equation:* `Final_Prediction = Probability_Spend * Predicted_Amount`
* **Performance:**
    * **Classification:** High AUC scores indicated the model effectively separates F2P players from payers.
    * **Regression:** The stacking approach with Lasso meta-learner significantly reduced error variance compared to single models.

## 5. Insights Gained from Model Behavior
* **The Power of Two Stages:** Trying to predict spending directly with a single regressor often yields poor results because the model learns to predict near-zero values for everyone. Separating the "Zero" vs "Non-Zero" decision drastically improved accuracy.
* **Linear Models in Stacking:** While Tree-based models (XGB/LGBM) are excellent feature extractors, including linear models (`Lasso`, `ElasticNet`) in the stacking layer helped the model extrapolate better for high-spending "Whale" values.
* **Imputation Matters:** Using MICE (`IterativeImputer`) preserved the correlation structure between `friend_count` and `social_interactions`, which simple imputation destroys.

## 6. Common Mistakes or Failed Experiments
* **Global Imputation:** In earlier versions, imputing data before splitting caused data leakage. V6 fixes this by using `Pipeline` to impute within cross-validation folds.
* **Ignoring Outliers:** Standard scaling failed because "Whales" skewed the mean. Switching to `RobustScaler` (based on quartiles) stabilized the linear models in the stack.
* **Single Stage Regression:** Initial attempts to use just XGBoost Regressor resulted in many negative predictions or under-prediction of high spenders.

## 7. Domain Interpretation of Results
* **Business Impact:** Accurately predicting LTV (Lifetime Value) allow game developers to:
    1.  **Target Whales:** Offer exclusive VIP packages to high-predicted spenders.
    2.  **Convert F2P:** Identify players with high conversion probability (Stage 1) but low predicted spending (Stage 2) and target them with "Starter Pack" discounts.
* **Segmentation:** The `segment` and `is_premium_member` features were critical, suggesting that monetization strategies should be tailored to specific player tiers rather than a one-size-fits-all approach.

---
*Created for CPE342 by [Chonathun Cheepsathis/66070507201]*
