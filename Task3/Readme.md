# CPE342 Karena Task 3 (V8): Hybrid Two-Stage Spending Prediction

This repository contains the solution for **CPE342 Task 3 (Karena)**, aimed at predicting the 30-day future spending of players. This version (V8) implements a **Hybrid Two-Stage Architecture** combining a **Voting Classifier** (for purchase probability) and a **Stacked Regressor** (for amount estimation), enhanced with Unsupervised Clustering features.

---

## 1. EDA Findings (Exploratory Data Analysis)
Exploration of the dataset highlighted specific challenges regarding player spending behavior:

* **Zero-Inflation:** A vast majority of players have `spending_30d = 0`. A direct regression model often fails to capture this, predicting small non-zero values for everyone.
* **Behavioral Clusters:** Players don't behave uniformly. There are distinct groups (e.g., high-time/low-spend, low-time/high-spend). This led to the introduction of **K-Means Clustering** to create a `cluster_group` feature.
* **Skewness:** Features like `historical_spending` and `total_transactions` span multiple orders of magnitude, requiring Log Transformation to stabilize model training.

## 2. Data Preprocessing
The preprocessing pipeline was upgraded to include both supervised and unsupervised techniques:

* **Unsupervised Feature Engineering (Clustering):** Applied **K-Means (k=7)** on log-transformed spending and playtime data to categorize players into distinct behavioral profiles (`cluster_group`).
* **Interaction Features:** Created `engagement_score` (Playtime × Friend Count) to capture the synergy between social activity and game engagement.
* **Encoding:** Switched to **TargetEncoder** for categorical variables (like `primary_game`, `cluster_group`), which is more effective for high-cardinality features in regression tasks than One-Hot Encoding.
* **Scaling:** Continued using `RobustScaler` to handle the extreme outliers typical of "Whale" players.

## 3. Model Design
The solution uses a **Hurdle Model** architecture to separate the "Decision to Buy" from the "Amount Spent":

### Stage 1: Probability (Will they spend?)
* **Objective:** Binary Classification (`spending_30d > 0`).
* **Model:** **VotingClassifier (Soft Voting)** combining:
    * **LGBMClassifier:** Tuned with `class_weight='balanced'`.
    * **XGBClassifier:** Tuned with `scale_pos_weight`.
* **Why:** Voting smoothens the decision boundary and reduces false negatives.

### Stage 2: Amount (If spending, how much?)
* **Objective:** Regression on `log1p(spending_30d)`.
* **Model:** **StackingRegressor** with a diverse set of learners:
    * **Base Layers:** `LGBMRegressor`, `XGBRegressor`, `CatBoostRegressor`, `GradientBoostingRegressor`, plus Linear Models (`LassoCV`, `ElasticNetCV`) to prevent overfitting.
    * **Meta Learner:** `KernelRidge` (Non-linear combination).
* **Why:** Stacking combines the pattern recognition of Trees with the extrapolation capabilities of Linear/Kernel models.

## 4. Evaluation & Results
Models were optimized using **Optuna** with Stratified K-Fold Cross-Validation.

* **Classification Performance:** The Voting Classifier achieved ~0.78 AUC, effectively filtering out non-spenders.
* **Regression Performance:** The Stacking Regressor achieved an RMSE of ~0.2221 (Log scale) on the spender subset.
* **Final Prediction Logic:**
    $$\text{Final Prediction} = P(\text{Buy}) \times \text{Expm1}(\text{Predicted Log Amount})$$
    This effectively dampens the predicted amount for users with low purchase probability.

## 5. Insights Gained from Model Behavior
* **The "Hybrid" Effect:** Multiplying the regression output by the classification probability acts as a gate. It pushes the prediction for unlikely spenders towards true zero, drastically reducing the error compared to using regression alone.
* **Cluster Importance:** The `cluster_group` feature became a strong predictor, as it effectively encoded the "player persona" (e.g., F2P Grinder vs. P2W Whale) into a single categorical variable.
* **Diversity in Stacking:** Adding `Lasso` and `ElasticNet` to the stack stabilized predictions for extreme values where Tree-based models often plateau.

## 6. Common Mistakes or Failed Experiments
* **Single Stage Regression:** Trying to predict `spending_30d` directly with one model resulted in poor performance due to the imbalance of zeros.
* **Over-reliance on Trees:** In previous versions, using only boosting trees (XGB/LGBM) caused overfitting. Adding linear models and `KernelRidge` in the stacking layer improved generalization on the test set.
* **Leakage in Encoding:** Target Encoding can cause leakage if not handled correctly. We mitigated this by using `Pipeline` to fit the encoder only within the cross-validation folds.

## 7. Domain Interpretation of Results
* **Monetization Strategy:** The model effectively identifies "Potential Spenders" (High Prob, Low Amount) vs "Whales" (High Prob, High Amount).
    * *Action:* Send "Starter Pack" offers to Potential Spenders.
    * *Action:* Send "VIP Exclusive" offers to Whales.
* **Social Influence:** The `engagement_score` (derived from friends and playtime) confirms that socially active players are monetarily valuable, suggesting that social features (guilds, chats) drive revenue.

---
*Created for CPE342 by [Chonathun Cheepsathis/66070507201]*
