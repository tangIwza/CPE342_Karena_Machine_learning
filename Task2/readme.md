# CPE342 Karena Task 2 (V4): Player Skill Rating Prediction

This repository contains the solution for **CPE342 Task 2 (Karena)**. The objective of this task is to predict a continuous variable (Task 2), which represents a player's **Skill Rating** or **Performance Score**, based on their gameplay statistics and account history. The solution implements a **Stacking Regressor** ensemble, optimized via **Optuna** and enhanced with **Robust Preprocessing**.

---

## 1. EDA Findings (Exploratory Data Analysis)
Exploratory analysis of the dataset revealed several key characteristics influencing the model design:

* **Correlation with Activity:** High correlations were observed between the target variable and engagement metrics like `total_playtime_hours` and `sessions_per_week`.
* **Outliers:** The dataset contains significant outliers in numerical features (e.g., some players have extreme playtime or spending). Standard scaling was insufficient; **Robust Scaling** was required.
* **Missing Data:** Several features contained missing values that were not random. Simple mean imputation reduced variance, so **Iterative Imputer (MICE)** was identified as a better approach to preserve feature relationships.

## 2. Data Preprocessing
A comprehensive preprocessing pipeline was built to clean and enrich the data:

* **Advanced Imputation:** Utilized `IterativeImputer` (MICE) to model missing values as a function of other features, providing more accurate estimates than simple mean/median filling.
* **Feature Engineering:**
    * **Ratio Features:** Created performance indicators such as `playtime_per_session` (Efficiency) and `spending_per_day` (Monetization intensity) to normalize data across different account ages.
    * **Log Transformation:** Applied `np.log1p` to skewed distributions like `historical_spending` and `total_playtime_hours` to linearize relationships for the models.
* **Scaling:** Implemented `RobustScaler` to scale features using statistics that are robust to outliers (Interquartile Range).

## 3. Model Design
The solution employs a **Stacking Ensemble** architecture to minimize prediction error (RMSE).

* **Base Learners (Level 0):**
    1.  **LightGBM Regressor:** Optimized for speed and handling large datasets.
    2.  **XGBoost Regressor:** Tuned to capture complex non-linear patterns.
    3.  **CatBoost Regressor:** Excellent at handling categorical nuances and reducing overfitting.
    4.  **Gradient Boosting Regressor:** Added diversity to the ensemble.
    5.  **Linear Models (Lasso/ElasticNet):** Included to capture linear trends and extrapolate better than tree-based models.
* **Meta Learner (Level 1):** `Ridge` (Linear Regression with L2 regularization) was used to combine the predictions of base learners effectively without overfitting.
* **Hyperparameter Tuning:** All key base models were tuned using **Optuna** to minimize RMSE using 3-fold Cross-Validation.

## 4. Evaluation & Results
The model was evaluated using **5-Fold Cross-Validation** to ensure stability and prevent overfitting.

* **Metric:** Root Mean Squared Error (RMSE).
* **Performance:**
    * **Single Models:** XGBoost and LightGBM performed well individually but struggled with edge cases.
    * **Stacking:** The full stacking ensemble consistently outperformed the best single model by reducing variance.
    * **Final Output:** The model predicts a continuous score, which is then post-processed (e.g., clipped to valid ranges if necessary).

## 5. Insights Gained from Model Behavior
* **Linear + Non-Linear Synergy:** Including linear models (`Lasso`, `ElasticNet`) in the stacking ensemble significantly improved performance. This suggests that while player skill is complex (Tree models), there are underlying linear progression factors (Account Age, Level) that linear models capture best.
* **Imputation Quality:** Switching from `SimpleImputer` to `IterativeImputer` yielded a noticeable improvement in CV scores, confirming that the "missingness" in player data holds information that shouldn't be discarded.

## 6. Common Mistakes or Failed Experiments
* **Overfitting with Deep Trees:** Initial experiments with very deep trees (`max_depth > 10`) led to overfitting. Restricting depth and increasing `min_child_samples` provided better generalization.
* **Ignoring Outliers:** Early versions using `StandardScaler` resulted in poor performance for "average" players because the scaler was skewed by "hardcore" players. `RobustScaler` fixed this.
* **Direct Prediction:** Directly predicting the target without Log Transformation of inputs resulted in heteroscedastic errors (higher error for high-value players).

## 7. Domain Interpretation of Results
* **Business Impact:** Accurate skill/performance prediction enables:
    * **Better Matchmaking:** Grouping players of similar true skill levels to improve user retention.
    * **Smurf Detection:** Identifying accounts that perform significantly higher than their "New Account" status would suggest.
* **Game Design:** The high importance of `playtime_per_session` suggests that "grinding" is a major component of the skill rating in this specific game context.

---
*Created for CPE342 by [Chonathun Cheepsathis/66070507201]*
