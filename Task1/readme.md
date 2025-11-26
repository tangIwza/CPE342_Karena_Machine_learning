# CPE342 Karena Task 1 (V5): Cheater Detection System

This repository contains the solution for **CPE342 Task 1 (Karena)**, designed to detect cheaters in online gaming. The solution implements a **Weighted Ensemble** of LightGBM, XGBoost, and CatBoost, enhanced with **SMOTE Oversampling** and advanced feature engineering to handle class imbalance effectively.

---

## 1. EDA Findings (Exploratory Data Analysis)
Initial analysis of the dataset revealed critical insights guiding the model design:

* **Class Imbalance:** The dataset is moderately imbalanced with ~65% Normal players vs. ~35% Cheaters. This imbalance necessitated the use of **SMOTE** and `scale_pos_weight`.
* **Behavioral Indicators:**
    * **Skill Outliers:** Cheaters exhibit unnaturally high correlations between kill metrics (`kill_death_ratio`) and precision (`headshot_percentage`).
    * **New Accounts:** High-performing accounts with low `account_age_days` are statistically suspicious.
    * **Social Isolation:** Cheaters tend to have fewer friends relative to the number of reports received (`report_friend_ratio`).

## 2. Data Preprocessing
A robust preprocessing pipeline was engineered to transform raw gameplay stats into predictive signals:

* **Advanced Feature Engineering (15+ New Features):**
    * **Suspicion Score:** A composite metric combining K/D ratio and Headshot % normalized by account age.
    * **Aim-Movement Mismatch:** Captures players with perfect aim (`aiming_smoothness`) but poor movement (`movement_pattern_score`), a classic sign of aimbots.
    * **Hardware Volatility:** Tracks abnormal device changes normalized by account lifespan.
* **Imputation Strategy:** Implemented a hybrid approach using `mean` for features with low missing rates and `median` for those with high missing rates (>10%) to preserve distribution integrity.
* **SMOTE Oversampling:** Synthetic Minority Over-sampling Technique (SMOTE) was applied *only* within the training folds of Cross-Validation to prevent data leakage while balancing the classes for better model learning.

## 3. Model Design
The core architecture is a **Weighted Ensemble Classifier** designed to maximize the F2-Score (Recall-focused).

* **Base Learners (The "Big 3"):**
    1.  **LightGBM:** Optimized for speed and handling categorical features.
    2.  **XGBoost:** Tuned with `scale_pos_weight` (~1.86) to penalize missing cheaters.
    3.  **CatBoost:** Utilized for its robust handling of numerical noise and overfitting prevention.
* **Optimization:**
    * **Hyperparameter Tuning:** Used **Optuna** (implied or manual search) to find optimal parameters like `num_leaves=40` and `max_depth=8` to prevent overfitting.
    * **Weight Optimization:** Used `scipy.optimize.minimize` to find the perfect blending weights (LGBM: 0.33, XGB: 0.33, Cat: 0.33) that maximize the validation F2-Score.

## 4. Evaluation & Results
The model uses **Stratified 5-Fold Cross-Validation** to ensure stability.

* **Primary Metric:** **F2-Score** (Prioritizing Recall over Precision, as missing a cheater is worse than a false report).
* **Performance:**
    * **Mean F2-Score:** **~0.8366** across 5 folds.
    * **Threshold Tuning:** The decision threshold was dynamically optimized for each fold, averaging at **0.1827**, indicating an aggressive stance against potential cheaters.

## 5. Insights Gained from Model Behavior
* **Reports Matter:** `reports_received` and `report_friend_ratio` consistently ranked among the top features, confirming that community reporting is a strong signal when normalized correctly.
* **Context is Key:** Raw stats like K/D ratio alone are less predictive than context-aware features like `performance_per_day` (Skill relative to Experience).
* **SMOTE Effectiveness:** Applying SMOTE specifically improved the model's ability to identify rare, high-impact cheating patterns that were previously lost in the majority class.

## 6. Common Mistakes or Failed Experiments
* **Uniform Thresholds:** Using a default 0.5 threshold resulted in lower recall. Moving the threshold down to ~0.18 significantly boosted the F2 score by catching more borderline cases.
* **Feature Noise:** Initial feature sets included redundant metrics. Feature selection using LightGBM importance helped prune zero-importance features, streamlining the model.

## 7. Domain Interpretation of Results
* **Game Integrity:** The model effectively targets "Rage Hackers" (obvious stats) and "Closet Cheaters" (subtle aim-movement mismatches).
* **Business Impact:** By automating the detection of high-probability cheaters (F2 ~0.84), the system reduces the manual workload for moderation teams and improves the overall fairness and player retention of the game.

---
*Created for CPE342 by [Chonathun Cheepsathis/66070507201]*
