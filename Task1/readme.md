# CPE342 Karena Task 1 (V7): Cheater Detection System

This repository contains the solution for **CPE342 Task 1 (Karena)**, focusing on detecting cheaters in an online game environment. The solution employs a **Stacking Ensemble** approach, combining XGBoost, LightGBM, and CatBoost, optimized via Optuna and enhanced with advanced feature engineering and anomaly detection.

---

## 1. EDA Findings (Exploratory Data Analysis)
Exploratory analysis revealed critical insights into the player behavior dataset:

* **Class Imbalance:** The dataset is imbalanced with significantly more "Normal" players (63,619) compared to "Cheaters" (34,129). This necessitated the use of class weighting strategies (`scale_pos_weight`) rather than standard sampling.
* **Skewed Distributions:** Several numerical features such as `survival_time_avg`, `damage_per_round`, and `account_age_days` showed highly skewed distributions, indicating the need for Log Transformation.
* **Correlation & Noise:** Correlation analysis (Pearson) was performed to identify feature relationships. Certain features like `friend_network_size` and `device_changes_count` were identified as low-signal noise and removed to reduce model complexity.

## ⚙️ 2. Data Preprocessing
To prepare the data for the ensemble model, a robust preprocessing pipeline was implemented:

* **Advanced Feature Engineering:** created context-aware features to capture cheating behavior better:
    * `report_per_kill`: Normalizes reports by kill count to distinguish skilled players from cheaters.
    * `sus_score`: Combines headshot percentage and win rate relative to account age.
    * `Aim_to_Brain_Ratio`: Contrasts raw mechanical aim against game knowledge/utility usage.
* **Anomaly Detection:** Integrated **Isolation Forest** (contamination=0.05) to generate an `anomaly_score` feature, effectively flagging outlier behaviors that statistically deviate from the norm.
* **Transformations:**
    * **Imputation:** Missing values were filled using the median strategy.
    * **Log Transform:** Applied `np.log1p` to skewed columns to normalize distributions.
    * **Scaling:** Used `StandardScaler` to standardize features for the stacking meta-learner.

## 3. Model Design
The solution utilizes a **Stacking Classifier** to leverage the strengths of multiple gradient boosting frameworks.

* **Base Learners:**
    1.  **XGBoost:** Tuned with `scale_pos_weight`.
    2.  **LightGBM:** Tuned with `class_weight='balanced'`.
    3.  **CatBoost:** Tuned with `auto_class_weights='Balanced'`.
* **Meta Learner:** `LogisticRegression` was used to blend the predictions of the base learners.
* **Hyperparameter Tuning:** utilized **Optuna** to maximize **ROC AUC** for all base models, running separate studies to find the optimal architecture (depth, learning rate, regularization) for each.

## 4. Evaluation & Results
The model was evaluated using **Stratified 5-Fold Cross-Validation** to ensure reliability.

* **Model Performance (AUC):**
    * XGBoost Best AUC: ~0.8531
    * LightGBM Best AUC: ~0.8540
    * CatBoost Best AUC: ~0.8543
* **Threshold Optimization:** Instead of using the default 0.5 threshold, the decision boundary was optimized to maximize the **F2 Score** (prioritizing Recall).
    * **Optimal Threshold:** 0.1050
    * **Max F2 Score:** **0.8160**

## 5. Insights Gained from Model Behavior
* **Recall Priority:** Optimization for the F2 score highlights that in cheat detection, missing a cheater (False Negative) is more detrimental than flagging a normal player for review (False Positive). The low threshold (0.1050) reflects this aggressive detection strategy.
* **Ensemble Power:** Stacking consistently outperformed individual models by smoothing out the variance and bias of single algorithms.
* **Behavioral Features:** Engineering features that combine "Skill" (Aim/Kills) with "Game Sense" (Map knowledge/Utility) proved crucial. Cheaters often have high mechanical stats but low game sense stats (high `Aim_to_Brain_Ratio`).

## 6. Common Mistakes or Failed Experiments
* **Avoided SMOTE:** Explicitly chose *not* to use SMOTE for oversampling. Previous experiments likely showed that synthesizing fake data points introduced noise, whereas using **Class Weights** provided a more stable signal for gradient boosting trees.
* **Feature Noise:** Initially included `friend_network_size` and `buy_decision_score`, but dropping them (as seen in the `noise_cols` list) likely improved model convergence and reduced overfitting.

## 7. Domain Interpretation of Results
* **Business Impact:** The high F2 Score (0.8160) indicates the model is highly effective at flagging potential cheaters for the moderation team. This reduces the manual workload of reviewing random reports and allows the team to focus on high-probability cases.
* **Game Integrity:** By effectively using `reaction_time_ms` and `headshot_percentage` in context (via `suspicious_aim`), the model distinguishes between "Pro Players" (natural high skill) and "Aimbots" (unnatural perfection), ensuring a fairer gaming environment.

---
*Created for CPE342 by [Chonathun Cheepsathis/66070507201]*
