# CPE342 Karena Task 5: Anomaly Detection (Unsupervised)

This repository contains the solution for **CPE342 Task 5**, which focuses on detecting suspicious player behavior (anomalies) such as hacked accounts, botting, or fraud without labeled training data. The solution employs a **Rank-Averaged Ensemble** of four distinct unsupervised algorithms, including Deep Learning (Autoencoder).

---

## 1. EDA Findings (Exploratory Data Analysis)
Exploration of the user login and activity data revealed specific patterns utilized for detection:

* **Sequential Behavior:** The dataset contains sequential snapshots (suffixes `_1`, `_2`, `_3`, `_4`) of user metrics like login counts and coordinates. Sudden spikes in these sequences often indicate anomalies.
* **Geographical Inconsistencies:** By analyzing `login_lat` and `login_lon`, we found that physically impossible travel speeds (teleportation) are strong indicators of suspicious activity.
* **Outlier Distribution:** The data contains heavy-tailed distributions (extreme outliers), necessitating robust scaling methods rather than standard normalization.

## 2. Data Preprocessing
A comprehensive pipeline was designed to transform raw sequential data into meaningful anomaly indicators:

* **Feature Engineering:**
    * **Trend Features:** Calculated `slope`, `std` (standard deviation), `cv` (coefficient of variation), and `max_jump` across time steps to capture abrupt behavioral changes.
    * **Geo-Spatial Features:** Calculated **Haversine distances** between consecutive logins to detect "impossible travel" scenarios (`login_dist_sum`, `login_dist_max`).
* **Cleaning & Scaling:**
    * **Imputation:** Missing values were filled with the median to maintain robustness.
    * **Winsorization:** Applied 1% - 99% clipping to limit the influence of extreme noise before training.
    * **Robust Scaler:** Used `RobustScaler` (based on IQR) instead of StandardScaler to ensure the models aren't skewed by the very outliers we intend to detect.

## 3. Model Design
Since no ground truth labels were provided, a **Consensus Ensemble** approach was used to maximize reliability.

* **Ensemble Components (4 Models):**
    1.  **Isolation Forest (IF):** Detects anomalies by randomly partitioning the feature space. Effective for global outliers.
    2.  **One-Class SVM (OCSVM):** Uses an RBF kernel to define a boundary around "normal" data density.
    3.  **Local Outlier Factor (LOF):** Measures the local density deviation of a data point compared to its neighbors (Novelty detection mode).
    4.  **Autoencoder (AE):** A PyTorch-based Neural Network trained to compress and reconstruct normal data. High **Reconstruction Error (MSE)** indicates an anomaly.

* **Aggregation Strategy:**
    * **Rank Averaging:** Scores from all 4 models were converted to ranks and averaged. This normalizes the score scales and prioritizes points that *all* models agree are suspicious.
    * **Calibration:** Used **Isotonic Regression** (with pseudo-labels from the top 5% vs. bottom 75%) to convert raw ensemble scores into interpretable probabilities (`p_anom`).

## 4. Evaluation & Results
Evaluation was performed using internal consistency metrics due to the lack of labels.

* **Metrics:**
    * **Pseudo-F3 Score:** Evaluated consistency between the ensemble consensus and individual predictions. High F3 scores (>0.90) at target rates 5-10% indicate stable agreement.
    * **Silhouette Score:** Used to measure how well the detected anomalies separate from the normal cluster.
* **Sensitivity Analysis:** Generated submissions for multiple anomaly rates (3% to 10%) to allow flexibility in the final threshold selection based on the leaderboard feedback.

## 5. Insights Gained from Model Behavior
* **Reconstruction vs. Density:** The Autoencoder was particularly good at finding complex pattern violations (e.g., a user who logs in frequently but acts strangely), while Isolation Forest was better at catching extreme value outliers (e.g., mass item sales). combining them covered both "Point Anomalies" and "Contextual Anomalies".
* **Rank Stability:** Raw scores from OCSVM and AE are on completely different scales. Rank averaging proved superior to min-max scaling for ensembling.

## 6. Common Mistakes or Failed Experiments
* **Standard Scaling:** Initial experiments with `StandardScaler` failed because the extreme outliers in the dataset shifted the mean/variance, making normal data look anomalous. Switching to `RobustScaler` fixed this.
* **Single Model Reliance:** Using only Isolation Forest resulted in a lower Silhouette score compared to the ensemble, suggesting that a single algorithm misses subtle fraud patterns.

## 7. Domain Interpretation of Results
* **Security Implications:** The model effectively flags users with:
    * **High Velocity:** Login locations changing faster than physical travel allows (Geo features).
    * **Bot-like Regularity:** Low variance in behavior (`ip_hash_entropy` or consistent sequences) combined with high volume.
* **Business Value:** By setting a threshold (e.g., top 5%), the game administrators can auto-ban high-probability offenders and manually review the borderline cases, optimizing the moderation workload.

---
*Created for CPE342 by [Thanakrit Rodngampring/66070507204]*
