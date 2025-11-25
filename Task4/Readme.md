# CPE342 Karena Task 4: Image Classification with Swin Transformer

This repository contains the solution for **CPE342 Task 4 (Karena)**. The project implements a **Swin Transformer (Small Patch 4 Window 7)** model using PyTorch and `timm` to classify game-related images. The training pipeline includes Stratified K-Fold cross-validation, heavy data augmentation, and class balancing strategies.

---

## 1. EDA Findings (Exploratory Data Analysis)
Before modeling, we explored the dataset and identified the following key characteristics:

* **Class Imbalance:** The distribution of classes in `train.csv` was found to be imbalanced. Some categories had significantly fewer images than others.
    * *Solution:* Implemented `WeightedRandomSampler` and `compute_class_weight` ('balanced') in the loss function to prevent the model from biasing towards majority classes.
* **File Extension Inconsistencies:** The dataset contained mixed file extensions (e.g., `.jpg`, `.png`, `.jpeg`).
    * *Solution:* Added a fallback logic in the `GameDataset` class to search for correct extensions automatically.
* **Image Quality:** Images vary in resolution and aspect ratio.

## 2. Data Preprocessing
To prepare the data for the Swin Transformer, the following steps were taken:

* **Resizing:** All images are resized to **224x224** pixels (Standard ImageNet size).
* **Normalization:** Normalized using ImageNet mean `[0.485, 0.456, 0.406]` and std `[0.229, 0.224, 0.225]`.
* **Data Augmentation (Train Set):** Used to reduce overfitting and improve generalization:
    * `RandomResizedCrop`: Scale 0.8-1.0 to handle different object scales.
    * `RandomHorizontalFlip`: For invariance to orientation.
    * `RandomRotation`: +/- 15 degrees.
    * `ColorJitter`: Slight adjustments to brightness, contrast, and saturation.
* **Validation/Test Set:** Only resizing and normalization were applied to maintain data integrity for evaluation.

## 3. Model Design
We selected a Transformer-based architecture over traditional CNNs for better feature extraction capabilities.

* **Architecture:** `swin_small_patch4_window7_224` (Pre-trained on ImageNet).
    * *Reasoning:* Swin Transformers use shifted windows to model both local and global dependencies effectively.
* **Loss Function:** `CrossEntropyLoss` with calculated **class weights** to penalize errors on minority classes more heavily.
* **Optimizer:** `AdamW` with learning rate `1e-4` and weight decay `1e-4`.
* **Scheduler:** `CosineAnnealingLR` to smoothly decay the learning rate over epochs.
* **Training Strategy:**
    * **5-Fold Stratified Cross-Validation:** Ensures reliability of results.
    * **Mixed Precision (AMP):** Used `GradScaler` to speed up training and reduce VRAM usage on the RTX 3050 GPU.
    * **Early Stopping:** Set to 2 epochs to prevent unnecessary computation if validation accuracy stalls.

## 4. Evaluation & Results
The model was evaluated using **Accuracy** across 5 folds.

* **Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU.
* **Training Time:** Approx. 45 minutes per epoch.
* **Performance:**
    * **Fold 1:** Validation Accuracy ~ **98.54%**
    * **Fold 2:** Validation Accuracy ~ **98.46%**
    * *Average:* The model consistently achieves >98% accuracy on the validation set.

> **Note:** Training converges very quickly, reaching ~96% accuracy within the second epoch.

## 5. Insights Gained from Model Behavior
* **Pre-training is Key:** Using the pre-trained weights from ImageNet allowed the model to converge extremely fast (less than 5 epochs) even on a custom dataset.
* **Robustness:** The small gap between Training Accuracy (~99%) and Validation Accuracy (~98.5%) indicates that the aggressive data augmentation and weight decay effectively prevented overfitting.
* **Transformer Efficiency:** Despite being a Transformer, the "Small" Swin variant fits well within 4GB VRAM (RTX 3050) when using a batch size of 32 and Mixed Precision.

## 6. Common Mistakes or Failed Experiments
* **Interrupting Training:** Training time on a laptop GPU is significant. A `KeyboardInterrupt` occurred during Fold 3, highlighting the need for checkpointing (which was implemented via `torch.save`).
* **File Path Issues:** Initially, some images failed to load due to case-sensitive extensions (`.JPG` vs `.jpg`). The dataset class was updated to handle this robustly using a loop check.
* **Dataloader Workers:** On Windows, setting `NUM_WORKERS > 0` often causes multiprocessing errors. We set `NUM_WORKERS = 0` to ensure stability.

## 7. Domain Interpretation of Results
* **Application:** This model is designed to classify assets or characters within the "Karena" game domain.
* **Business Impact:** With >98% accuracy, this model can be reliably used to automate asset tagging, verify uploaded content, or assist in game state analysis without manual intervention.
* **Scalability:** The pipeline allows for easy integration of new classes by simply updating the dataset and re-running the training script.

---
*Created for CPE342 by [Chonathun Cheepsathis/66070507201]*
