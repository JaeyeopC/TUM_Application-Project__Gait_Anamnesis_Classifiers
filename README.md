
# Gait Event Detection & Pain Anamnesis Classification (TUM Application Project)

## Project Overview
This project implements and compares machine learning models that:
- Detect **gait events** from wearable **6‑axis IMU sensor time‑series data** (accelerometer + gyroscope).
- Predict **pain anamnesis outcomes** (binary and ordinal) using **biomechanical features derived from gait**.

Two modeling approaches were developed and compared:
- **LSTM (sequence modeling)** for time‑series gait event detection.
- **XGBoost (lagged feature approach)** for structured time‑series learning.

The repository separates final artifacts (reports, notebooks, trained models, and experiment results) into different branches.

---

## Repository Branches
- **LSTM_final_models**
  - LSTM-based gait event classification
  - Neural network models for pain anamnesis prediction (binary and multi-task)
  - Includes notebooks and data files

- **XGBoost_final_models**
  - Final XGBoost models for gait event detection and pain anamnesis
  - Optuna hyperparameter tuning
  - Saved models and evaluation reports

- **XGBoost_Gait_Experiment**
  - Experimental branch for feature engineering and tuning experiments for gait event detection

---

## Project Objectives

### 1. Gait Event Detection
Classify gait events from **IMU sensor signals embedded in shoe insoles**.

Target classes include:
- Heel Strike
- Foot Flat
- Heel Off
- Toe Off

### 2. Pain Anamnesis Classification
Predict patient pain questionnaire results using **7 biomechanical features extracted from gait**.

Targets include:
- **18 binary pain indicators (CLP)**
- **24 ordinal pain severity variables (0–5 scale)**

---

## Datasets

### Gait Event Dataset
- **2,324 collection IDs**
- **508,446 rows**
- **8 columns** including:
  - collection_id
  - timestamps
  - 6‑axis IMU sensor measurements
  - gait phase labels

Class distribution:

| Event | Count |
|------|------|
| No Event | 1,806 |
| Heel Strike | 44,230 |
| Foot Flat | 175,126 |
| Heel Off | 97,929 |
| Toe Off | 189,335 |

The dataset is **highly imbalanced**.

### Pain Anamnesis Dataset
- **2,603 patient records**
- **7 input features**
- **42 targets**
  - 18 binary
  - 24 ordinal (0–5 scale)

637 records contain missing values.

---

## Feature Engineering

### LSTM Pipeline
- Continuous IMU streams segmented into **sequence windows**
- Example window length: **30 timesteps**
- Inputs:
  - 3-axis gyroscope
  - 3-axis accelerometer
- **Min‑Max normalization**
- Sequence classification using the final timestep label

### XGBoost Pipeline
Time‑series data transformed into supervised learning features.

Key idea:

event(t) = function(sensor(t), sensor(t−1), ... sensor(t−19))

Steps:
1. Remove missing values
2. Remove **No Event** samples
3. Convert labels to 4‑class classification
4. Determine lag size using **Autocorrelation Function (ACF)**
5. Final lag size: **19**

Feature count:

6 sensors × 19 lags ≈ **114 lagged features**

---

## Models

### LSTM Gait Event Classifier

Architecture:
- 3 LSTM layers
- Hidden dimension: 128
- Dropout: 0.5
- Fully connected output layer

Training setup:
- Train / Validation / Test split: **70 / 15 / 15**
- Batch size: **128**
- Learning rate: **0.001**
- Optimizer: **Adam**
- Epochs: **10**

---

### XGBoost Gait Event Model

Model:

XGBClassifier(
objective="multi:softmax",
eval_metric="mlogloss"
)

Hyperparameter tuning:
- **Optuna TPE sampler**
- **50 trials**
- Best validation accuracy ≈ **0.914**

Key parameters tuned:
- max_depth
- learning_rate
- n_estimators
- subsample
- colsample_bytree
- gamma
- min_child_weight

---

### Neural Networks for Pain Anamnesis

Two architectures were tested.

#### Binary‑Only Neural Network
- 7 hidden layers
- BatchNorm + LeakyReLU + Dropout
- Optimizer: AdamW
- Epochs: 50

#### Multi‑Task Neural Network
Shared trunk + multiple output heads.

Components:
- Shared layers (64 → 32 → 32)
- Binary classification head
- Ordinal regression heads using **CORAL method**

Loss function:
- Binary Cross‑Entropy
- CORAL ordinal loss

---

## Results

### Gait Event Detection

| Model | Accuracy | Notes |
|------|------|------|
| LSTM | ~0.91 | Rare classes difficult |
| XGBoost | ~0.921 | Best overall performance |

Heel Strike remained the hardest class due to imbalance.

### Pain Anamnesis Prediction

| Model | Binary Accuracy | Additional Metric |
|------|------|------|
| Binary NN | ~0.78 | — |
| Multi‑Task NN | ~0.80 | MAE ≈ 1.4 |
| XGBoost | mean accuracy ≈ 0.668 | mean F1 ≈ 0.285 |

Performance varied significantly between targets.

---

## Limitations

1. **Class imbalance**
   - Rare gait events reduce recall.

2. **Preprocessing artifacts not saved**
   - Scalers and encoders were not stored with the models.

3. **Data split differences**
   - XGBoost used group‑based splitting.
   - LSTM used random splits, which may risk data leakage.

4. **Limited features**
   - Only 7 biomechanical inputs for predicting 42 pain targets.

---

## Future Work

Recommended improvements:

1. Save **scalers and encoders** with model checkpoints.
2. Improve **rare class detection** using resampling or weighted loss.
3. Use **group‑based splitting** consistently.
4. Expand biomechanical features for pain prediction.

---

## Repository

GitHub:
https://github.com/JaeyeopC/TUM_Application-Project__Gait_Anamnesis_Classifiers

This repository contains the notebooks, datasets, and trained models used in the TUM Application Project.
