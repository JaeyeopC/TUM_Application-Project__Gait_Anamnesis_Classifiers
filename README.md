# Gait Event Detection & Pain Anamnesis Classification

## Project Overview

This project was conducted as part of the TUM Application Project.  
It investigates machine learning approaches for two related healthcare tasks:

1. **Gait Event Detection** using IMU sensor time-series data  
2. **Pain Anamnesis Classification** using biomechanical gait-derived features  

Two modeling strategies were implemented and compared:

- **XGBoost (tree-based gradient boosting)**
- **LSTM / Deep Neural Networks**

The detailed experimental results are documented in the final report. 

---

## Data 

### Gait Event Dataset

Before building the models, an exploratory analysis was conducted to understand the structure and characteristics of the IMU sensor data.

**Dataset structure**

- Total rows: **508,446**
- Collection IDs: **2,324**
- Each **collection ID corresponds to one walking session**, meaning multiple rows form a continuous time-series sequence. 

Each observation contains:

- 3-axis accelerometer data
- 3-axis gyroscope data
- timestamp
- foot side (left/right)
- step information
- gait event label

**Class distribution**

The dataset contains five gait event labels:

- 0 — No Event  
- 1 — Heel Strike  
- 2 — Foot Flat  
- 3 — Heel Off  
- 4 — Toe Off  

The distribution is highly imbalanced.  
Foot Flat and Toe Off appear most frequently, while Heel Strike is relatively rare.  
This imbalance later influenced model performance, particularly for the Heel Strike class.

**Temporal characteristics**

Because the dataset consists of time-series sensor data, temporal dependency between observations was analyzed using the **Autocorrelation Function (ACF)**. The average cutoff lag across collection IDs was approximately **19 time steps**, indicating that recent sensor history strongly affects the current gait phase.  This finding motivated the creation of **lag-based features** for the XGBoost model. 


---

### Pain Anamnesis Dataset

The pain anamnesis dataset contains **patient-reported pain information together with gait-derived biomechanical features**.

**Dataset structure**

- Total records: **2,603 patients**
- Input features: **7 biomechanical gait features**
- Targets: **42 pain-related variables**

Targets include:

- **18 binary variables**
  - indicate presence of localized pain

- **24 ordinal variables**
  - represent pain severity on a **0–5 scale**

This allows modeling both **pain presence and pain severity** simultaneously. :contentReference[oaicite:2]{index=2}

**Feature distributions**

Most biomechanical features show approximately **Gaussian-like distributions**, while **shoe size** is a discrete variable.

Examples of features include:

- lateral deviation during walking
- static standing deviation
- heel strike timing
- shoe size

**Missing values**

Among the 2,603 records, **637 samples contain missing values** in either the target variables or the shoe size feature.  
These missing values were handled during preprocessing depending on the experiment setup. :contentReference[oaicite:3]{index=3}


**Feature–target relationships**

Correlation analysis between gait features and pain outcomes showed that:

- Individual features have **low pairwise correlations (< 0.3)** with pain variables.
- Pain prediction likely depends on **complex interactions between multiple features** rather than a single dominant variable.

This observation motivated the use of **machine learning models capable of capturing nonlinear relationships**, such as XGBoost and neural networks.
---

## 2. Pain Anamnesis Dataset

- **Total patients**: 2,603  
- **Input features**: 7 biomechanical gait features  
  - e.g., lateral deviations, heel strike timing, shoe size  

### Target Variables (42 total)

- **18 Binary variables** → localized pain presence (0/1)  
- **24 Ordinal variables** → pain severity (0–5 scale)  

The dataset allows simultaneous prediction of pain presence and severity.

---

# Task: Gait Event Detection

Goal: Classify gait phase from IMU time-series signals.

---

## XGBoost

### Method

Instead of using raw sequences, we transformed time-series data into tabular form using **lag-based feature engineering**.

- Autocorrelation (ACF) used to determine optimal lag size  
- Past sensor values added as lag features  
- MinMax scaling applied  
- GroupShuffleSplit used to avoid leakage across collection IDs  

### Hyperparameter Optimization

- **Optuna** used for automated hyperparameter search  
  - max_depth  
  - learning_rate  
  - n_estimators  
  - subsample  
  - colsample_bytree  

### Model Interpretation

- **SHAP analysis** used to interpret feature importance  
- Identified which sensor axes and lag steps contributed most to event prediction  

---

## LSTM

### Method

Raw IMU sequences were directly used as input to an LSTM network.

- Sliding window segmentation (2–3 seconds per window)  
- MinMax normalization  
- Sequence labeling approach  

### Architecture

- 3 stacked LSTM layers  
- 128 hidden units each  
- Dropout for regularization  
- Fully connected softmax output  

### Training

- Optimizer: Adam (lr=0.001)  
- 70/15/15 train-validation-test split  
- Early stopping  

### Result

- ~91% overall test accuracy  
- Strong performance on frequent classes (Foot Flat, Toe Off)  
- Lower recall for rare class (Heel Strike)

---

# Task: Pain Anamnesis Classification

Goal: Predict patient pain profile using gait-derived features.

---

## XGBoost

### Method

- Structured 7 biomechanical features used as input  
- Each pain target modeled independently  
- Ordinal variables optionally converted to binary for analysis  
- Standardization applied  

### Hyperparameter Optimization

- **Optuna** used for tuning each target model  
- Threshold optimization applied for better class balance  

### Model Interpretation

- **SHAP summary plots and dependence plots** used  
- Identified biomechanical features influencing specific pain regions  
- Provided interpretable relationships between gait deviations and pain  

---

## LSTM / Multi-Output Neural Network

Since pain inputs are structured (not sequential), a **multi-output neural network** was used.

### Input

- 7 biomechanical gait features  

### Architecture

- Shared dense trunk (64 → 32 → 32)  
- Two output branches:
  - 18 binary outputs (sigmoid)
  - 24 ordinal outputs using **CORAL** method  

### Training

- Binary Cross Entropy for binary tasks  
- CORAL-based ordinal loss  
- Multi-task loss balancing  
- Adam optimizer  

### Results

- ~80% multi-label binary accuracy  
- ~1.4 MAE for ordinal pain severity  
- Multi-task learning improved consistency and performance  

---

# Summary

- **LSTM** performs strongly for sequential gait event detection  
- **XGBoost + lag features** provides competitive performance with high interpretability  
- **Optuna** improves model performance through automated tuning  
- **SHAP** enables clinical interpretability of gait–pain relationships  
- Multi-task neural networks effectively handle mixed binary + ordinal outputs  

This project demonstrates how wearable gait data can be leveraged to build automated, interpretable healthcare prediction systems.
