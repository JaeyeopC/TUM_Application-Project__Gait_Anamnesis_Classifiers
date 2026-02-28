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

## Exploratory Data Analysis (EDA)

Before training the models, exploratory data analysis (EDA) was conducted to understand the statistical properties of the datasets and identify patterns relevant for modeling.

### Gait Event Dataset

#### Dataset Structure

- Total rows: **508,446**
- Collection IDs: **2,324**

Each **collection ID represents one walking session**, meaning multiple rows together form a continuous IMU time-series sequence recorded from the same participant. 

Each observation includes:

- 3-axis accelerometer signals
- 3-axis gyroscope signals
- timestamp
- foot side (left / right)
- gait event label

---

#### Class Distribution

The dataset contains five gait event classes:

| Label | Event |
|------|------|
| 0 | No Event |
| 1 | Heel Strike |
| 2 | Foot Flat |
| 3 | Heel Off |
| 4 | Toe Off |

Event counts in the dataset:

| Event | Count |
|------|------|
| No Event | 1,806 |
| Heel Strike | 44,230 |
| Foot Flat | 175,126 |
| Heel Off | 97,929 |
| Toe Off | 189,335 |

Key observations:

- **Foot Flat** and **Toe Off** dominate the dataset.
- **Heel Strike** is relatively rare.
- **No Event** appears very rarely compared to other phases.

This class imbalance explains why some models showed weaker performance on the **Heel Strike** class.

---

#### Temporal Dependency Analysis

Because the dataset consists of time-series sensor signals, the **Autocorrelation Function (ACF)** was used to analyze temporal dependencies.

Results:

- Average optimal lag ≈ **19 timesteps**

Interpretation:

- Sensor measurements from the previous **~19 timestamps significantly influence the current gait phase**.

Based on this result:

- **Lag features were created for the XGBoost model**
- With **6 IMU channels × 19 lags = 114 lag features**

After lag generation:

- Dataset size became approximately **464,590 rows × 116 features**

---

### Pain Anamnesis Dataset

#### Dataset Structure

- Total records: **2,603 patients**
- Input features: **7 biomechanical gait features**
- Targets: **42 pain variables**

The pain anamnesis dataset contains 42 target variables:

- 18 binary targets → pain presence (0 / 1)
- 24 ordinal targets → pain severity (0–5)

To increase the statistical association between features and target variables, different target processing strategies were applied depending on the modeling approach.

For the neural network model, the original target structure was preserved. The model was designed as a multi-task learning framework that simultaneously predicts binary pain presence and ordinal pain severity.

For the XGBoost approach, ordinal targets were converted into binary variables (pain vs. no pain). This transformation simplifies the prediction task and helps increase the effective association between the biomechanical gait features and the target variables.

---

#### Missing Values

Among the **2,603 records**:

- **637 samples contain missing values**
  - either in the pain targets or in the shoe size feature

These records were handled during preprocessing depending on the modeling experiment.

---

#### Feature Distribution

Descriptive statistics were computed for all biomechanical features.

Key observations:

- Most biomechanical features exhibit approximately **Gaussian-like distributions**, indicating that they can be treated as continuous variables.
- Although **shoe size** is technically discrete, it spans a sufficiently wide range and was therefore handled as a continuous variable in the analysis.
- Since the feature ranges differ substantially, **feature normalization** was applied prior to model training.

Therefore:

- **Standardization or MinMax scaling** was applied before training machine learning models.

---

#### Feature–Target Correlation Analysis

Several statistical methods were used to examine relationships between gait features and pain outcomes:
To investigate the statistical relationship between biomechanical gait features and pain targets, several dependency measures were used:

- **Point-biserial correlation**  
  Evaluates the relationship between continuous input features and binary pain indicators.

- **Distance correlation**  
  Captures both linear and nonlinear dependencies between variables.

- **Mutual information**  
  Measures the amount of shared information between features and targets without assuming any specific functional relationship.

These analyses indicated that individual feature–target relationships were generally weak, suggesting that pain outcomes are influenced by complex interactions among multiple gait features rather than a single dominant variable.

Results:

- Most feature–target correlations were **weak (< 0.3)**.
- No single biomechanical feature strongly predicts pain.

Interpretation:

- Pain prediction likely depends on **complex interactions between multiple gait features** rather than a single variable.

---

#### Target Dependency Analysis

Relationships between pain variables were analyzed using **Cramér’s V**.

Results:

- Some pairs of pain variables showed **moderate associations (Cramér’s V > 0.4)**.

Interpretation:

- Certain pain regions tend to **co-occur**, suggesting shared biomechanical or physiological causes.

---

#### Key EDA Insights

The exploratory analysis provided several insights that influenced model design:

- The **gait dataset shows strong temporal dependency**, motivating lag features and sequence models.
- The **gait event classes are imbalanced**, especially for Heel Strike.
- The **pain dataset shows weak individual correlations**, suggesting the need for models capable of capturing nonlinear interactions.
- Some **pain targets are correlated**, indicating potential multi-task learning benefits.

These observations motivated the use of both **XGBoost models (with engineered features)** and **neural network models (for complex feature interactions)** in the project.

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
