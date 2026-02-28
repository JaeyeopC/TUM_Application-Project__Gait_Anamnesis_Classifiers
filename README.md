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

- **Total records:** 2,603 patients  
- **Input features:** 7 biomechanical gait features  
- **Targets:** 42 pain variables  

Each record corresponds to one patient and contains biomechanical features derived from gait analysis using IMU sensors embedded in instrumented insoles.

#### Feature Variables

The dataset contains **seven biomechanical features** describing characteristics of a patient's walking pattern and posture:

1. **Left movement deviation**  
   Lateral deviation of the movement trajectory on the left side during walking.

2. **Right movement deviation**  
   Lateral deviation of the movement trajectory on the right side during walking.

3. **Resting deviation**  
   Postural deviation measured while the patient is standing still.

4. **Average step length (left)**  
   Average step length of the left foot during walking.

5. **Average step length (right)**  
   Average step length of the right foot during walking.

6. **Average heel strike timing**  
   Temporal characteristic of the gait cycle describing heel strike timing during walking.

7. **Shoe size**  
   Patient’s shoe size, included as a proxy for foot length which may influence gait mechanics and pressure distribution.

#### Target Variables

The pain anamnesis dataset contains **42 target variables**:

- **18 binary variables** → indicate the presence or absence of localized pain (0 / 1)  
- **24 ordinal variables** → represent pain severity levels on a **0–5 scale**

#### Target Processing for Modeling

To improve the relationship between input features and prediction targets, different target processing strategies were applied depending on the modeling approach.

- **Neural Network (LSTM-based model)**  
  The original target structure was preserved. The model was designed as a **multi-task learning framework** that simultaneously predicts binary pain presence and ordinal pain severity.

- **XGBoost model**  
  The ordinal targets were converted into **binary variables (pain vs. no pain)**. This simplification reduces the complexity of the classification task and strengthens the statistical association between biomechanical gait features and the target variables.

#### Feature Distribution

Descriptive statistics were computed for all biomechanical features.

Key observations:

- Most biomechanical features exhibit approximately **Gaussian-like distributions**, indicating that they can be treated as continuous variables.
- Although **shoe size** is technically discrete, it spans a sufficiently wide range and was therefore handled as a continuous variable in the analysis.
- Since the feature ranges differ substantially, **feature normalization** was applied prior to model training.

Therefore:

- **Standardization or MinMax scaling** was applied before training machine learning models.

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

#### Target Dependency Analysis

Relationships between pain variables were analyzed using **Cramér’s V**.

Results:

- Some pairs of pain variables showed **moderate associations (Cramér’s V > 0.4)**.

Interpretation:

- Certain pain regions tend to **co-occur**, suggesting shared biomechanical or physiological causes.

---

## XGBoost

### Method

Instead of using raw sequences, we transformed time-series data into tabular form using **lag-based feature engineering**.

- Autocorrelation (ACF) used to determine optimal lag size  
- Past sensor values added as lag features  
- MinMax scaling applied  
- GroupShuffleSplit used to avoid leakage across collection IDs  

### Hyperparameter Optimization

- **Optuna** was used for resource-efficient hyperparameter optimization based on Bayesian optimization.  
  Instead of performing exhaustive grid search, Optuna explores the search space adaptively by learning from previous trials and focusing on promising regions.

The following hyperparameters were tuned:

  - `max_depth`  
    Controls the maximum depth of individual trees, balancing model complexity and overfitting.

  - `learning_rate`  
    Shrinks the contribution of each tree to improve generalization.

  - `n_estimators`  
    Determines the number of boosting rounds (trees).

  - `subsample`  
    Fraction of training samples used per tree, helping reduce overfitting.

  - `colsample_bytree`  
    Fraction of features randomly sampled for each tree, improving robustness and generalization.

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

- **SHAP summary plots and dependence plots** used to
- Identify biomechanical features influencing specific pain regions  
- Provide interpretable relationships between gait deviations and pain  

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


This project demonstrates how wearable gait data can be leveraged to build automated, interpretable healthcare prediction systems.
