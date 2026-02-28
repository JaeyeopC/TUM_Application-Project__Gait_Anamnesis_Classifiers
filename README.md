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

# Task: Gait Event Detection
The goal of this task is to classify gait events from IMU time-series signals.
Two different modeling approaches were implemented.

---

## XGBoost

### Method
Lag-based feature engineering was used to convert IMU time-series into tabular features for XGBoost.
- ACF-based lag selection (average optimal lag ≈ 19 timesteps)
- Lag feature generation (6 IMU channels × 19 lags = 114 lag features)
- MinMax normalization
- Group-aware splitting to avoid leakage across collection IDs (GroupShuffleSplit)

### Hyperparameter Optimization (Optuna)
Optuna was used for resource-efficient hyperparameter optimization based on Bayesian optimization (TPE sampler).  
The objective was to maximize validation accuracy using a single train/validation split.

**Final hyperparameters (Optuna):**
- `learning_rate`: 0.07
- `max_depth`: 9
- `n_estimators`: 306
- `subsample`: 0.52
- `colsample_bytree`: 0.80
- `gamma`: 0.85

### Result
- **Test accuracy:** ≈ 92%
- **Weighted F1-score:** ≈ 0.92

Per-class performance highlights:
- Strong performance on frequent classes (e.g., **Foot Flat**, **Toe Off**) with scores around ~0.90+
- Lower performance on the rare class **Heel Strike**:
  - Precision: 0.71
  - Recall: 0.53
  - F1-score: 0.61

Collection-level evaluation:
- ~74% of test collection IDs achieved **>90% accuracy** (342 / 465 collection IDs)

Distribution shift analysis (**Jensen–Shannon Divergence, JSD**) was used to quantify the similarity between the predicted and true gait phase distributions:

- JSD excluding Heel Strike: **0.006**
- JSD including Heel Strike: **0.12**

This result indicates that most distribution mismatch is caused by the Heel Strike phase, while the remaining gait phases are predicted with highly similar distributions.

---

## LSTM

### Method

Raw IMU sequences were directly used as input to an LSTM network in order to capture temporal dependencies in gait signals.

- Sliding window segmentation (**2–3 seconds per window**)
- Sequence-based classification
- MinMax normalization applied to all IMU channels

### Architecture

The model consists of stacked recurrent layers designed to capture temporal patterns in gait signals.

- **3 stacked LSTM layers**
- **128 hidden units per layer**
- **Dropout** applied for regularization
- **Fully connected layer with softmax activation** for multi-class gait phase prediction

### Training

- **Optimizer:** Adam (learning rate = 0.001)
- **Loss function:** Cross-entropy loss
- **Train / validation / test split:** 70 / 15 / 15
- **Early stopping** used to prevent overfitting

### Result

- **Overall test accuracy:** ~91%

Performance characteristics:

- Strong performance on frequent classes such as **Foot Flat** and **Toe Off**
- Lower recall for the rare class **Heel Strike**, similar to the XGBoost model due to class imbalance

Compared to the XGBoost approach, the LSTM model directly learns temporal dependencies from raw IMU sequences, which allows it to achieve competitive performance without manual lag feature engineering.

---
# Task: Pain Anamnesis Classification

Goal: Predict patient pain profile using gait-derived biomechanical features.

---

## XGBoost

### Method

The XGBoost model was applied to structured biomechanical gait features to predict pain-related targets.

- **7 biomechanical gait features** used as input  
- Each pain variable modeled as an **independent classification task**  
- Ordinal pain scores (0–5) converted to **binary indicators (pain vs. no pain)** to simplify the prediction task  
- **Feature standardization** applied before training  

This approach enables interpretable models while handling multiple pain prediction tasks independently.

### Hyperparameter Optimization

Hyperparameters were optimized using **Optuna**, which performs efficient Bayesian optimization using the **Tree-structured Parzen Estimator (TPE)** sampler.

Key hyperparameters tuned include:

- `max_depth`
- `learning_rate`
- `n_estimators`
- `subsample`
- `colsample_bytree`
- `scale_pos_weight`

Additionally, **decision threshold optimization** was applied for each pain target to improve classification performance under class imbalance.

Instead of using the default probability threshold (**0.5**), the optimal threshold was selected using **Youden’s J statistic**, defined as:

J = Sensitivity + Specificity − 1

The threshold that maximized **Youden’s J** on the validation set was chosen for each pain variable.  
This approach identifies the decision boundary that best balances **true positive rate (sensitivity)** and **true negative rate (specificity)**, which is particularly useful in medical classification tasks.

### Model Interpretation

To interpret the trained models, **SHAP (SHapley Additive exPlanations)** was used.

- **SHAP summary plots** were used to visualize global feature importance.
- **SHAP dependence plots** were used to analyze relationships between biomechanical features and pain predictions.

This analysis helped identify which gait deviations contribute most strongly to specific pain regions and provided interpretable insights into the model behavior.

---

## LSTM / Multi-Output Neural Network

Since the pain anamnesis dataset consists of structured biomechanical features rather than time-series sequences, a **multi-output neural network** was used instead of a recurrent LSTM model.

### Input

- **7 biomechanical gait features**

### Architecture

The neural network was designed as a **multi-task learning model** with shared feature extraction layers.

- Shared dense trunk: **64 → 32 → 32**

Two output branches were used:

Binary pain prediction:
- **18 binary outputs**
- Sigmoid activation

Ordinal pain severity prediction:
- **24 ordinal outputs**
- Implemented using the **CORAL (Cumulative Ordinal Regression with Logistic models)** framework

This architecture allows the model to simultaneously learn **pain presence** and **pain severity**.

### Training

The model was trained using a multi-task loss function combining binary and ordinal objectives.

- **Binary Cross Entropy** for binary targets
- **CORAL-based ordinal regression loss** for severity prediction
- **Adam optimizer**
- Multi-task loss balancing to stabilize training

### Results

The multi-task neural network achieved the following performance:

- **Binary pain prediction accuracy:** ~80%
- **Ordinal pain severity prediction:** ~1.4 MAE

Compared to single-task models, the multi-task approach improved prediction consistency by allowing the model to jointly learn relationships between pain presence and pain severity across different body regions.
