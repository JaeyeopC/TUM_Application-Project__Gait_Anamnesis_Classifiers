# Machine Learning for Gait Event Detection & Pain Anamnesis Classification

## Project Summary

This project was developed for the course **Application Project [IN2328]** at **Technische Universität München**, conducted in association with **Eversion Technologies GmbH**, a company specializing in wearable gait analysis and biomechanical assessment systems.

The project was supervised by **Pietro Fantini** from the *Lehrstuhl für Technologie- und Innovationsmanagement*.

The objective of this project was to develop machine learning models that link biomechanical gait patterns to clinically relevant pain information. Two main tasks were addressed:

1. **Gait event detection** from wearable IMU sensor data  
2. **Pain anamnesis classification** using gait-derived biomechanical features  

---

## Data

Two structured datasets were used:

### Gait Event Dataset
Collected using 6-axis IMU sensors embedded in instrumented insoles.  
Each time-series sample contains:

- 3-axis acceleration and 3-axis gyroscope signals  
- timestamps and step information  
- foot side (left / right)  
- labeled gait events (e.g., heel strike, foot flat, heel off, toe off)

The dataset captures full walking cycles across multiple participants and serves as ground truth for event detection.

### Pain Anamnesis Dataset
Derived from standardized patient questionnaires combined with gait-based biomechanical measurements.  
It includes:

- movement and posture-related features extracted from gait analysis  
- binary indicators of localized pain  
- ordinal pain severity scores (0–5 scale) across multiple body regions  

This dataset enables modeling both the presence and intensity of pain.

---

## Methods

For gait event detection, sequence-based deep learning models (LSTM) were trained on raw IMU time-series data to identify gait phases in real time.  
As a complementary approach, XGBoost models with engineered temporal features and automated hyperparameter optimization (Optuna) were applied for comparison.

For pain analysis, a multi-output neural network was designed to jointly predict:

- binary pain presence  
- ordinal pain severity levels  

This multi-task learning framework captures relationships between biomechanical patterns and subjective pain reports.

To improve model interpretability, explainable AI techniques were applied. In particular, **SHAP (SHapley Additive exPlanations)** was used to analyze feature contributions and identify biomechanical factors most strongly associated with specific pain outcomes. This enabled transparent interpretation of model predictions and supported clinically meaningful insights.

---

## Outcome

The project compares deep learning and tree-based approaches in terms of predictive performance, interpretability, and practical deployment potential.  
Results show that data-driven modeling can reliably detect gait events, provide meaningful predictions of patient-reported pain, and offer interpretable insights into biomechanical risk factors, supporting the development of automated and scalable clinical assessment tools.

## Repository Structure

The repository is organized into dedicated branches to separate documentation, model development, and finalized implementations.

- **main**  
  Contains the final project report and documentation.

- **LSTM_final_models**  
  Includes the implemented and trained models based on LSTM architectures for gait event detection and pain anamnesis classification.

- **XGBoost_final_models**  
  Contains the finalized XGBoost models, including trained models and evaluation outputs.

- **XGBoost_Gait_Experiment**  
  Development and experimentation branch for XGBoost-based gait event detection, including feature engineering, model tuning, and exploratory analysis.

