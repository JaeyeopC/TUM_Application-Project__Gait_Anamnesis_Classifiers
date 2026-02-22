# Machine Learning for Gait Event Detection & Pain Anamnesis Classification

## Project Summary

This project was developed for the course **Application Project [IN2328]** at **Technische Universität München**.

The project investigates two closely related problems:
1. Gait event detection from wearable IMU sensor data  
2. Pain anamnesis classification based on biomechanical gait features  

For gait analysis, deep learning models based on Long Short-Term Memory (LSTM) networks were used to identify key gait events (e.g., heel strike, toe-off) from time-series sensor data. In parallel, gradient boosting methods (XGBoost) with automated hyperparameter optimization were applied as an alternative approach using engineered temporal features.

For clinical assessment, a multi-output neural network was developed to predict both the presence and severity of pain across multiple body regions using gait-derived biomechanical features. The model jointly handles binary pain indicators and ordinal pain severity levels through a multi-task learning framework.

The project compares deep learning and tree-based methods in terms of predictive performance, interpretability, and practical applicability in clinical and real-world industry settings. Overall, the results demonstrate that data-driven models can support automated gait analysis and provide meaningful insights into patient-reported pain, contributing to more objective and scalable diagnostic support systems.
