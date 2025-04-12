# Machine Learning for Gait Event Detection and Pain Anamnesis Analysis - XGBoost approach 
--- 
This project includes two approaches:
1. Gait Event Detection using lagged features
2. Pain Anamnesis Analysis with XGBoost ( with the exploratory data analysis )
---
## Gait Event Detection 
**Features** 
- Autocorrelation Function to calculate the lag size  
- Classifies 4 gait events: heel strike, foot flat, heel off, toe off 
- XGBoost for time series approach is used
- Hyperparameter optimized using Optuna 
    - Bayesian hyperparameter tuning 
- Final model and the performance report are stored in 
    - ```Gait_Analysis/model/``` and ```Gait_Analysis/report/```
- Optuna study object is stored in
    - ```Gait_Analysis/study/```

**Performance** 
- ~92% overall accuracy 
- ~92% overall weighted F1 score 
- Relatively low evaluation metrics on heel strike 

**Future Work** 
- Correct implementation of resampling techniques in a way that keeps the data distribution and temporal relevance to compensate for the data imbalance. (e.g., SMOTE + Tomek Links) 

## Pain Anamnesis Analysis 
**Features** 
- Exploratory Data Analysis (EDA)
    - Mainly explored the correlation between 
        - (1) features vs features  
        - (2) features vs targets
        - (3) targets vs targets  
    - EDA was done for two different approaches:
        - One with original target columns ( 18 binary targets, 24 ordinal targets )
        - The other with converted target columns ( 42 binary targets )
- Separate models for each target columns are trained 
- Hyperparameters are optimized using Optuna 
    - Bayesian hyperparameter tuning
    - Dyanamic thresholding was used 
- SHAP analysis was conducted to investigate the behavior of the final models 
- Final model and the performance report are stored in 
    - ```Anamnesis_Analysis/model/``` and ```Anamnesis_Analysis/report/```

**Performance** 
- ~62% overall accuracy 
- around 50~55% F1 and PRAUC scores 

**Future work** 
- Resampling might be needed for negative dominant targets 
- Investigate the most effective features for each target columns via EDA and SHAP analysis 
- Currently Optuna is trained to optimize validation accuracies, however, depending on the requirements, train the model based on the other evaluation metrics such as F1 scores or recall scores. 

**miscellaneous** 
- Previous experiments with different settings are included in the directory
    - 1,2,3,4,5_Anamnesis_XGBoost_optuna__... 
    - For the future investigation 
- Final submission file :
    - ```Final_anamnesis_XGBoost_optuna__...```
    - ```Final_anamnesis_XGBoost_optuna__...__Reduced_Size``` the same file as before but with some figures removed as the original file size is too big to be displayed on github. 

---
## Data 
1. Gait Analysis 
- ```Gait_Analysis/csv_output_with_phases/``` 
2. Pain Anamnesis Analysis 
- data used for EDA are included in each directory 
    - ```Anamnesis_Analysis/EDA/converted_to_binary_columns/data``` 
    - ```Anamnesis_Analysis/EDA/original_target_columns/data```
- data used for training is included in 
    - ```Anamnesis_Analysis/data```

---
## Dependencies 
```bash
pip install pandas>=2.2.3 scikit-learn>=1.5.2 xgboost>=2.1.4 matplotlib>=3.9.2 seaborn>=0.12.2 optuna>=4.2.1 shap>=0.46.0 joblib>=1.4.2 tqdm>=4.66.2 openpyxl>=3.1.2
```

or 
```bash 
pip install -r requirements.txt 
``` 


```
pandas>=2.2.3
scikit-learn>=1.5.2
xgboost>=2.1.4

# Data processing and visualization
matplotlib>=3.9.2
seaborn>=0.12.2
optuna>=4.2.1
shap>=0.46.0

# Utilities
joblib>=1.4.2
tqdm>=4.66.2
openpyxl>=3.1.2
```
## Project Structure 
```
├── Anamnesis_Analysis
│   ├── …  
│   ├── EDA
│   ├── Final_Anamnesis_XGBoost_optuna__different_params_for_models__no_resample__all_features__Default_Threshold_No_Scale.ipynb
│   ├── data
│   ├── model
│   └── report
├── Gait_Analysis
│   ├── csv_output_with_phases
│   ├── final_model
│   ├── gait_analysis_final_submission.ipynb
│   ├── report
│   └── study
├── README.md
└── requirements.txt
```

