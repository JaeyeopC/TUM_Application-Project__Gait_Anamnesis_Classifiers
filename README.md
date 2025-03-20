# Machine Learning for Gait Event Detection and Pain Classification

This repository contains two main projects:
1. Gait Event Classification using LSTM Networks
2. Pain Anamnesis Classification with Neural Networks

## Project Overview

### 1. Gait Event Classification (LSTM)

This project implements a Long Short-Term Memory (LSTM) neural network to classify gait events using IMU (Inertial Measurement Unit) sensor data.

#### Features
- Processes time-series data from IMU sensors (gyroscope and accelerometer)
- Classifies five gait events: no event, heel strike, foot flat, heel off, toe off
- Uses sequence-based LSTM architecture for temporal pattern recognition

#### Model Architecture
- Multi-layer LSTM network
- Input: 6-dimensional sensor data (3-axis gyroscope + 3-axis accelerometer)
- Hidden layers: 3 LSTM layers with 128 hidden units
- Output: 5 classes (gait events)
- Includes dropout for regularization

#### Performance
- Achieves ~91% overall accuracy
- Strong performance on major gait phases
- Some challenges with rare events (e.g., heel strike)

### 2. Pain Anamnesis Classification

This project offers two approaches to classify pain symptoms based on biomechanical measurements:

#### A. Binary Classification Approach
- Converts all pain indicators to binary (0/1) format
- Deep neural network with 7 hidden layers
- Multi-target classification for various body regions
- Achieves ~78% accuracy on binary predictions

#### B. Multi-Task Approach
- Handles both binary and ordinal pain indicators
- Shared-trunk architecture with task-specific heads
- Binary classification for pain presence
- Ordinal classification (0-5) for pain intensity
- Uses CORAL (Consistent Rank Logits) for ordinal prediction
- Achieves ~80% accuracy on binary tasks and MAE of ~1.4 on ordinal tasks

## Setup Instructions

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
# For macOS/Linux
python -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Setup data:
- For Gait Classification:
  - Place IMU sensor data in `csv_output_phases` directory (can change the name in the code)
  - Data should contain gyroscope and accelerometer readings
- For Pain Classification:
  - Place the pain anamnesis data in `project_2/colored_columns_output_filtered.xlsx` (can change the name in the code)

5. Run Jupyter Notebook:
```bash
jupyter notebook
```

## Project Structure
```
├── gait_event_classification_lstm_final.ipynb # Gait event classification
├── pain_anamnesis_classification_binary_only.ipynb # Binary pain classification
├── pain_anamnesis_classification_updated.ipynb # Multi-task pain classification
├── requirements.txt # Project dependencies
```


## Data Requirements

### Gait Classification Data
- Time-series data from IMU sensors
- Features: gyroscope (x,y,z) and accelerometer (x,y,z)
- Labeled gait events

### Pain Classification Data
- 7 biomechanical measurements including:
  - Left/Right movement deviation averages
  - Left/Right resting deviation averages
  - Left/Right step averages
  - Shoe size
- Pain indicators for various body regions

## Model Training

Each notebook contains detailed steps for:
1. Data preprocessing
2. Model architecture implementation
3. Training configuration
4. Model evaluation
5. Inference examples

## Dependencies

See `requirements.txt` for detailed package requirements.
