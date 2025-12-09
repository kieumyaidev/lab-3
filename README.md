# Lab 3: Deep Learning - EEG Mental State Classification

## Project Overview

This project implements binary emotion classification from EEG signals using deep learning architectures. The goal is to distinguish between **focused** and **drowsed** mental states using neural networks (MLPs and CNNs), demonstrating improvements over traditional machine learning approaches.

## Problem Statement

The task involves classifying EEG signals into two states:
- **Class 0**: Focused mental state
- **Class 1**: Drowsed mental state

This is the same problem addressed in the Supervised Learning Final Project, but using deep learning approaches to achieve better performance.

## Dataset

**SEED-IV Dataset** from Shanghai Jiao Tong University
- **Source**: https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html
- **Subjects**: 15 participants
- **Sessions**: 3 sessions per subject (recorded on different days)
- **Trials**: 24 trials per subject
- **Channels**: 62-channel ESI NeuroScan System
- **Sampling Rate**: 128 Hz
- **Total Samples**: 25,794 samples
- **Features**: 2,232 features (62 channels × 36 frequency bands)

### Data Preprocessing
- Feature extraction using Short-Time Fourier Transform (STFT)
- Two input representations:
  - **MLP**: Flattened features (2,232-dimensional vector)
  - **CNN**: 2D spatial structure (1×62×36 tensor)

## Methodology

### Architectures Implemented

1. **Multi-Layer Perceptron (MLP)**
   - Architecture: 2232 → 512 → 128 → 32 → 2
   - Fully connected layers with ReLU activation
   - Input: Flattened features

2. **Convolutional Neural Network (CNN)**
   - Architecture: 2 convolutional layers (16 and 32 filters) + fully connected layers
   - Input: 2D spatial structure (62 channels × 36 frequency bands)
   - Preserves spatial relationships between channels and frequency bands

### Training Details
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Cross-Entropy Loss
- **Epochs**: 20 (with potential for longer training)
- **Cross-Validation**: 3-fold leave-one-session-out
- **Hyperparameter Tuning**: Grid search for learning rates and batch sizes

### Key Features
- Comprehensive EDA
- Comparison with traditional ML methods
- Hyperparameter optimization
- Detailed performance analysis

## Results

| Model | Mean Accuracy | Improvement |
|-------|--------------|-------------|
| **CNN** | **75.34%** | +17% vs SVM |
| MLP | 59.75% | +1.4% vs SVM |

**Comparison with Supervised Learning Final Project:**
- **CNN**: 75.34% (vs SVM: 58.37%) - **17% improvement**
- **MLP**: 59.75% (vs SVM: 58.37%) - Slight improvement

**Key Findings:**
- CNN significantly outperformed traditional ML, demonstrating the power of deep learning for EEG classification
- Preserving spatial structure (2D representation) is crucial for capturing channel-frequency relationships
- 75% accuracy is competitive with published results (typically 70-85% with extensive tuning)

## Installation & Setup

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- PyTorch (with CUDA/MPS support recommended)

### Dependencies
Install required packages:
```bash
pip install -r requirements.txt
```

### Dataset Setup
1. Download SEED-IV dataset from: https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html
2. Extract the dataset to a `SEED_IV/` folder in the project directory
3. Ensure the folder structure: `SEED_IV/eeg_raw_data/{session}/{subject}.mat`

### Running the Notebook
1. Open `lab_3.ipynb` in Jupyter Notebook
2. Run all cells sequentially
3. The notebook includes:
   - Data loading and preprocessing
   - Exploratory Data Analysis (EDA)
   - MLP and CNN implementation
   - Model training and evaluation
   - Results visualization
   - Comparison with traditional ML
   - Discussion and analysis

## Project Structure

```
Lab_3a/
├── README.md
├── lab_3.ipynb          # Main notebook
├── loading_data.py      # Data loading utilities
├── CONSTANT.py          # Configuration constants
└── requirements.txt     # Python dependencies
```

## Key Concepts

- **Short-Time Fourier Transform (STFT)**: Frequency-domain feature extraction
- **Multi-Layer Perceptron (MLP)**: Feedforward neural network with multiple hidden layers
- **Convolutional Neural Network (CNN)**: Deep learning architecture using convolutional layers
- **PyTorch**: Deep learning framework
- **Adam Optimizer**: Adaptive learning rate optimization algorithm
- **Cross-Entropy Loss**: Loss function for classification
- **Leave-One-Session-Out Cross-Validation**: Ensures generalization across sessions

## References

- **Dataset**: SEED-IV Dataset - https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html
- **Key Concepts**: CU Boulder's Deep Learning Course Note

## Author

Completed as part of the Deep Learning Final Project.

