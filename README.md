# ML_AD_WinterWork

Machine Learning & Autonomous Driving Winter Work Project

## 📁 Project Structure

```
ML_AD_WinterWork/
├── README.md                              # Project documentation
├── Backpropagation/                       # Backpropagation algorithm implementation
│   ├── backpropagation_EG1.py            # Core multi-layer neural network
│   ├── demo1.py                          # XOR and binary classification training script
    └── Loss_result_overTrainingSteps.png # Training loss curve 
```

## 🎯 Project Overview

### Task 1: Backpropagation Algorithm Implementation

A pure Python implementation of multi-layer neural network with backpropagation algorithm, featuring:

#### Core Module：[backpropagation_EG1.py]
-  Multi-layer neural network architecture 
-  Sigmoid activation function and its derivative
-  Forward Propagation
-  Backward Propagation (Backpropagation)
-  Gradient Descent optimizer
-  use MSE as the Loss function

#### Demos

**XOR Problem** : [demo1.py]
- Network architecture: 2-4-1
- Solves the classic non-linearly separable problem successfully
- Loss curve visualization in "Loss_result_overTrainingSteps.png"

## 🔧 Dependencies

### Python Version
- Python 3.7+

### Required Libraries
```bash
pip install numpy
pip install matplotlib
```

