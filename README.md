# ML_AD_WinterWork

Machine Learning & Autonomous Driving Winter Work Project

## 📁 Project Structure

```
ML_AD_WinterWork/
├── README.md                              
├── Backpropagation/                       # Backpropagation algorithm implementation
│   ├── backpropagation_EG1.py             # Core neural network
│   ├── demo1.py                           # classification training script（XOR）
    └── Loss_result_overTrainingSteps.png  # The Loss curve 
```

## 🎯 Project Overview

### Task 1: Backpropagation Algorithm Implementation

A implementation of multi-layer neural network with backpropagation algorithm, 

#### NN Module：[backpropagation_EG1.py]
-  Multi-layer neural network architecture 
-  Sigmoid activation function and its derivative
-  Gradient Descent optimizer and weight update
-  use MSE as the Loss function

#### Demos

**XOR Problem** : [demo1.py]
- Network architecture: 2-4-1
- Solves the classic non-linearly separable problem successfully
- Loss curve visualization in "Loss_result_overTrainingSteps.png"

## 🔧 Dependencies

### Python Version
- Python 3.11.7

### Required Libraries
```bash
pip install numpy
pip install matplotlib
```

