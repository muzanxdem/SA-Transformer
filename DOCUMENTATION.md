# SA-Transformer - Transformer Architecture Documentation

## 📋 Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Dependencies](#dependencies)
4. [Installation](#installation)
5. [Project Components](#project-components)
6. [Workflow](#workflow)
7. [Configuration](#configuration)
8. [Model Architecture](#model-architecture)
9. [Data Processing](#data-processing)
10. [Training](#training)
11. [Model Conversion & Deployment](#model-conversion--deployment)
12. [Evaluation Criteria](#evaluation-criteria)
13. [Troubleshooting](#troubleshooting)

---

## Overview

**SA-Transformer** is a deep learning project focused on **Sentiment Analysis** using PyTorch with a **Transformer architecture**. The project implements a Transformer encoder model to classify text sentiment into three categories: **neutral**, **positive**, and **negative**. This implementation is separate from the MLP architecture in `SAAssignment2025/SA/`.

### Key Features

- **Transformer Architecture**: Self-attention mechanism for learning feature interactions
- **Data Preprocessing**: Automated CSV processing with feature encoding and normalization
- **Training Pipeline**: Complete training workflow with early stopping and learning rate scheduling
- **Model Deployment**: ONNX conversion for production deployment
- **Experiment Tracking**: Weights & Biases (WandB) integration
- **Performance Benchmarking**: ONNX runtime testing with inference speed metrics

### Project Goals

- Build a robust sentiment classification model using Transformer architecture
- Demonstrate proper deep learning workflow (data → preprocessing → training → evaluation)
- Implement best practices: modular code, proper data splits, appropriate loss functions
- Provide comprehensive analysis of results
- Create production-ready model with ONNX export
- Compare Transformer performance with MLP architecture

---

## Project Structure

```
SA-Transformer/
├── data/
│   └── SAAssignment2025/
│       ├── train.npy                               # Training data (features + labels)
│       ├── test.npy                                 # Test data
│       ├── val.npy                                  # Validation data
│       └── class_names.npy                          # Class names array
│
├── SA-Transformer/                                  # Main project directory
│   ├── train.py                                     # Main training script
│   ├── model.py                                     # Transformer model architecture
│   ├── preprocess.py                                # Data preprocessing utilities
│   ├── convert.py                                   # PyTorch to ONNX conversion
│   ├── onnxtest.py                                  # ONNX model testing
│   ├── config.yaml                                  # Training configuration
│   ├── README.md                                    # SA-Transformer directory documentation
│   ├── SA-Transformer.pth                          # Trained model weights
│   ├── SA-Transformer.onnx                         # ONNX model file (after conversion)
│   ├── scaler.pkl                                   # Saved StandardScaler
│   └── wandb/                                       # WandB experiment logs
│
├── models/                                          # Additional model checkpoints
│
├── helper_functions.py                              # Utility functions
├── pyproject.toml                                   # Project dependencies
├── uv.lock                                          # Dependency lock file
├── README.md                                        # Project overview
└── DOCUMENTATION.md                                 # This file

```

---

## Dependencies

### Required Python Packages

The project uses `uv` for dependency management. All dependencies are specified in `pyproject.toml`:

**Core Deep Learning:**
- `torch>=2.9.1` - PyTorch framework
- `torchvision>=0.24.1` - Computer vision utilities
- `torchmetrics==0.9.3` - Metrics for PyTorch

**Data Processing:**
- `numpy>=2.3.5` - Numerical computing
- `pandas>=2.3.3` - Data manipulation
- `scikit-learn>=1.8.0` - Machine learning utilities

**Model Deployment:**
- `onnx>=1.20.0` - ONNX model format
- `onnxruntime>=1.23.2` - ONNX inference runtime
- `onnxruntime-gpu>=1.23.2` - GPU-accelerated ONNX runtime

**Development & Tracking:**
- `jupyter>=1.1.1` - Jupyter notebooks
- `ipykernel>=7.1.0` - Jupyter kernel
- `wandb>=0.23.1` - Weights & Biases for experiment tracking
- `tqdm>=4.67.1` - Progress bars

**Utilities:**
- `matplotlib>=3.10.8` - Plotting
- `torchinfo>=1.8.0` - Model summary
- `torch-summary>=1.4.5` - Model architecture summary
- `kagglehub>=0.3.13` - Kaggle dataset access
- `pyyaml>=6.0` - YAML configuration parsing

### Python Version

- **Python 3.12+** required

### CUDA Support

- **CUDA 13.0+** (optional, for GPU acceleration)
- PyTorch CUDA packages from `pytorch-cu130` index

---

## Installation

### Prerequisites

1. **Python 3.12+** installed
2. **uv** package manager installed
3. **CUDA 13.0+** (optional, for GPU training)

### Setup Steps

1. **Navigate to the project directory:**
   ```bash
   cd /home/user/jupyter/DL-Obi/SA-Transformer
   ```

2. **Install dependencies using uv:**
   ```bash
   uv sync
   ```
   This will create a virtual environment and install all dependencies.

3. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

---

## Project Components

### 1. Main Training Script (`SA-Transformer/train.py`)

**Purpose**: Trains the Transformer sentiment analysis model with full monitoring and evaluation.

**Key Functions:**

#### `train_model()`
- Trains model for specified epochs
- Implements early stopping (patience=5)
- Saves best model based on validation loss
- Uses learning rate scheduling (ReduceLROnPlateau)
- Applies gradient clipping (max_norm=1.0)
- Logs metrics to WandB

#### `evaluate_model()`
- Evaluates model on given dataloader
- Computes loss and predictions
- Returns average loss, predictions, and labels

#### `test_and_report()`
- Final evaluation on test set
- Generates classification report
- Creates confusion matrix
- Logs accuracy to WandB

**Features:**
- Automatic device detection (CUDA/CPU)
- Label range validation
- Automatic class count detection
- WandB experiment tracking
- Model checkpointing

**Usage:**
```bash
cd SA-Transformer/SA-Transformer
uv run train.py
# or alternatively: python train.py
```

**Expected Input Files:**
- `../data/SAAssignment2025/train.npy` - Training data
- `../data/SAAssignment2025/test.npy` - Test data
- `../data/SAAssignment2025/val.npy` - Validation data
- `../data/SAAssignment2025/class_names.npy` - Class names array
- `config.yaml` - Configuration file

**Output Files:**
- `SA-Transformer.pth` - Best model weights
- `scaler.pkl` - StandardScaler for data normalization
- WandB logs in `wandb/` directory

### 2. Transformer Model Architecture (`SA-Transformer/model.py`)

**Purpose**: Defines the Transformer encoder architecture for tabular data.

**Architecture:**
```
Input Features (batch, n_features)
    ↓
Feature Projection: (n_features, batch, 1) → (n_features, batch, d_model)
    ↓
Positional Encoding
    ↓
Transformer Encoder (num_layers × TransformerEncoderLayer)
    ↓
Global Pooling (mean over sequence)
    ↓
Classification Head
    ↓
Output Logits (batch, num_classes)
```

**Key Components:**
- **Input Projection**: Maps each feature to `d_model` dimensions
- **Positional Encoding**: Sinusoidal positional encoding for feature order
- **Transformer Encoder**: Multi-head self-attention layers
- **Global Pooling**: Mean pooling over sequence length
- **Classification Head**: Fully connected layers for classification

**Initialization:**
```python
from model import TransformerModel
model = TransformerModel(
    input_features=num_features,
    num_classes=num_classes,
    d_model=128,
    nhead=8,
    num_layers=3,
    dim_feedforward=512,
    dropout=0.1
)
```

### 3. Data Preprocessing Utilities (`SA-Transformer/preprocess.py`)

**Purpose**: Handles data normalization and PyTorch DataLoader creation.

**Functions:**
- `preprocess()` - Main preprocessing for training
  - Fits StandardScaler on training data
  - Transforms train/test/val sets
  - Creates PyTorch DataLoaders
  - Saves scaler for later use

- `preprocess_onnx()` - Preprocessing for ONNX inference
  - Loads saved scaler
  - Transforms test data
  - Creates DataLoader for ONNX testing

**Data Format:**
- Input arrays: `(n_samples, n_features + 1)`
- Last column contains labels (integers)
- Features are normalized using StandardScaler (mean=0, std=1)

### 4. ONNX Conversion (`SA-Transformer/convert.py`)

**Purpose**: Converts trained PyTorch model to ONNX format for deployment.

**Features:**
- Loads trained model weights
- Exports to ONNX with dynamic batch size
- Supports both CPU and GPU inference

**Usage:**
```bash
cd SA-Transformer/SA-Transformer
uv run convert.py
# or alternatively: python convert.py
```

**Requirements:**
- `SA-Transformer.pth` must exist (trained model)
- `config.yaml` must be configured
- Training data needed to determine input shape

**Output:**
- `SA-Transformer.onnx` - ONNX model file

### 5. ONNX Testing (`SA-Transformer/onnxtest.py`)

**Purpose**: Tests ONNX model performance and benchmarks inference speed.

**Features:**
- Loads ONNX model with CUDA/CPU providers
- Performs inference on test set
- Calculates accuracy
- Measures inference time and throughput
- Warm-up run for accurate timing

**Usage:**
```bash
cd SA-Transformer/SA-Transformer
uv run onnxtest.py
# or alternatively: python onnxtest.py
```

**Output:**
- Accuracy percentage
- Total inference time
- Average time per batch
- Throughput (samples/second)

**Requirements:**
- `SA-Transformer.onnx` - ONNX model file
- `scaler.pkl` - Saved scaler
- `../data/SAAssignment2025/test.npy` - Test data

### 6. Configuration File (`SA-Transformer/config.yaml`)

YAML configuration file for training parameters.

**Current Configuration:**
```yaml
class_names:
   - 'neutral'
   - 'positive'
   - 'negative'

batch_size: 128
learning_rate: 0.001
num_epochs: 30

d_model: 128
nhead: 8
num_layers: 3
dim_feedforward: 512
dropout: 0.1
max_seq_len: 1000
```

**Configuration Parameters:**
- `class_names`: List of class names (used for reporting)
- `batch_size`: Number of samples per batch
- `learning_rate`: Initial learning rate for AdamW optimizer
- `num_epochs`: Maximum number of training epochs
- `d_model`: Model dimension
- `nhead`: Number of attention heads
- `num_layers`: Number of transformer encoder layers
- `dim_feedforward`: Feedforward network dimension
- `dropout`: Dropout rate
- `max_seq_len`: Maximum sequence length for positional encoding

---

## Workflow

### Complete Training Pipeline

1. **Data Preparation** (uses preprocessed data from `SAAssignment2025`):
   - Data should already be preprocessed
   - Files: `train.npy`, `test.npy`, `val.npy`, `class_names.npy`

2. **Training**:
   ```bash
   cd SA-Transformer/SA-Transformer
   uv run train.py
   ```
   - Loads data and configuration
   - Validates label ranges
   - Trains Transformer model with early stopping
   - Saves best model to `SA-Transformer.pth`

3. **Model Conversion** (Optional):
   ```bash
   cd SA-Transformer/SA-Transformer
   uv run convert.py
   ```
   - Converts PyTorch model to ONNX format

4. **ONNX Testing** (Optional):
   ```bash
   cd SA-Transformer/SA-Transformer
   uv run onnxtest.py
   ```
   - Tests ONNX model performance
   - Benchmarks inference speed

---

## Model Architecture Details

### Network Structure

```
Input (n_features)
    ↓
[Feature Projection: Linear(1 → d_model)]
    ↓
[Positional Encoding]
    ↓
[Transformer Encoder Layer 1]
    ├── Multi-Head Self-Attention
    ├── Feedforward Network
    └── Residual Connections + Layer Norm
    ↓
[Transformer Encoder Layer 2]
    ...
    ↓
[Transformer Encoder Layer N]
    ↓
[Global Pooling: Mean over sequence]
    ↓
[Linear(d_model → dim_feedforward//2)] → [GELU] → [Dropout]
    ↓
[Linear(dim_feedforward//2 → num_classes)]
    ↓
Output (logits)
```

### Training Configuration

- **Optimizer**: AdamW
- **Loss Function**: CrossEntropyLoss
- **Learning Rate Scheduler**: ReduceLROnPlateau
  - Mode: min (monitor validation loss)
  - Factor: 0.1 (reduce by 10x)
  - Patience: 3 epochs
- **Early Stopping**: 
  - Patience: 5 epochs
  - Monitors: validation loss
- **Gradient Clipping**: max_norm=1.0
- **Regularization**: 
  - Dropout: 10%
  - Layer Normalization in Transformer layers

---

## Data Format

### Input Data Structure

The numpy arrays (`train.npy`, `test.npy`, `val.npy`) should have shape:
```
(n_samples, n_features + 1)
```

- **Last column**: Integer labels (0 to num_classes-1)
- **Other columns**: Feature values (will be normalized)

### Label Requirements

- Labels must be integers in range `[0, num_classes-1]`
- The script validates label ranges before training
- Invalid ranges will raise a `ValueError`

### Class Names

- Stored in `class_names.npy` as a numpy array
- Automatically loaded during training
- Used for classification reports and confusion matrices

---

## Training Process

### Training Loop

1. **Data Loading**: Loads train/val/test arrays and class names
2. **Validation**: Checks label ranges match expected class count
3. **Preprocessing**: Normalizes features using StandardScaler
4. **Model Creation**: Initializes Transformer model with correct dimensions
5. **Training**: 
   - Forward pass through Transformer
   - Loss calculation
   - Backward pass with gradient clipping
   - Optimizer step
   - Learning rate scheduling
   - Early stopping check
6. **Evaluation**: Tests on test set with detailed metrics

### Monitoring

- **WandB Integration**: Logs training/validation loss, learning rate, accuracy
- **Console Output**: Progress bars, epoch summaries, final reports
- **Model Checkpointing**: Saves best model based on validation loss

---

## Evaluation Metrics

The `test_and_report()` function generates:

1. **Accuracy**: Overall classification accuracy
2. **Classification Report**: Per-class precision, recall, F1-score
3. **Confusion Matrix**: Class-wise prediction distribution

---

## Troubleshooting

### Common Issues

1. **Out of Memory**
   - Reduce `batch_size` in config.yaml
   - Reduce `d_model` or `dim_feedforward`
   - Reduce `num_layers`

2. **Slow Training**
   - Use GPU if available (CUDA)
   - Reduce model size
   - Increase batch size if memory allows

3. **Poor Performance**
   - Try different learning rates
   - Adjust dropout rate
   - Increase model capacity
   - Train for more epochs

---

## Comparison with MLP Architecture

| Aspect | MLP (SAAssignment2025) | Transformer (SA-Transformer) |
|--------|------------------------|----------------------------|
| Architecture | Feedforward | Self-attention |
| Feature Interactions | Limited | Rich (via attention) |
| Parameters | Fewer (~100K) | More (~600K) |
| Training Time | Faster | Slower |
| Interpretability | Low | Higher (attention weights) |
| Best For | Simple patterns | Complex feature relationships |

---

## Notes

- The Transformer model treats each feature as a token in a sequence
- Positional encoding helps the model understand feature order
- Self-attention allows learning complex feature interactions
- Global pooling aggregates the sequence representation
- Model saves best weights based on validation loss
- ONNX conversion requires the model to be in eval mode

---

**Last Updated:** December 2024  
**Architecture:** Transformer Encoder  
**Status:** Production Ready


