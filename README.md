# MLP from Scratch

A configurable **Multi-Layer Perceptron** implemented from scratch in pure **NumPy** without any ML frameworks (no PyTorch, no TensorFlow, no scikit-learn for the model itself). Built to understand how neural networks actually work: forward pass, backpropagation, gradient descent, weight initialization.

> ⚠️ Work in progress — see [Roadmap](#roadmap) below.

## Features

- **Custom forward and backward propagation** — every gradient computed by hand
- **Activation functions:** ReLU, sigmoid, tanh, softmax
- **Loss functions:** MSE, categorical cross-entropy
- **He weight initialization** for ReLU-friendly training
- **Learning rate decay** during training
- **Configurable architecture** — any number of layers, neurons per layer, activations
- **Data preprocessing pipeline:**
  - Automatic numeric/categorical column detection
  - Min-max normalization for numeric features
  - One-hot encoding for categorical features and targets
  - Train/test split with reproducible seed
- **Model persistence** — save/load weights and architecture (NPZ or JSON)
- **Training visualization** — loss and accuracy plots generated automatically

## Tech stack

Python 3.x · NumPy · pandas · Matplotlib

## Project structure

```
mlp-from-scratch/
├── layer.py              # Single layer: weights, biases, forward/backward
├── network.py            # Full network: training loop, save/load
├── convertersModule.py   # Data loading, preprocessing, train/test split
├── plotsModule.py        # Loss/accuracy visualization
├── main.py               # Entry point
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/Blurry13/mlp-from-scratch.git
cd mlp-from-scratch

# Create and activate virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate      # Windows
source .venv/bin/activate   # Linux / macOS

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python main.py
```

The program will ask whether to train a new model or load a previously saved one. For training, it will then prompt you to:

1. Select a CSV file via file dialog
2. Enter the name of the target column
3. Configure each layer (number of neurons, activation function)
4. Choose a loss function (`mse` or `crossentropy`)

After training, the model is saved to disk along with loss and accuracy plots.

## Example output

After training, the script generates two plots:

- `lossPlot.png` — training loss per epoch
- `accuracyPlot.png` — training accuracy per epoch

## Roadmap

- [ ] Move normalization fit/transform after train/test split (fix data leakage)
- [ ] More optimizers (Momentum, Adam)
- [ ] Unit tests with pytest
- [ ] Type hints and docstrings throughout
- [ ] CLI flags instead of interactive prompts
- [ ] Adaptive weight initialization based on activation function (currently He init for all layers; add Xavier/Glorot for tanh/sigmoid)
