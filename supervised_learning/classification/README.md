# Classification

A neural network classification library built from scratch using NumPy, implementing binary and multiclass classification models of increasing complexity — from a single neuron up to deep neural networks.

## Project Structure

```
supervised_learning/classification/
├── 0-neuron.py       # Single neuron (public attributes)
```

## Concepts Covered

- Single neuron binary classification
- Multi-layer neural networks
- Forward and backward propagation
- Gradient descent optimization
- Binary and multiclass (softmax) classification
- Loss and cost functions (binary cross-entropy, categorical cross-entropy)
- One-hot encoding

## Requirements

- Python 3.5
- NumPy 1.15
- Ubuntu 16.04 LTS

## Usage

```python
import numpy as np
Neuron = __import__('0-neuron').Neuron

np.random.seed(0)
neuron = Neuron(784)
print(neuron.W)   # shape (1, 784)
print(neuron.b)   # 0
print(neuron.A)   # 0
```

## Data

Training and evaluation use three datasets stored in `../data/`:

- `Binary_Train.npz` / `Binary_Dev.npz` — binary classification (cat vs. non-cat)
- `MNIST.npz` — multiclass classification (handwritten digits 0–9)

## Author

Mukasa Simiyu