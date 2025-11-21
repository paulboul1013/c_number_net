# c_number_net

A simple neural network implementation in C for handwritten digit recognition using the MNIST dataset format.

## Features

- **Neural Network**: 3-layer feedforward network (784 input nodes, 128 hidden nodes, 10 output nodes)
- **Training**: Train the model on CSV-formatted MNIST data
- **Testing**: Evaluate model accuracy on test samples
- **Interactive Drawing**: Real-time digit recognition using SDL2 drawing window

## Requirements

- GCC compiler
- SDL2 development libraries
- Math library (-lm)

## Installation

Install SDL2 development libraries:

```bash
# Ubuntu/Debian
sudo apt-get install libsdl2-dev
```

## Building

```bash
make
```

## Usage

The program supports three modes:

### Training Mode

Train the neural network on training data:

```bash
./main train
# or
./main t
```

This will train the model for 3 epochs and save it to `model.bin`.

### Test Mode

Test the model on samples from `train.csv`:

```bash
./main test
# or
./main test 10  # Test first 10 samples (default: 5)
# or
./main e
```

### Draw Mode

Open an interactive drawing window for real-time digit recognition:

```bash
./main draw
# or
./main d
```

**Drawing Window Controls:**
- Left mouse button: Draw on canvas
- C key: Clear canvas
- ESC key: Exit

## Data Format

The program expects a CSV file named `train.csv` with the following format:
- First column: Label (0-9)
- Remaining 784 columns: Pixel values (0-255) representing a 28x28 image

## Model File

Trained models are saved as `model.bin` in binary format. The file contains:
- Hidden layer weights (784 × 128)
- Output layer weights (128 × 10)
- Hidden layer biases (128)
- Output layer biases (10)

## Architecture

- **Input Layer**: 784 nodes (28×28 pixels)
- **Hidden Layer**: 128 nodes with sigmoid activation
- **Output Layer**: 10 nodes (digits 0-9) with sigmoid activation
- **Learning Rate**: 0.1
- **Optimization**: Backpropagation with gradient descent

