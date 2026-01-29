# Fashion-MNIST with CNN

A simple Convolutional Neural Network (CNN) implementation and experiments on the Fashion-MNIST dataset. This repository demonstrates data loading, model definition, training, evaluation, and inference for classifying Fashion-MNIST images.

## Contents
- Training and evaluation code (e.g., Jupyter notebook or Python scripts)
- Model definition using Keras/TensorFlow or PyTorch
- Instructions for reproducing results

> Note: File names and exact structure may vary. Update the commands below to match your actual filenames (for example `train.py`, `notebook.ipynb`, or `requirements.txt`).

## Requirements
- Python 3.8+
- TensorFlow (or PyTorch) — example uses Keras API
- Jupyter (optional, for notebooks)
- pip

A typical requirements.txt might include:
```text
tensorflow>=2.0
numpy
matplotlib
jupyter
```

## Setup
1. Clone the repository:
```bash
git clone https://github.com/kavinayac/Fashion-mnist-using-CNN.git
cd Fashion-mnist-using-CNN
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```
If there is no `requirements.txt`, install the packages manually:
```bash
pip install tensorflow numpy matplotlib jupyter
```

## Usage

### Run the notebook
If there is a notebook (e.g., `notebook.ipynb`), start Jupyter:
```bash
jupyter notebook
```
Open the notebook and run the cells to load the dataset, train the model, and view metrics/plots.

### Run a training script
If the repo contains a script (e.g., `train.py`) you can run:
```bash
python train.py --epochs 20 --batch-size 128
```
Adjust flags/parameters according to the script's CLI.

### Quick example (Keras)
A minimal Keras training loop (for reference):
```python
from tensorflow.keras import datasets, layers, models, utils

# Load data
(x_train, y_train), (x_test, y_test) = datasets.fashion_mnist.load_data()
x_train = x_train[..., None] / 255.0
x_test = x_test[..., None] / 255.0
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# Model
model = models.Sequential([
  layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
  layers.MaxPooling2D((2,2)),
  layers.Conv2D(64, (3,3), activation='relu'),
  layers.MaxPooling2D((2,2)),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(10, activation='softmax'),
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)

# Evaluate
model.evaluate(x_test, y_test)
```

## Model architecture
- Conv2D(32) → ReLU → MaxPool
- Conv2D(64) → ReLU → MaxPool
- Flatten → Dense(128) → ReLU → Dropout
- Dense(10) → Softmax

This simple CNN generally achieves strong performance on Fashion-MNIST; typical test accuracy depends on hyperparameters but often reaches ~90%+ with a small model and proper training.

## Reproducing results
1. Ensure dependencies are installed.
2. Run the data-preparation and training notebook/script.
3. Adjust hyperparameters (learning rate, epochs, batch size, regularization) for improved performance.
4. Save the best model and run evaluation/inference scripts on test images.

## Saving / Loading model
For Keras:
```python
model.save('fashion_cnn.h5')
# load
from tensorflow.keras.models import load_model
model = load_model('fashion_cnn.h5')
```