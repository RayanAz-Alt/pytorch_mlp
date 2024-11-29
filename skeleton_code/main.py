import pandas as pd
import numpy as np
from hw1_q1 import Perceptron, LinearModel

### LOAD DATA
data = np.load("../intel_landscapes.npz")

### SPLIT DATA
train_images = data["train_images"]
test_images = data["test_images"]
val_images = data["val_images"]

train_labels = data["train_labels"].squeeze()
test_labels = data["test_labels"].squeeze()
val_labels = data["val_labels"].squeeze()

### DATA CARACTERISTICS
n_classes = len(np.unique(train_labels))
n_features = train_images.shape[1]*train_images.shape[2]*train_images.shape[3]

### RESHAPE IMAGES TO 1D ARRAY
train_images = train_images.reshape(train_images.shape[0], -1) / 255.0
val_images = val_images.reshape(val_images.shape[0], -1)/ 255.0
test_images = test_images.reshape(test_images.shape[0], -1)/ 255.0


# ### Q1.1 Train Perceptron
model = Perceptron(n_classes=n_classes, n_features=n_features)

print(train_labels)
for epoch in range(100):
    model.train_epoch(train_images, train_labels)  
    train_accuracy = model.evaluate(train_images, train_labels)  
    val_accuracy = model.evaluate(val_images, val_labels)  
    print(f"Epoch {epoch + 1}: Training accuracy = {train_accuracy:.4f}")
    print(f"Epoch {epoch + 1}: Validation accuracy = {val_accuracy:.4f}")
    print("")