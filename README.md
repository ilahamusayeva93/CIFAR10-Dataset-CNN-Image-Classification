# Convolutional Neural Network (CNN) for CIFAR-10 Image Classification

## Overview

This script demonstrates the process of training a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The script covers the following steps:

1. Downloading, extracting, and loading the CIFAR-10 image dataset using torchvision.
2. Displaying random batches of images in a grid using torchvision.utils.make_grid.
3. Creating a convolutional neural network with nn.Conv2d and nn.MaxPool2d layers.
4. Training the CNN and visualizing the losses and errors.
5. Understanding overfitting and implementing strategies to avoid it.
6. Generating predictions on single images from the test set.
7. Saving and loading the model for further purposes.

## How to Use

1. Download the CIFAR-10 dataset from [Kaggle](https://www.kaggle.com/c/cifar-10/data) and place it in the appropriate folders.

2. Run the script in a Python environment with PyTorch and torchvision installed.

3. Execute the script and follow the output for training and testing results.

## Script Structure

### 1. Data Loading and Preprocessing

- **Dataset:** Downloads and loads the CIFAR-10 dataset using torchvision.transforms for normalization.

- **Data Splitting:** Splits the training dataset into training and validation sets.

### 2. Model Definition

- **Convolutional Neural Network (CNN):** Defines a CNN architecture for image classification with three convolutional layers, batch normalization, ReLU activation, and max-pooling.

### 3. Training and Evaluation

- **Loss Function and Optimizer:** Specifies the CrossEntropyLoss as the loss function and Adam optimizer.

- **Training Loop:** Iterates over epochs, trains the model, and prints loss at regular intervals.

- **Validation:** Evaluates the model on the validation set and saves the best model based on validation accuracy.

- **Learning Rate Scheduler:** Adjusts the learning rate based on validation performance.

- **Early Stopping:** Implements early stopping if the model does not show improvement in validation accuracy.

### 4. Model Testing and Evaluation

- **Loading Best Model:** Loads the saved model with the best validation accuracy.

- **Test Accuracy:** Evaluates the model on the test set and prints the accuracy.

### 5. Image Prediction

- **Random Image Prediction:** Selects random images from the test set and displays the true labels along with model predictions.

### 6. Save and Load Model

- **Save Model:** Saves the trained model for future use.

- **Load Model:** Loads a pre-trained model.

## Author

- **Author:** Ilaha Musayeva
- **Date:** 11/05/2023


