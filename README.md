# Digit Classification using CNN 


## Overview

This project implements an end-to-end deep learning pipeline for digit classification using the Street View House Numbers (SVHN) dataset. The goal is to classify digits (0–9) from real-world images by applying convolutional neural networks, data augmentation and regularization techniques to achieve high accuracy while preventing overfitting.


## Objectives

- Load and preprocess the SVHN image dataset
- Normalize, reshape, and encode image data for CNN input
Apply data augmentation to improve generalization
- Implement and compare a Simple CNN and AlexNet
- Evaluate models using accuracy, precision, recall, F1-score, loss, and training efficiency
- Analyze trade-offs between model complexity and performance

## Tech Stack

- Python
- TensorFlow / Keras
- NumPy
- scikit-learn
- Matplotlib & Seaborn
- Jupyter Notebook


## Data Preprocessing

- Converted images from uint8 to float32
- Normalized pixel values to the [0, 1] range
- Checked and cleaned NaN values
- One-hot encoded digit labels for softmax classification
- Split training data into training and validation sets (90/10)

## Data Augmentation

To improve generalization and reduce overfitting, real-time data augmentation was applied using ImageDataGenerator:
- Small rotations
- Width and height shifts
- Zoom variations
- Brightness adjustments
- No horizontal flipping (to preserve digit semantics)
Augmentation generates new image variations each epoch, creating a virtually infinite training set without increasing storage requirements.



## Model Architectures

### Simple CNN
- Two convolutional layers with ReLU activation
- Max pooling layers for spatial reduction
- Fully connected layer with dropout
- Lightweight and computationally efficient

### AlexNet 
Deep convolutional architecture with:
- Multiple convolution layers
- Batch normalization
- Max pooling
- Two large fully connected layers
- Higher representational power for complex features



## Evaluation Metrics

Both models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Cross-entropy loss
- Training time per sample
- Confusion matrices
- Training and validation curves

## Summary

- AlexNet slightly outperformed Simple CNN due to deeper feature extraction
- Simple CNN was significantly faster and more computationally efficient


---

This project was completed as part of an assignment for CS6140: Machine Learning at Northeastern University.