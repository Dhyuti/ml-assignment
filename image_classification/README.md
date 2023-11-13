# Image Classifier for Fashion MNIST

This program trains a convolutional neural network (CNN) model on the Fashion MNIST dataset to classify images of clothing into 10 categories.

## Introduction

The model is trained on the Fashion MNIST dataset which contains 60,000 grayscale images of clothing items across 10 categories. 
It is compiled with the Adam optimizer and sparse categorical crossentropy loss. The model is evaluated on the validation set and saved to `image_classifier.model`.
To make predictions, load the saved model and pass preprocessed images to the `predict()` method. The output will be prediction probabilities for each class.

## Usage

Clone or download this repository to your local machine.
   - Refer to: [ml-assignments/README.md](../README.md)

## Dependencies

- Keras 
- TensorFlow
- OpenCV
- Numpy
- Matplotlib

## References

- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Keras Documentation](https://keras.io/)
