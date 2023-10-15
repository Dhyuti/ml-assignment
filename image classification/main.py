import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import datasets, layers, models

# Load the Fashion MNIST dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (val_images, val_labels) = mnist.load_data()

# Preprocess the data
training_images = training_images.reshape(60000, 28, 28, 1)
val_images = val_images.reshape(10000, 28, 28, 1)
training_images = training_images / 255.0
val_images = val_images / 255.0

# Display the first training image
plt.figure()
plt.imshow(training_images[0][:, :, 0], cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()

# Create a convolutional neural network model
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(20, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile and train the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, validation_data=(val_images, val_labels))

# Evaluate the model
loss, accuracy = model.evaluate(val_images, val_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Save and load the model
model.save('image_classifier.model')
model = models.load_model('image_classifier.model')

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Load an image for prediction
img = cv.imread('shoe.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()

# Preprocess the image for prediction
img = img[:, :, 0]  # Convert to grayscale
img = cv.resize(img, (28, 28))
img = img.reshape(1, 28, 28, 1)

# Make a prediction
prediction = model.predict(img / 255)
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')
