import tensorflow as tf
from keras.datasets import fashion_mnist
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# Load the Fashion MNIST dataset
mnist = fashion_mnist
(training_images, training_labels), (val_images, val_labels) = mnist.load_data()

# Reshape the images to (28, 28, 1) and display the first image
training_images = training_images.reshape(60000, 28, 28, 1)
val_images = val_images.reshape(10000, 28, 28, 1)

plt.figure()
plt.imshow(training_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# Normalize the pixel values to a range of 0 to 1
training_images = training_images / 255.0
val_images = val_images / 255.0

# Create a Convolutional Neural Network (CNN) model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layersMaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train the model
model.fit(training_images, training_labels, validation_data=(val_images, val_labels), epochs=10, batch_size=10)

# Evaluate the model
loss, accuracy = model.evaluate(val_images, val_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

# Save and load the model
model.save('image_classifier.model')
model = tf.keras.models.load_model('image_classifier.model')

# Define class names for Fashion MNIST labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Validation Labels:")
for label in val_labels:
    print(class_names[label])

# Load and preprocess a new image for prediction
img = cv.imread('shoe1.png')
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img = cv.resize(img, (28, 28))
img = img.reshape(28, 28, 1)

plt.figure()
plt.imshow(img, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()

# Make a prediction on the new image
prediction = model.predict(np.array([img]))
index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')