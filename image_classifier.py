import numpy as np  # For numerical operations in Python
import tensorflow as tf  # Deep learning library used to build and train the CNN
from tensorflow.keras import datasets, layers, models  # type: ignore # Import necessary modules from Keras
from tensorflow.keras.utils import to_categorical  # type: ignore # Function to convert labels to one-hot encoding
import matplotlib.pyplot as plt  # type: ignore # Library for data visualization

# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Preprocessing: Normalize the pixel values to be between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Reshape the images to (28, 28, 1) as they are grayscale
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Convert the labels to one-hot encoded format
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Build the CNN Model
model = models.Sequential()

# First Convolutional Layer with 32 filters of size (3,3), ReLU activation, and input shape (28,28,1)
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))  # Max pooling layer to reduce spatial dimensions

# Second Convolutional Layer with 64 filters of size (3,3) and ReLU activation
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Third Convolutional Layer with 64 filters of size (3,3) and ReLU activation
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the 3D output to 1D and add a Dense Layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))  # Fully connected layer with 64 neurons

# Output layer with 10 neurons (for 10 digit classes) and softmax activation
model.add(layers.Dense(10, activation='softmax'))

# Compile the model with Adam optimizer and categorical crossentropy loss function
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model with training data, 5 epochs, batch size 64, and validation data
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_data=(test_images, test_labels))

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc * 100:.2f}%")

# Make predictions on test images
predictions = model.predict(test_images)
print(f"Prediction for first test image: {np.argmax(predictions[0])}")

# Display the first test image along with the predicted label
plt.imshow(test_images[0].reshape(28, 28), cmap='gray')  # Corrected 'comp' to 'cmap'
plt.title(f"Predicted Label: {predictions[0].argmax()}")
plt.show()
