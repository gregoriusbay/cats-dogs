import numpy as np
import matplotlib.pyplot as plt
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D

import warnings
warnings.filterwarnings('ignore')

# Set the image batch on every process
batch_size = 16

# Set the image shape (Width, Height, Channels)
image_shape = (150, 150, 3)

# Create the ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range=30,  # Rotate the image for 30 degrees
                               width_shift_range=0.1,  # Shift the picture width by 10% maximum
                               height_shift_range=0.1,  # Shift the picture height by 10% maximum
                               rescale=1/255,  # Rescale the image by normalizing it.
                               shear_range=0.2,  # Cutting away part of the image by maximum of 20%)
                               zoom_range=0.2,  # Zoom in by 20% maximum
                               horizontal_flip=True,  # Allow horizontal flipping
                               fill_mode='nearest'  # Fill in missing pixels with the nearest filled value
                              )

# Set the train image generator by using image_gen variable
train_image_gen = image_gen.flow_from_directory(r'D:\Data Knowledge\Computer-Vision-with-Python\DATA\CATS_DOGS\train',
                                                target_size=image_shape[:2],
                                                batch_size=batch_size,
                                                class_mode='binary')

# Set the test image generator by using image_gen variable
test_image_gen = image_gen.flow_from_directory(r'D:\Data Knowledge\Computer-Vision-with-Python\DATA\CATS_DOGS\test',
                                               target_size=image_shape[:2],
                                               batch_size=batch_size,
                                               class_mode='binary')

# Import the Sequential model type
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(filters=32, kernel_size=(3, 3), input_shape=(150, 150, 3), activation='relu'))  # 32 filters of size 3x3, ReLU activation
model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling with a 2x2 window to reduce spatial dimensions

# Second Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # 64 filters for more complex features
model.add(MaxPooling2D(pool_size=(2, 2)))  # Additional max pooling to further reduce spatial dimensions

# Third Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))  # 64 filters with 3x3 kernel, no change in input_shape needed here
model.add(MaxPooling2D(pool_size=(2, 2)))  # Max pooling to reduce spatial dimensions further

# Flatten layer to transform 2D matrices to a 1D vector
model.add(Flatten())

# Fully Connected (Dense) Layer with ReLU Activation
model.add(Dense(128))             # Dense layer with 128 neurons
model.add(Activation('relu'))      # ReLU activation for non-linearity

# Dropout layer for regularization (prevents overfitting)
model.add(Dropout(0.5))            # Randomly drops 50% of neurons during training to prevent overfitting

# Output Layer for Binary Classification
model.add(Dense(1))                # Output layer with 1 neuron (binary classification)
model.add(Activation('sigmoid'))   # Sigmoid activation to output probability between 0 and 1

# Compile the model
model.compile(
    loss='binary_crossentropy',    # Binary crossentropy for binary classification
    optimizer='adam',              # Adam optimizer for adaptive learning rate
    metrics=['accuracy']           # Use accuracy as the metric to evaluate model performance
)

# Fitting the model
results = model.fit(train_image_gen, epochs=100,
                    steps_per_epoch=150,
                    validation_data=test_image_gen,
                    validation_steps=12)

# Save the model
model.save('cats_dogs', save_format="h5")