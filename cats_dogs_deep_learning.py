import matplotlib.pyplot as plt
from keras import layers, models, optimizers, regularizers, preprocessing, callbacks

import warnings
warnings.filterwarnings('ignore')

image_shape = (150, 150, 3)
batch_size = 64

train_image_gen = preprocessing.image_dataset_from_directory(r'D:\Data Knowledge\Computer-Vision-with-Python\DATA\CATS_DOGS\train',
                                                             batch_size=batch_size,
                                                             image_size=image_shape[:2],
                                                             label_mode='binary'
                                                            )

test_image_gen = preprocessing.image_dataset_from_directory(r'D:\Data Knowledge\Computer-Vision-with-Python\DATA\CATS_DOGS\test',
                                                             batch_size=batch_size,
                                                             image_size=image_shape[:2],
                                                             label_mode='binary'
                                                            )

# Define the model with L2 regularization
model = models.Sequential()

# Data Augmentation
model.add(layers.RandomRotation(0.0833))
model.add(layers.RandomFlip("horizontal_and_vertical"))
model.add(layers.RandomZoom(0.2))

# Normalize the input
model.add(layers.Rescaling(1./255))

# Layers for Modelling
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=image_shape, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)))
model.add(layers.Activation('relu'))

# Dropout for regularization
model.add(layers.Dropout(0.5))

# Final layer for binary classification
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.Adam(learning_rate=1e-4),
              metrics=['accuracy'])

model.summary()

# Early stopping to prevent overfitting
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

results = model.fit(train_image_gen, epochs=100,
                    validation_data=test_image_gen, 
                    callbacks=[early_stopping])

# Plot accuracy
plt.plot(results.history['accuracy'], label='accuracy')
plt.plot(results.history['val_accuracy'], label='val_accuracy')
plt.show()

model.save('cats_dogs', save_format="h5")