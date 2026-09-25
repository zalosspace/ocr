import gzip
import struct
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

train_sample_count = 20000
with gzip.open("data/emnist-byclass-train-images-idx3-ubyte.gz", "rb") as f:
    # get file metadata, 4 byte each
    magic, n, rows, cols = struct.unpack(">IIII", f.read(16))

    data = f.read(train_sample_count * rows * cols)
    x_train = np.frombuffer(data, dtype=np.uint8)
    x_train = x_train.reshape(train_sample_count, rows*cols) 

with gzip.open("data/emnist-byclass-train-labels-idx1-ubyte.gz", "rb") as f:
    # get file metadata, 4 byte each
    magic, n = struct.unpack(">II", f.read(8))

    data = f.read(train_sample_count)
    y_train = np.frombuffer(data, dtype=np.uint8)

print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)

# Normalize
x_train = x_train.astype(np.float32) / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)

cnn = models.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),

    layers.Dense(128, activation="relu"),
    layers.Dense(62, activation="softmax")
])

cnn.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

history = cnn.fit(
        x_train, y_train,
        epochs=15, batch_size=32, validation_split=0.2, verbose=1
        )
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

test_loss, test_acc = cnn.evaluate(x_train,  y_train, verbose=2)
print(test_acc)

cnn.save("emnist_cnn_model.keras")
