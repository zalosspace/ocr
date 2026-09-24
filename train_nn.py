import gzip
import struct
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

train_sample_count = 100000
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

# import matplotlib.pyplot as plt
#
# plt.imshow(x_train[3].reshape(28, 28), cmap="gray")
# plt.show()

# Normalize
x_train = x_train.astype(np.float32) / 255.0

nn = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(784,)),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),

    tf.keras.layers.Dense(62, activation='softmax')
    ])

nn.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

history = nn.fit(
        x_train, y_train,
        epochs=200, batch_size=32, validation_split=0.2, verbose=1
        )

plt.figure()
plt.plot(history.history["loss"], label="loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.show()


plt.figure()
plt.plot(history.history["accuracy"], label="accuracy")
plt.plot(history.history["val_accuracy"], label="val_accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()


outputs = nn.predict(x_train[:10], verbose=0)

predictions = np.argmax(outputs, axis=1)

for prediction, actual in zip(predictions, y_train[:10]):
    print(
            "predicted:", prediction,
            "actual:", int(actual)
            )

nn.save("emnist_model.keras")
