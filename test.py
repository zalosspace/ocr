import gzip
import struct
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

test_sample_count = 20
with gzip.open("data/emnist-byclass-test-images-idx3-ubyte.gz", "rb") as f:
    # get file metadata, 4 byte each
    magic, n, rows, cols = struct.unpack(">IIII", f.read(16))

    data = f.read(test_sample_count * rows * cols)
    x_test = np.frombuffer(data, dtype=np.uint8)
    x_test = x_test.reshape(test_sample_count, rows*cols) 

with gzip.open("data/emnist-byclass-test-labels-idx1-ubyte.gz", "rb") as f:
    # get file metadata, 4 byte each
    magic, n = struct.unpack(">II", f.read(8))

    data = f.read(test_sample_count)
    y_test = np.frombuffer(data, dtype=np.uint8)

print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# Normalize
x_test = x_test.astype(np.float32) / 255.0

nn = tf.keras.models.load_model("emnist_model.keras")

outputs = nn.predict(x_test[:test_sample_count], verbose=0)
predictions = np.argmax(outputs, axis=1)
for prediction, actual in zip(predictions, y_test[:test_sample_count]):
    print("predicted:", prediction, "actual:", int(actual))
