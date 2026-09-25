import gzip
import struct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train_sample_count = 20
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

x_train = x_train.reshape(train_sample_count, 28, 28)
x_train = np.rot90(x_train, axes=(1,2))
x_train = np.fliplr(x_train)

import matplotlib.pyplot as plt

def class_to_char(class_id):
    if class_id < 10:
        return str(class_id)
    elif class_id < 36:
        return chr(ord('A') + class_id - 10)
    else:
        return chr(ord('a') + class_id - 36)

for i in range(20):
    plt.imshow(x_train[i], cmap="gray")
    plt.xlabel(class_to_char(y_train[i]))
    plt.show()

