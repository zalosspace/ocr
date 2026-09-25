import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from PIL import ImageEnhance
import tensorflow as tf

def preprocess(img, sharpness=1.5, contrast=1.3):
    # invert
    # img = ImageOps.invert(img)
    # Grayscale
    img = img.convert('L')

    # Sharpness
    # img = ImageEnhance.Contrast(img).enhance(contrast)
    # # Contrast
    # img = ImageEnhance.Sharpness(img).enhance(sharpness)

    return img

img = Image.open('7.jpg')
img = preprocess(img)

# Match EMNIST orientation
img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
img = img.transpose(Image.Transpose.ROTATE_90)

arr = np.array(img, dtype=np.float32) / 255.0

# BEFORE resizing
print("original processed:", arr.shape)
print("min/max:", arr.min(), arr.max())

small = Image.fromarray(
    (arr * 255).astype(np.uint8)
).resize((280, 280))

small.show()

# Resize
img = img.resize((28, 28))
img.show()

# Normalize
img = np.array(img, dtype=np.float32) / 255.0
# img = img.reshape(1, 784)
img = img.reshape(1, 28, 28, 1)

print(img.shape)

cnn = tf.keras.models.load_model("emnist_cnn_model.keras")

#################
# EMNIST MAPPING:
# 0-9: 0 to 9
# 10-35: A to Z
# 36-61: a to z
#################

def class_to_char(class_id):
    if class_id < 10:
        return str(class_id)
    elif class_id < 36:
        return chr(ord('A') + class_id - 10)
    else:
        return chr(ord('a') + class_id - 36)

# Predict
output = cnn.predict(img, verbose=0)[0]

class_id = np.argmax(output)

print("char:", class_to_char(class_id))
print("confidence:", output[class_id])

# top = np.argsort(output)[::-1][:10]
#
# for i in top:
#     print(i, class_to_char(i), output[i])
