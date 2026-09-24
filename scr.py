import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageEnhance
import tensorflow as tf

def preprocess(img, sharpness=1.5, contrast=1.3):
    # Grayscale
    img = img.convert('L')

    # Sharpness
    img = ImageEnhance.Contrast(img).enhance(contrast)
    # Contrast
    img = ImageEnhance.Sharpness(img).enhance(sharpness)

    return img

img = Image.open('p.jpg')
img = preprocess(img, 2, 2)

# Resize
img = img.resize((28, 28))

# Match EMNIST orientation
img = img.rotate(270, expand=True)
img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
img.show()


# Normalize
img = np.array(img, dtype=np.float32) / 255.0
img = img.reshape(1, 784)

print(img.shape)

nn = tf.keras.models.load_model("emnist_model.keras")

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
output = nn.predict(img, verbose=0)
prediction = np.argmax(output)
character = class_to_char(prediction)

print("predicted:", character)
print("class_id:", prediction)
