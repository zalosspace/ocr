import numpy as np
import tensorflow as tf
import tkinter as tk

nn = tf.keras.models.load_model("emnist_model.keras")

def class_to_char(class_id):
    if class_id < 10:
        return str(class_id)
    elif class_id < 36:
        return chr(ord('A') + class_id - 10)
    else:
        return chr(ord('a') + class_id - 36)

def savePosn(event):
    global lastx, lasty
    lastx, lasty = event.x, event.y

def addLine(event):
    # create a line between last mouse_pos and curr_mouse_pos
    canvas.create_line(lastx, lasty, event.x, event.y)
    savePosn(event)

# Canvas
GRID = 28
SCALE = 10

root = tk.Tk()

canvas = tk.Canvas(root, width=GRID*SCALE, height=GRID*SCALE, bg="black")
canvas.pack()

pixels = np.zeros((GRID, GRID), dtype=np.float32)

def draw(event):
    x = event.x // SCALE
    y = event.y // SCALE

    if 0 <= x < GRID and 0 <= y < GRID:
        # row x col
        pixels[y, x] = 1.0
        canvas.create_rectangle(
                x*SCALE,
                y*SCALE,
                (x+1)* SCALE,
                (y+1)* SCALE,
                fill="white",
                outline="white"
                )

# press left mouse btn
canvas.bind("<Button-1>", draw)
# drag mouse while holding lmb
canvas.bind("<B1-Motion>", draw)

# Funtions
def predict():
    img = pixels.copy()

    # Match EMNIST orientation
    img = np.fliplr(img)
    img = np.rot90(img)

    # Normalize
    img = img.reshape(1, 784)

    print(img.shape)

    nn = tf.keras.models.load_model("emnist_model.keras")
    
    output = nn.predict(img, verbose=0)
    prediction = np.argmax(output)
    character = class_to_char(prediction)

    print("predicted:", character)
    print("class_id:", prediction)

    result.config(
        text=f"Prediction: {character}\n")

def clear():
    global pixels

    pixels.fill(0)
    canvas.delete("all")
    result.config(text="Prediction: -")

# Buttons
buttons = tk.Frame(root)
buttons.pack(pady=10)

predict_button = tk.Button(buttons, text="PREDICT", command=predict)
predict_button.pack(side="left", padx=5)

clear_button = tk.Button(buttons, text="CLEAR", command=clear)
clear_button.pack(side="left", padx=5)

result = tk.Label(
        root, 
        text="Prediction: -",
        font=("Arial", 20)
        )

result.pack(pady=10)

root.mainloop()
