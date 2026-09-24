import math
import gzip
import struct
import numpy as np
import pandas as pd
import random
from PIL import Image
from PIL import ImageEnhance

# def preprocess(img, sharpness=1.5, contrast=1.3):
#     # Grayscale
#     img = img.convert('L')
#
#     # Sharpness
#     img = ImageEnhance.Contrast(img).enhance(contrast)
#     # Contrast
#     img = ImageEnhance.Sharpness(img).enhance(sharpness)
#
#     return img
#
# img = Image.open('input.jpeg')
# img = preprocess(img, 2, 2)
# img.show()

############################################################
# NN from scratch bitch 
# took 10-12 hours spanned over 3 days to get the hang of it
############################################################
class Value():
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = _children
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Value({self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
            
        out._backward = _backward

        return out

    def __radd__(self, other): # other + self
        return self + other

    def __pow__(self, other):
      assert isinstance(other, (int, float)), "only supporting int/float powers for now"
      out = Value(self.data**other, (self,), f'**{other}')

      def _backward():
          self.grad += other * (self.data ** (other - 1)) * out.grad
      out._backward = _backward

      return out

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad # out.grad due to chain rule
            other.grad += self.data * out.grad
            
        out._backward = _backward

        return out

    def __rmul__(self, other): # other * self
        return self * other

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self, ))
        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward

        return out;

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)

                topo.append(v)

        build_topo(self);

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = sum((w1*x1 for w1, x1 in zip(self.w, x)), self.b)
        out = act.tanh()

        return out

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
      params = []
      for neuron in self.neurons:
        ps = neuron.parameters()
        params.extend(ps)
      return params

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
      layers = []
      for layer in self.layers:
        a = layer.parameters()
        layers.extend(a)
      return layers

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

# Normalize
x_train = x_train.astype(np.float32) / 255.0

print(np.unique(y_train, return_counts=True))

# n = MLP(784, [32, 32, 62])

# for k in range(40):
#     total_loss = 0.0
#     for x, target in zip(x_train, y_train):
#         # Forward pass
#         outputs = n(x)
#         # One hotass target
#         target_vector = [0.0] * 62
#         target_vector[int(target)] = 1.0
#
#         loss = 0.0
#
#         for output, target_value in zip(outputs, target_vector):
#             loss += (output - target_value) ** 2
#
#         # Reset gradients
#         for p in n.parameters():
#             p.grad = 0.0
#
#         # Backward pass
#         loss.backward()
#
#         # Update pass
#         for p in n.parameters():
#             p.data += -0.05 * p.grad
#
#         total_loss += loss.data
#
#     print(k, total_loss)
#
# for x, y_true in zip(x_train[:10], y_train[:10]):
#
#     outputs = n(x)
#
#
#     prediction = max(
#         range(62),
#         key=lambda i: outputs[i].data
#     )
#
#     print(len(outputs))
#     print(outputs[:5])
#     print(
#         "predicted:", prediction,
#         "actual:", y_true
#     )

# print("parameters:", n.parameters(),
#     "size:", len(n.parameters()))
