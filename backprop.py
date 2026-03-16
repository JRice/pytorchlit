import math
import numpy as np
import matplotlib.pyplot as plt

# Following along with Andrej Karpathy's "micrograd" implementation of backpropagation
# https://www.youtube.com/watch?v=VMj-3S1tku0

class Value:
    def __init__(self, data):
        self.data = data

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data)
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data)
        return out
    
a = Value(2.0)
b = Value(-3.0)
c = Value(15.0)
print(a * b + c)
