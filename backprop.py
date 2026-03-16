import math
import numpy as np
import matplotlib.pyplot as plt

from graphviz import Digraph

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'})
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label="{%s | data=%.4f | grad=%.4f}" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            dot.node(name=uid + n._op, label=n._op)
            dot.edge(uid + n._op, uid)
    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)
    dot.render('data/graph.gv', format='svg', view=True)
    return dot

# Following along with Andrej Karpathy's "micrograd" implementation of backpropagation
# https://www.youtube.com/watch?v=VMj-3S1tku0

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None
        self.label = label
        self.grad = 0.0

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad = out.grad
            other.grad = out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad = other.data * out.grad
            other.grad = self.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad = (1 - t**2) * out.grad
        out._backward = _backward
        return out

def simple_backprop():
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a*b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'
    L.grad = 1.0
    f.grad = d.data
    d.grad = f.data
    c.grad = d.grad # Really, times 1, but still.
    e.grad = d.grad # Again, times 1...
    a.grad = e.grad * b.data
    b.grad = e.grad * a.data
    draw_dot(L)

    # Now that we have the gradients, we can do a simple parameter update step:
    rate = 0.01
    a.data += rate * a.grad
    b.data += rate * b.grad
    c.data += rate * c.grad
    f.data += rate * f.grad

    e = a*b
    d = e + c
    L = d * f

    print("After one step of gradient descent:")
    print(L.data)

def show_tanh():
    plt.plot(np.arange(-5,5,0.2), np.tanh(np.arange(-5,5,0.2)))
    plt.grid(True)
    plt.savefig('data/plot.png')

def tiny_nn_backprop():
    # Let's build a tiny neural network and backprop through it.
    x1 = Value(2.0, label='connection1')
    x2 = Value(0.0, label='connection2')

    w1 = Value(-3.0, label='weight1')
    w2 = Value(1.0, label='weight2')
    b = Value(6.8813735870195432, label='bias')

    x1w1 = x1 * w1; x1w1.label = 'connection1*weight1'
    x2w2 = x2 * w2; x2w2.label = 'connection2*weight2'
    x1w1xx2w2 = x1w1 + x2w2; x1w1xx2w2.label = 'connection1*weight1+connection2*weight2'
    n = x1w1xx2w2 + b; n.label = 'neuron'
    o = n.tanh(); o.label = 'output'

    o.grad = 1.0
    n.grad = 1 - o.data**2 # Again: TECHNICALLY, times 1 from the composition rule, but we can ignore that.
    x1w1xx2w2.grad = n.grad
    b.grad = n.grad
    x1w1.grad = x1w1xx2w2.grad
    x2w2.grad = x1w1xx2w2.grad
    x1.grad = x1w1.grad * w1.data
    w1.grad = x1w1.grad * x1.data
    x2.grad = x2w2.grad * w2.data
    w2.grad = x2w2.grad * x2.data

    draw_dot(o)

tiny_nn_backprop()