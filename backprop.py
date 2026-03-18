import math
from multiprocessing.dummy import connection
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
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * other**-1

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
                
        return topo

class Neuron:
    def __init__(self, nin):
        self.w = [Value(np.random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(np.random.uniform(-1,1))

    def __call__(self, x):
        act = sum((weight*connection for weight, connection in zip(self.w, x)), self.b)
        return act.tanh()
    
    def parameters(self):
        return self.w + [self.b]

# Note the number of outputs (nout) is the number of neurons in the layer, and the number of inputs (nin) is the number of connections
# to each neuron.
class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outs = [neuron(x) for neuron in self.neurons]
        return outs if len(outs) > 1 else outs[0]

    def parameters(self):
        return [param for neuron in self.neurons for param in neuron.parameters()]

# Simple Multi-layer Perceptron (MLP) with one hidden layer. The number of inputs is the number of inputs to the first layer, and the
# number of outputs (nout) is a list of the number of neurons in each layer (including the number of outputs at the end). For example,
# if nouts is [4, 4, 1], then there are two layers with 4 neurons, and there is one output.
class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x # will return the last layer's output.
    
    def parameters(self):
        return [param for layer in self.layers for param in layer.parameters()]

# An early version of the code that manually did the backpropagation steps, before we implemented the automatic backpropagation in the
# Value class. This is just to show how the gradients are calculated, and to check that our implementation of the backward() method is
# correct.
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

# Just a quick check to see what the tanh function looks like, since we're using it as an activation function in our tiny neural network.
def show_tanh():
    plt.plot(np.arange(-5,5,0.2), np.tanh(np.arange(-5,5,0.2)))
    plt.grid(True)
    plt.savefig('data/plot.png')

# Let's build a tiny neural network and backprop through it.
def tiny_nn_backprop():
    x1 = Value(2.0, label='connection1')
    x2 = Value(0.0, label='connection2')

    w1 = Value(-3.0, label='weight1')
    w2 = Value(1.0, label='weight2')
    b = Value(6.8813735870195432, label='bias')

    x1w1 = x1 * w1; x1w1.label = 'connection1*weight1'
    x2w2 = x2 * w2; x2w2.label = 'connection2*weight2'
    x1w1xx2w2 = x1w1 + x2w2; x1w1xx2w2.label = 'connection1*weight1+connection2*weight2'
    n = x1w1xx2w2 + b; n.label = 'neuron'
    # o = n.tanh(); o.label = 'output'
    e = (2*n).exp(); e.label = 'exp(2*neuron)'
    o = (e - 1) / (e + 1); o.label = 'output'

    o.backward()
    draw_dot(o)

# Simple test of the MLP and stepping forward through it.
def fwd_through_mlp():
    x = [2.0, 3.0, -1.0]
    mlp = MLP(3, [4, 4, 1])
    result = mlp(x)
    print(result)
    draw_dot(result)

# Let's test a simple binary classifier:
inputs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    # [1.0, -1.0, -1.0],
    # [-1.0, 1.0, 1.0],
    # [-1.0, -1.0, 1.0],
    # [-1.0, 1.0, -1.0],
    # [-2.0, 2.0, 1.5],
]
desired_outs = [1.0, -1.0, -1.0, 1.0] # Desired outputs for the above inputs.

mlp = MLP(3, [4, 4, 1])

def forward_pass():
    y_outputs = [mlp(x) for x in inputs]
    print("Outputs:")
    for out, deso in zip(y_outputs, desired_outs):
        print(f"out: {out.data:.4f} vs desired: {deso}")

    loss = sum((y_out - y_ground_truth)**2 for y_ground_truth, y_out in zip(desired_outs, y_outputs))
    return loss

def backward_pass():
    loss.backward()

def check_first_neuron():
    print("First param data:")
    print(mlp.layers[0].neurons[0].w[0].data)
    print("First param bias:")
    print(mlp.layers[0].neurons[0].b.data)
    print("First param grad:")
    print(mlp.layers[0].neurons[0].w[0].grad)

def zero_grads():
    for p in mlp.parameters():
        p.grad = 0.0

def nudge_parameters():
    for p in mlp.parameters():
        p.data += -0.1 * p.grad

for k in range(30):
    print(f"## PASS {k+1} ##")
    loss = forward_pass()
    zero_grads()
    backward_pass()
    nudge_parameters()
    check_first_neuron()
    print(f"-- LOSS={loss.data:.4f}")
