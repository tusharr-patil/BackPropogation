"""
in order to do back prop I need to do forward pass first

a * b = c 
a -> * -> c 
b
"""
import networkx as nx
import matplotlib.pyplot as plt


class Value:
    def __init__(self, data, parent=(), op="", label=""):
        self.data = data
        self.grad = 0
        self.op = op
        self.parent = parent
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), "+")

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), "*")

    def __repr__(self):
        return f"Value(data={self.data})"


# w * x + b = L

w = Value(4)
w.label = "w"

x = Value(5)
x.label = "x"

b = Value(2)
b.label = "b"

wx = w * x
wx.label = "wx"

L = wx + b
L.label = "L"

L.grad = 1

"""
G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 1)])

# Visualizing the graph
nx.draw(G, with_labels=True, font_weight="bold")
plt.show()
"""
