import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph
from IPython.display import SVG, display


class Value:
    def __init__(self, data, parent=(), op="", label=""):
        self.data = data
        self.grad = 0
        self.op = op
        self.parent = set(parent)
        self.backward = lambda: None
        self.label = label

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        next = Value(self.data + other.data, (self, other), "+")

        def backward():
            self.grad += 1.0 * next.grad  # chain rule
            other.grad += 1.0 * next.grad

        next.backward = backward
        return next

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        next = Value(self.data * other.data, (self, other), "*")

        def backward():
            self.grad += other.data * next.grad
            other.grad += self.data * next.grad

        next.backward = backward
        return next

    def __repr__(self):
        return f"Value(data={self.data})"

    def back_propogation(self):
        topo = []
        vis = set()

        def build_topo(v):
            if v not in vis:
                vis.add(v)
                for neighbour in v.parent:
                    build_topo(neighbour)
                topo.append(v)

        build_topo(self)
        for node in reversed(topo):
            node.backward()


# Draw the Graph
def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.parent:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # LR = left to right

    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(
            name=uid,
            label="{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad),
            shape="record",
        )
        if n.op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name=uid + n.op, label=n.op)
            # and connect this node to it
            dot.edge(uid + n.op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    with open("output_graph.svg", "wb") as f:
        f.write(dot.pipe(format="svg"))


# Neuron
# wx = w * x
# wx + b = L

# Forward Pass
w = Value(7)
w.label = "w"

x = Value(5)
x.label = "x"

b = Value(2)
b.label = "b"

wx = w * x
wx.label = "wx"

L = wx + b
L.label = "L"

# Back Prop
L.grad = 1
L.back_propogation()

draw_dot(L)
