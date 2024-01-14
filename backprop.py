import graph


# Equivalent to Tensor in pytorch
class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._parent = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    # add op
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    # substraction is also additon ;)
    def __sub__(self, other):
        return self + (-other)

    #  unary op -> neg
    def __neg__(self):
        return -1 * self

    # mul op
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    # pow op
    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f"**{other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward

        return out

    # div is just another form of multiplication
    def __truediv__(self, other):
        return self * other**-1

    def back_propogation(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._parent:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# w1 * x1 - w2 / x2 + b

# Forward Pass
w1 = Value(7)
w1.label = "w1"

x1 = Value(5)
x1.label = "x1"

w2 = Value(8)
w2.label = "w2"

x2 = Value(7)
x2.label = "x2"

b = Value(2)
b.label = "b"

w1x1 = w1 * x1
w1x1.label = "w1x1"

w2x2 = w2 / x2
w2x2.label = "w2x2"

L = w1x1 - w2x2 + b
L.label = "L"

# Back Prop
L.grad = 1
L.back_propogation()

graph.draw_dot(L)
