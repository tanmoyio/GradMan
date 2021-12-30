import numpy as np


class Tensor:
    def __init__(self, data, _ctx=(), _op=""):
        self.data = (
            np.array(data, dtype=np.float32)
            if not isinstance(data, np.ndarray)
            else data
        )
        self.grad = 0

        self._backward = lambda x: None
        self._ctx = set(_ctx)
        self._op = _op

    def __repr__(self):
        return f"<Tensor {self.data!r}>"

    @property
    def shape(self):
        return self.data.shape

    @classmethod
    def eye(cls, dim, **kwargs):
        return cls(np.eye(dim).astype(np.float32), **kwargs)

    def __matmul__(self, i):
        o = Tensor(np.matmul(self.data, i.data), (self, i), _op="matmul")

        def _backward(input_grad):
            input_grad = (
                np.array([[input_grad]])
                if not isinstance(input_grad, np.ndarray)
                else input_grad
            )
            self.grad = input_grad @ i.data.swapaxes(-2, -1)
            i.grad = self.data.swapaxes(-2, -1) @ input_grad

        o._backward = _backward
        return o

    matmul = __matmul__

    def __add__(self, i):
        o = Tensor(self.data + i.data, (self, i), _op="add")

        def _backward(input_grad):
            input_grad = (
                np.array([[input_grad]])
                if not isinstance(input_grad, np.ndarray)
                else input_grad
            )
            if input_grad.shape == self.shape:
                self.grad = input_grad
            else:
                self.grad = np.reshape(
                    np.array([[sum(k) for k in input_grad]]), self.shape
                )
            if input_grad.shape == i.shape:
                i.grad = input_grad
            else:
                i.grad = np.reshape(np.array([[sum(k) for k in input_grad]]), i.shape)

        o._backward = _backward
        return o

    add = __add__

    def __sub__(self, i):
        o = Tensor(self.data - i.data, (self, i), _op="sub")

        def _backward(input_grad):
            input_grad = (
                np.array([[input_grad]])
                if not isinstance(input_grad, np.ndarray)
                else input_grad
            )
            if input_grad.shape == self.shape:
                self.grad = input_grad
            else:
                self.grad = np.reshape(
                    np.array([[sum(k) for k in input_grad]]), self.shape
                )
            if input_grad.shape == i.shape:
                i.grad = -input_grad
            else:
                i.grad = np.reshape(-np.array([[sum(k) for k in input_grad]]), i.shape)

        o._backward = _backward
        return o

    sub = __sub__

    dot = matmul

    def backward(self):
        graph, checked = [], set()

        def build_graph(n):
            if n not in checked:
                checked.add(n)
                for branch in n._ctx:
                    build_graph(branch)
                graph.append(n)

        build_graph(self)

        self.grad = 1
        for n in reversed(graph):
            n._backward(n.grad)
