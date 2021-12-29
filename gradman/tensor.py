import numpy as np

class Tensor:
    def __init__(self, data, _ctx=(), _op=''):
        self.data = np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data
        self.grad = 0

        self._backward = lambda x: None
        self._ctx = set(_ctx)
        self._op=_op

    def __repr__(self):
        return f"<Tensor {self.data!r}>"

    @property
    def shape(self): return self.data.shape

    @classmethod
    def eye(cls, dim, **kwargs):
        return cls(np.eye(dim).astype(np.float32), **kwargs)

    def matmul(self, i):
        o = Tensor(np.matmul(self.data,i.data), (self,i), _op='matmul')

        def _backward(input_grad):
            input_grad = np.array([[input_grad]]) if input_grad not isinstance(np.ndarray) else input_grad
            self.grad = input_grad @ i.data.swapaxes(-2, -1)
            i.grad = self.data.swapaxes(-2, -1) @ input_grad
                
        o._backward = _backward
        return o

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
