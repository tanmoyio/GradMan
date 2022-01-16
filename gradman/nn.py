import numpy as np

from gradman import Tensor


class Module:
    def _forward(self, i):
        return self.forward(i)

    __call__ = _forward

    def parameters(self):
        return [getattr(self, i) for i in self.__dict__]


class Linear(Module):
    def __init__(self, idim: int, odim: int):
        super(Linear, self).__init__()

        self.idim = idim
        self.odim = odim

        self.w = Tensor(np.random.randn(idim, odim), requires_grad=True)
        self.b = Tensor(np.random.randn(), requires_grad=True)

    def forward(self, i: Tensor):
        return (i @ self.w) + self.b

    __call__ = forward

    def parameters(self):
        return [self.w, self.b]

    def update(self, p_new):
        self.w, self.b = p_new
