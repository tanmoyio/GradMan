import gradman.nn as nn
from gradman import Tensor

class Optimizer:
    def step(self, params):

        for l in params:
            if not isinstance(l.parameters()[-1], Tensor):
                self.step(l.parameters())
            else:
                p_new = [self.f(i) for i in l.parameters()]
                l.update(p_new)

class GDE(Optimizer):
    def __init__(self, lr=0.001):
        super(GDE, self).__init__()

        self.lr = Tensor(lr)

    def f(self, p):
        return p - (self.lr * p.grad)

    
