from gradman import Tensor


class Optimizer:
    def step(self, params):

        for layer in params:
            if not isinstance(layer.parameters()[-1], Tensor):
                self.step(layer.parameters())
            else:
                p_new = [self.f(i) for i in layer.parameters()]
                layer.update(p_new)


class GDE(Optimizer):
    def __init__(self, lr: float = 0.001):
        super(GDE, self).__init__()

        self.lr = Tensor(lr)

    def f(self, p):
        return p - (self.lr * p.grad)
