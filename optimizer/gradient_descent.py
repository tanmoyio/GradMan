import numpy as np



class gradient_descent:
    def __init__(self,lr=0.01):
        self.lr = lr



    def optimize(weights,grad,derivative):
        return weights - 0.0004*grad*derivative
