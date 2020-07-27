import numpy as np


class gradient_descent:
    def __init__(self,lr=0.0001):
        self.lr = lr

    def optimize(weights,grad):
        return (weights - grad)
