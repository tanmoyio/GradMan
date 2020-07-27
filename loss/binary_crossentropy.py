import numpy as np


class binary_crossentropy:

    def calculate(p,y):
        return sum((y*np.log(p)) + ((1-y)*np.log(1-p)))/p.shape[0]

    def compute_grad(p,y):
        return sum((-y/p)+((1-y)/(1-p)))/y.shape[0]
