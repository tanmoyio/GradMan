import numpy as np


class Sigmoid:

    def calculate(input_array):
        print(input_array)
        print(1.0/(1.0+np.exp(-input_array)))
        return(1.0/(1.0+np.exp(-input_array)))


    def compute_grad(input_array):
        grad = sum(1/(1+np.exp(-input_array)))/input_array.shape[0]
        return grad*(1-grad)
