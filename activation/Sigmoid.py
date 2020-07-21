import numpy as np


class Sigmoid:

    def calculate(input_array):
        input_array = input_array.astype("float32")
        return(1./(1.+np.exp(-input_array)))

