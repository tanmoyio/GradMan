import numpy as np


class Flatten:
    def __init__(self):
        self.input_array = None
        self.result = 0
        self.name = "Flatten"

    def calculate(self):
        self.result = np.array([element for subarray in self.input_array for element in subarray])
        return self.result

