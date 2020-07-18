import numpy as np



class Dense:
    def __init__(self, input_array, units, output_units):
        self.input_array = input_array
        self.units = units
        self.weights = np.random.rand(units,output_units)
        self.result = 0
    def calculate(self):
        x1 = np.expand_dims(self.input_array, axis = 0)
        x2 = self.weights
        self.result = np.squeeze(np.matmul(x1,x2),axis=0)
        return self.result

