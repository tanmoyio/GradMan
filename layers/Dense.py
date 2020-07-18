import numpy as np



class Dense:
    def __init__(self, output_units):
        self.input_array = None
        self.units = None
        self.output_units = output_units
        self.weights = []
        self.result = 0
        self.name = "Dense"


    def calculate(self):
        self.weights = np.random.rand(self.units,self.output_units)
        x1 = np.expand_dims(self.input_array, axis = 0)
        x2 = self.weights
        self.result = np.squeeze(np.matmul(x1,x2),axis=0)
        return self.result

