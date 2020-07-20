import numpy as np


class Linear:
    def __init__(self):
        self.graph = []
        self.result = 0

    def add(self, layer):
        self.graph.append(layer)

    def summary(self):
        for layer in self.graph:
            print(layer.name + "\n |")

    def eval(self, inputs):
        self.result = inputs
        for layer in self.graph: 
            layer.input_array = self.result
            layer.units = layer.input_array.shape[0]
            try:
                if layer.weights == []:
                    layer.init_weights()
            except:
                pass
            self.result = layer.calculate()
        return self.result
            

        


