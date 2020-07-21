import numpy as np


class Linear:
    def __init__(self):
        self.graph = []
        self.result = 0

    def add(self, layer):
        self.graph.append(layer)
        

    def summary(self):
        temp_shape = self.graph[0].input_shape
        self.eval(np.ones(temp_shape))
        print("\n__Network Architecture__")
        for index,layer in enumerate(self.graph):
            print(index,layer.name,layer.output_shape,layer.number_of_params)
        print(f"Total number of parameters: {sum([i.number_of_params for i in self.graph]):,}\n\n")

    def eval(self, inputs):
        self.result = inputs
        if self.graph[0].input_shape == None:
            print(f"input_shape in {self.graph[0].name} is missing")
            exit()
        if self.graph[0].input_shape != inputs.shape:
            print(f"input array shape must be {self.graph[0].input_shape} not {inputs.shape}")
            exit()
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
            

        


