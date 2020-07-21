import numpy as np



class Conv2D:
    def __init__(self,units,input_shape=None,kernel_size=(3,3),stride=2):
        self.input_array = None
        self.kernel_size = kernel_size
        self.units = units
        self.stride = stride
        self.weights = []
        self.result = []
        self.name = "Conv2D"
        self.input_shape = input_shape
        self.output_shape = None
        self.number_of_params = units * kernel_size[0] * kernel_size[1]


    def init_weights(self):
        self.weights = np.random.rand(self.units, self.kernel_size[0], self.kernel_size[1])


    def calculate(self): 
        i, j, row, col = 0, 0, 0, 0
        sum_conv = 0
        while i<= self.input_array.shape[1] and i+self.kernel_size[1] <= self.input_array.shape[1]:
            while j<= self.input_array.shape[0] and j+self.kernel_size[0] <= self.input_array.shape[0]:
                sum_conv = [np.sum(np.multiply(self.input_array[i:i+3,j:j+3],weight)) for weight in self.weights]
                self.result.append(sum_conv)
                j = j + self.stride
                if i == 0:
                    col = col + 1
            i = i + self.stride
            j = 0
            row = row + 1

        self.result = np.array(self.result).reshape(row,col,self.units)
        temp_result = self.result
        self.result = []
        self.output_shape = temp_result.shape
        return temp_result

