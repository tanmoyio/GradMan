import numpy as np



class MaxPooling2D:
    def __init__(self, pool_size=(2,2)):
        self.input_array = None
        self.pool_size = pool_size
        self.result = []
        self.name = "MaxPooling2D"
        self.output_shape = None
        self.number_of_params = 0



    def calculate(self):
        input_shape = self.input_array.shape
        i,j,row,col = 0,0,0,0
        result = []
        while i <= input_shape[0] and i+self.pool_size[0]<=input_shape[0]:
            while j<=input_shape[1] and j+self.pool_size[1]<=input_shape[1]:
                result.append(np.max(self.input_array[i:i+self.pool_size[0],j:j+self.pool_size[1]],axis=(0,1)))
                j = j + self.pool_size[1]
                if row == 0:
                    col = col + 1
            i = i + self.pool_size[0]
            j = 0
            row = row+1
        self.result = result
        if len(input_shape) ==3:
            self.result = np.array(self.result).reshape(row,col,input_shape[2])
        else:
            self.result = np.array(self.result).reshape(row,col)
        self.output_shape=self.result.shape
        return self.result
