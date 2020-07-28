import numpy as np



class Conv2D:
    def __init__(self,units,input_shape=None,kernel_size=(3,3),stride=2,activation=None):
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
        self.activation = activation


    def init_weights(self):
        self.weights = np.random.uniform(low=-5.0, high=5.0, size=(self.units, self.kernel_size[0], self.kernel_size[1]))

    def normalize(self,input_array):
        return (input_array/(np.amax(input_array)+np.amin(input_array)))*5.0

    
    def compute_grad(self,input_array):
        return

    def optimize(self,optimizer,loss_grad):
        if self.activation != None:
            activation_grad = self.activation.compute_grad(self.input_array)
            self.weights = optimizer.optimize(self.weights,loss_grad*activation)
        else:
            self.weights = optimizer.optimize(self.weights,loss_grad)


    def calculate(self): 
        i, j, row, col = 0, 0, 0, 0
        sum_conv = 0
        if len(self.input_array.shape) ==2:
            self.input_array = np.expand_dims(self.input_array, axis = 2)
        
        while i<= self.input_array.shape[0] and i+self.kernel_size[0] <= self.input_array.shape[0]:
            while j<= self.input_array.shape[1] and j+self.kernel_size[1] <= self.input_array.shape[1]:
                for weight in self.weights:
                    sum_conv = np.sum([np.multiply(slice_mat[i:i+self.kernel_size[0], j:j+self.kernel_size[1]],weight) for slice_mat in self.input_array.reshape(self.input_array.shape[::-1])])
                    self.result.append(sum_conv)
                    sum_conv = 0
                j = j + self.stride
                if i == 0:
                    col = col + 1
            i = i + self.stride
            j = 0
            row = row + 1

        self.result = np.array(self.result).reshape(row,col,self.units)
        #self.result = self.normalize(self.result)
        if self.activation != None:
            self.result = self.activation.calculate(self.result)
        temp_result = self.result
        self.result = []
        self.output_shape = temp_result.shape
        return temp_result
        


