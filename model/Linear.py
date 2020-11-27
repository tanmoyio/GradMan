import cursor
import numpy as np
import itertools
import sys
sys.path.insert(1, '../MudGrad/')
from MudGrad import autograd



class Linear:
    def __init__(self):
        self.graph = []
        self.result = 0
        self.optimizer = None
        self.loss = None



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



    def compile(self,optimizer,loss):
        self.optimizer = optimizer
        self.loss = loss



    def fit(self,input_batch,label_batch,epochs,batch_size=32):
        cursor.hide()
        number_of_batchs = input_batch.shape[0]/batch_size
        input_batch=np.array_split(input_batch,number_of_batchs)
        label_batch=np.array_split(label_batch,number_of_batchs)
        for epoch in range(epochs):
            print("\nEpoch " + str(epoch+1) + "/" + str(epochs))
            for (i,j) in enumerate(zip(input_batch,label_batch)):
                pred_label_batch = np.array([self.eval(k) for k in j[0]])
                ground_label_batch = j[1]
                counter = 0
                loss_val = self.loss.calculate(pred_label_batch,ground_label_batch)[-1]
                loss_grad = self.loss.compute_grad(pred_label_batch,ground_label_batch)
                autograd(self,loss_grad)
                grids = 50
                print(str(i+1)+"/"+str(int(number_of_batchs))+" ["+int(i/number_of_batchs*grids)*"="+ int((number_of_batchs-i)/number_of_batchs*grids)*" " +"] "+" loss:"+str(loss_val),end="\r")
                dw = self.graph[0].weights







