import numpy as np



def compute_grad(f, weights, delta=0.00001):
    return (f(weights+delta)-f(weights-delta))/2*delta



def autograd(model, loss_grad, delta=0.000001):
    for layer in model.graph[::-1]:
        try:
            layer = layer.weights + delta
            d1 = layer.calculate()
            layer = layer.weights - (2*delta)
            d2 = layer.calculate()
            df = (d1-d2)/(2*delta)
            layer.weights = model.optimizer(layer.weights, df, loss_grad)

        except:
            pass


    
    






















