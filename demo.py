from gradman import Tensor
import numpy as np

'''
a = Tensor([1.0, 0.5, 0.8], requires_grad=True)
b = a.sum()
b.backward()
print(a.grad)
'''

a = Tensor(np.random.randn(10,10))
print(a[:5, :2])
