# GradMan [![CI](https://github.com/tanmoyio/GradMan/actions/workflows/lint.yml/badge.svg)](https://github.com/tanmoyio/GradMan/actions/workflows/lint.yml)

Baby deep learning library.

### Tensor Operation
```python3
from gradman import Tensor

x = Tensor([[1.0,2.0,3.0]])
y = Tensor([[-3.0],[-2.0],[-1.0]])
z = x @ y
print(z)
```

### Autograd engine
```python3
from gradman import Tensor

x = Tensor([[1,2,3]])
y = Tensor([[-3,5,3],[-2,3,5],[-1,9,8]])
z = Tensor([[4],[5],[6]])

result = x @ y @ z
result.backward()
print(x.grad,y.grad,z.grad) 
```

### Model and Training Loop
```python3
import gradman.nn as nn
from gradman import Tensor, Module
from gradman.optim import gradient_descent


class Model(Module):
    def __init__(self):
        self.l1 = nn.Linear(3,2)
        self.l2 = nn.Linear(2,1)

    def forward(self,i):
        return self.l2(self.l1(i))


# input for the model
x = Tensor([[1.0, 2.0, 3.0]])

# creating the model object
m = Model()

# creating the optimizer
optimizer = gradient_descent

for i in range(5):
    r = m(x)

    # I haven't added any loss function yet, lets just backprop from here
    r.backward()
    m.optimize(optimizer)
```
