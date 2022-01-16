# GradMan [![CI](https://github.com/tanmoyio/GradMan/actions/workflows/lint.yml/badge.svg)](https://github.com/tanmoyio/GradMan/actions/workflows/lint.yml)[![Tensor-Test](https://github.com/tanmoyio/GradMan/actions/workflows/tensor-test.yml/badge.svg)](https://github.com/tanmoyio/GradMan/actions/workflows/tensor-test.yml)
Baby deep learning library



### Install
```
pip install gradman
```
### Gradman Tensor Operations

Just like `numpy.ndarray` operations `gradman` tensor supports mathematical operations. 

```python3
from gradman import Tensor

a = Tensor([[2.0, 0.3, 0.5]], requires_grad=True)
a = Tensor([[9.0], [0.1], [0.8]], requires_grad=True)
print(a @ b)
```

### BabyGrad Engine 🐣
```python3
from gradman import Tensor

a = Tensor([1.0, 0.5, 0.8], requires_grad=True)
b = a.sum()
b.backward()
print(a.grad)
```
```
$ <Tensor ([1. 1. 1.], requires_grad=False)>
```

### Use `gradman.nn.Module` to create complex neural network
```python3
import gradman.nn as nn
from gradman import Tensor

class BabyModel(nn.Module):
    def __init__(self):
        super(BabyModel, self).__init__()
        self.l1 = nn.Linear(6,3)
        self.l2 = nn.Linear(3,1)

    def forward(self, i):
        return self.l2(self.l1(i))
        
model = BabyModel()
out = model(Tensor([1, 2, 3, 4, 5, 6]))
print(out)
```
It makes everything easy. Still there is always option of creating your own nn operations from scratch, and the `BabyGrad` engine will handle the backprop.

### Build from source
```
git clone https://github.com/tanmoyio/GradMan 
cd GradMan
pip install poetry
poetry build
cat pyproject.toml | grep "version"
cd dist/
pip install gradman-<version>-none-any.whl
```
