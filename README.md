# GradMan [![CI](https://github.com/tanmoyio/GradMan/actions/workflows/lint.yml/badge.svg)](https://github.com/tanmoyio/GradMan/actions/workflows/lint.yml)[![Tensor-Test](https://github.com/tanmoyio/GradMan/actions/workflows/tensor-test.yml/badge.svg)](https://github.com/tanmoyio/GradMan/actions/workflows/tensor-test.yml)

<img src="https://imgur.com/qdTcXdY.png" height=200>

Baby deep learning library. This library doesn't do much in terms of solving complex models. Check [gradman resources](https://github.com/tanmoyio/GradMan/tree/master/resources). We are in the way of replacing the mathematical computations with c++ backend also cuda for GPU support. Mail me if you want to join GradMan's acccelarated computing group. 

### Install
```
pip install gradman
```
### Gradman Tensor Operations 🥚

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

### Use `gradman.nn.Module` to create complex neural network 🐥
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

### Training loop 🐙
Simpler than `torch`

```python3
from gradman.optim import GDE

model = BabyModel()
optim = GDE(lr=0.001)

for _ in range(EPOCH):
    '''Dataloading, batching module will be added in future versions'''
    '''also the `criterion` is a dummy loss func. We will add those in future versions. But you can always create loss functions from basic tensor operations.'''
    
    y = model(inputs)
    loss = criterion(y, labels)
    y.backward()
    optim.step(model.parameters())
```
`gradman` doesn't do `model.zero_grad()`. Why? Whenever the contents of a `Tensor` object being changed it will invalidate the gradients by itself and initialize fresh zero gradients.


### Build from source 🐛
```
git clone https://github.com/tanmoyio/GradMan 
cd GradMan
pip install poetry
poetry build
cat pyproject.toml | grep "version"
cd dist/
pip install gradman-<version>-none-any.whl
```

### Contribute 🍯

Before contributing, you must know the purpose of this library. I haven't made this library to create SOTA models with it but to preserve the core mathematical foundation of deep learning. 

#### Spaces where you can contribute. 
1. Zero level - Writing Tensor operations (Backward needed)
2. Creating High level API/Layers for Deep Neural Network (No need to do backward)
3. Writing unit tests
4. Creating examples of the library, models and sharing.

Run these two lines of command to pass the unit tests. 

```
make lint
python -m pytest --import-mode=append tests -v
```

Current version of gradman uses basic tensor operation wrapped arround standard numpy. But I am also working on a gpu and RISCV version of `gradman`. Mail me if you are interested of being part of core developer of `gradman`.
