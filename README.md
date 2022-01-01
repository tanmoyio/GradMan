# GradMan [![CI](https://github.com/tanmoyio/GradMan/actions/workflows/lint.yml/badge.svg)](https://github.com/tanmoyio/GradMan/actions/workflows/lint.yml)[![Tensor-Test](https://github.com/tanmoyio/GradMan/actions/workflows/tensor-test.yml/badge.svg)](https://github.com/tanmoyio/GradMan/actions/workflows/tensor-test.yml)
Baby deep learning library

### Install
```
pip install gradman
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
