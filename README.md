# GradMan

Baby deep learning library.

### Tensor Operation
```python3
from gradman import Tensor

x = Tensor([[1.0,2.0,3.0]])
y = Tensor([[-3.0],[-2.0],[-1.0]])
z = x.dot(y)
print(z)
```

### Autograd engine
```python3
from gradman import Tensor

x = Tensor([[1,2,3]])
y = Tensor([[-3,5,3],[-2,3,5],[-1,9,8]])
z = Tensor([[4],[5],[6]])

result = x.matmul(y).matmul(z)
result.backward()
print(x.grad,y.grad,z.grad) 
```
