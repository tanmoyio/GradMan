# GradMan

Baby deep learning library.

### Tensor Operation
```python3
from gradman import Tensor

x = Tensor([[1.0,2.0,3.0]])
y = Tensor([[-3.0],[-2.0],[-1.0]])
z = x.matmul(y)
print(z)
```
