# Convex Optimization 

Every deep learning problem is optimization problem. The goal is finding parameter values to minimize an objective function. We have provided a solution of a simple convex optimization problem with `GradMan` library.

#### 1. Initialization 
First step, if you haven't installed `GradMan`, install it with `pip install -U gradman`

#### 2. Creating objective function/hypothesis with tensor operations

Here we have created `Σ[(x - a)^2 + 5]` where `a = [2, 1, 0]` and x will be a 3 dim `Tensor`. Its clear that the minimum value of this function can be 15, and at that specific condition the value of `x` will be `[-2, -1, 0]`

```python3
from gradman import Tensor

def hypothesis(x):
    a = Tensor([2, 1, 0])
    return (((x + a) * (x + a)) + Tensor(5)).sum()
```

#### 3. Initialize a random `x` value

```python3
x = Tensor([100, 200, 300], requires_grad=True)
```

#### 4. Create an optimizer
You can always use `gradman.optim` but its easy to create a basic optimizer from scratch.

```python3
def optimize(dy_dx, lr=0.1):
    ''' Gradient descent optimizer '''
    return x - (Tensor(lr) * dy_dx)
```

#### 5. Convex optimization loop

```python3
# to check the progress of optimization
import tqdm 

# number of epochs
EPOCH = 50

for _ in tqdm.tqdm(range(EPOCH)):
    y = hypothesis(x)
    y.backward()
    x = optimize(x.grad)

print('\nMinimum value of the hypothesis for any given x is: ', y)
print('Value of x so that the hypothesis will be minimum is: ', x)
```
It will output something like this 
```
100%|█████████████████████████████████████████████████████████████████████| 50/50 [00:00<00:00, 7306.64it/s]

Minimum value of the hypothesis for any given x is:  <Tensor (15.000044816382914, requires_grad=True)>
Value of x so that the hypothesis will be minimum is:  <Tensor ([-1.99854421 -0.99713123  0.00428174], requires_grad=True)>
```
Easy right? Now you can try to increase the complexity of the equation and you can also increase the dimension of the parameters. Have fun, finding global minima 🐣
