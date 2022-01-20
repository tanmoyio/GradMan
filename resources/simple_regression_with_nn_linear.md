# Simple Regression with `gradman.nn.Linear`

Regression is one of the most common problems in ML, there are several ways to solve and solving this is easy comparing to classifications as it doesn't require non-linearity.
In this documentation we will try to solve this using `Linear` module from `gradman`. This is not a ideal way of solving linear regression, try this as a fun experiment.

### Dataset preparation 
Lets use [California housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) from `scikit-learn`.
```python3
from gradman import Tensor
from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing()

# creating inputs and labels 
inputs = dataset.data     # shape-> (20640, 8)
labels = dataset.targets  # shape-> (20640,)

# normalizing the dataset
"""There are some other ways to normalizing the dataset, but if you are seeking fun, try this"""
inputs, labels = inputs/inputs.max(), labels/labels.max()

# Tensorify 
inputs, labels= Tensor(inputs), Tensor(labels)

# creating a data generator
def generator(inputs, labels, batch_size):

    collected = []
    num_batch =int((inputs.shape[0]/batch_size))
    
    for i in range(num_batch):
        start = i * batch_size
        end = start + batch_size
        yield inputs[start:end], labels[start:end]
```

The training might fail due to recursion limit so you might want to add this following snippet too. We will fix that in future versions.
```python3
import sys, threading
sys.setrecursionlimit(10**7)
threading.stack_size(2**27)
```

### Create the model
So the structure of the model is simple, its taking 8 dimensions and outputing 1 dimension with `Linear` layer.

```python3
import gradman.nn as nn

class CalNet(nn.Module):
    """California Housing price prediction model"""
    
    def __init__(self):
        super(RegressionNet, self).__init__()
        self.l1 = nn.Linear(8,1)

    def forward(self, i):
        return self.l1(i)
```
lets initialize the model and the optimizer

```python3
from gradman.optim import GDE

model = RegressionNet()
optim = GDE(lr=0.001)
```

### MSELoss from scratch
Lets create a loss function from scratch. This way you can create your own loss function and experiment. 
```python3
def MSELoss(pred, label):
    """Mean Square Error loss: 1/n Σ(predictions - labels)^2"""
    return ((pred - label)**2).sum()/Tensor(pred.shape[0])
```

### Training loop

```python3
EPOCH = 20
batch_size = 32
generator_length = sum(1 for _ in generator(inputs, labels, batch_size))

for i in range(EPOCH):
    avg_loss = 0
    
    for data in generator(inputs, labels, batch_size):
        i, o = data
        loss = MSELoss(model(inputs), out)
        loss.backward()
        optim.step(model.parameters())
        avg_loss += loss.data
        
    print("loss: ", avg_loss)
```
