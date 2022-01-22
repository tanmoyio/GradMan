# Simple Regression with `gradman.nn.Linear`

Regression is one of the most common problems in ML, there are several ways to solve and solving this is easy comparing to classifications as it doesn't require non-linearity.
In this documentation we will try to solve this using `Linear` module from `gradman`. This is not a ideal way of solving linear regression, try this as a fun experiment.

### Dataset preparation 
Lets use [California housing dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) from `scikit-learn`.
```python3
from gradman import Tensor
import pandas as pd
from sklearn.datasets import fetch_california_housing

dataset = fetch_california_housing()

# creating inputs and labels 
inputs = pd.DataFrame(dataset.data)     # shape-> (20640, 8)
labels = pd.DataFrame(dataset.target)  # shape-> (20640,)

# normalizing the dataset
"""There are some other ways to normalizing the dataset, but if you are seeking fun, try this"""

# preparing the normalizer
def normalizer(x):
    return (x - x.mean()) / x.std()
    
# normalization
inputs = inputs.apply(lambda x: normalizer(x))
labels = labels.apply(lambda x: normalizer(x))


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
        super(CalNet, self).__init__()
        self.l1 = nn.Linear(8,1)

    def forward(self, i):
        return self.l1(i)
```
lets initialize the model and the optimizer

```python3
from gradman.optim import GDE

model = CalNet()
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
import tqdm 

EPOCH = 10
batch_size = 16
generator_length = sum(1 for _ in generator(inputs, labels, batch_size))

history = []
for i in tqdm.tqdm(range(EPOCH)):
    avg_loss = 0
    
    for data in generator(inputs, labels, batch_size):
        i, o = data
        loss = MSELoss(model(i), o)
        loss.backward()
        optim.step(model.parameters())
        avg_loss += loss.data
        
    history.append(f"Epoch:{i}, loss: {avg_loss/generator_length}")
    
print(history)
```
Output
```
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 10/10 [16:42<00:00, 100.26s/it]

['Epoch:0, loss: 1.0708987647046646', 
'Epoch:1, loss: 0.44320831007158107', 
'Epoch:2, loss: 0.42144465194942876', 
'Epoch:3, loss: 0.4148460952723904', 
'Epoch:4, loss: 0.41056946676311107', 
'Epoch:5, loss: 0.40720832080746344', 
'Epoch:6, loss: 0.40454978511430495', 
'Epoch:7, loss: 0.4024258263336942', 
'Epoch:8, loss: 0.4007285866164422', 
'Epoch:9, loss: 0.3993680843491425']
```
So you can see how the convergence is going. Congratulations, you have successfully trained a linear regression model with `gradman.nn.Linear` 🐣
