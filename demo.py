import gradman.nn as nn
from gradman import Tensor
from gradman.optim import GDE


class DemoModel(nn.Module):
    def __init__(self):
        super(DemoModel, self).__init__()
        self.l1 = nn.Linear(6,3)
        self.l2 = nn.Linear(3,1)

    def forward(self, i):
        return self.l2(self.l1(i))

class D2(nn.Module):
    def __init__(self):
        super(D2, self).__init__()
        self.model = DemoModel()
        
    def forward(self,i):
        return self.model(i)

model = D2()
optim = GDE()
for _ in range(10):
    y = model(Tensor([[1,2,3,3,4,6]]))
    y = (y - Tensor([2])).sum()
    print(y)
    y.backward()
    optim.step(model.parameters())
