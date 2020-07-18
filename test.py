from layers import Dense
from model import Linear
import numpy as np

a = np.array([1,2,3,1,3,4,6,9,1,5])


model = Linear()
model.add(Dense(10))
model.add(Dense(5))
model.add(Dense(3))
model.summary()

print(model.eval(a))



