from layers import Dense
from layers import Flatten
from model import Linear
import numpy as np

a = np.array([
              [1,4,5,9],
              [8,7,5,7],
              [7,9,5,7]
])

model = Linear()
model.add(Flatten())
model.add(Dense(12))
model.add(Dense(8))
model.add(Dense(4))
model.summary()

print(model.eval(a))



