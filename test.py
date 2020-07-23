import time
from layers import Dense, Flatten, Conv2D, MaxPooling2D
from model import Linear
import numpy as np
from activation import Sigmoid

start_time = time.time()
image = np.random.rand(64,64,12)

model = Linear()
model.add(Conv2D(64,input_shape=(64,64,12),activation=Sigmoid))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128,activation=Sigmoid))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(512,activation=Sigmoid))
model.add(Dense(256))
model.add(Dense(64,activation=Sigmoid))
model.summary()

print(model.eval(image))

end_time = time.time()
print(f"Total execution time::  {end_time - start_time}")




model2 = Linear()
model2.add(Dense(64, input_shape=(50,)))
model2.add(Dense(32))
model2.add(Dense(1,activation=Sigmoid))
model2.summary()
print(model2.eval(np.random.rand(50,)))














