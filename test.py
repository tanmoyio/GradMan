import time
from layers import Dense, Flatten, Conv2D
from model import Linear
import numpy as np

start_time = time.time()
image = np.random.rand(64,64,3)

model = Linear()
model.add(Conv2D(64,input_shape=(64,64,3)))
model.add(Flatten())
model.add(Dense(512))
model.add(Dense(256))
model.add(Dense(64))
model.summary()

print(model.eval(image))

print(model.eval(image))
end_time = time.time()
print(f"Total execution time::  {end_time - start_time}")


