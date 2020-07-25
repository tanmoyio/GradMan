import time
import pickle
import numpy as np
from layers import Dense, Flatten, Conv2D, MaxPooling2D
from loss import binary_crossentropy
from model import Linear
from activation import Sigmoid

start_time = time.time()
image = np.random.rand(64,64)

model = Linear()
model.add(Conv2D(64,input_shape=(64,64)))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(128))
model.add(MaxPooling2D((2,2)))
model.add(Flatten())
model.add(Dense(128))
model.add(Dense(64))
model.add(Dense(1))
model.summary()

model.eval(image)
input_data = open("input_data.pickle","rb")
input_data = np.array(pickle.load(input_data))

label = open("label.pickle","rb")
label = np.expand_dims(np.array(pickle.load(label)),axis=1)
print(input_data.shape)
gradient_descent = "gradient_descent"
model.compile(optimizer=gradient_descent, loss=binary_crossentropy)
model.fit(input_data,label,epochs=20,batch_size=16)


end_time = time.time()
print(f"Total execution time::  {end_time - start_time}")











'''
model2 = Linear()
model2.add(Dense(64, input_shape=(50,)))
model2.add(Dense(32))
model2.add(Dense(1,activation=Sigmoid))
model2.summary()
print(model2.eval(np.random.rand(50,)))
'''













