from layers import Dense
import numpy as np



a = np.array([1,2,3])


print(Dense(a, units=3, output_units=2).calculate())
