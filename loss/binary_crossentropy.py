import numpy as np


class binary_crossentropy:

    def calculate(p,y):
        return sum((y*np.log(p)) + ((1-y)*np.log(1-p)))
