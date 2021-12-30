import os, sys
os.chdir(os.path.realpath(os.path.dirname(__name__)))
sys.path.append(".")
from gradman import Tensor
import numpy as np

x = Tensor([[1.5,-2.8,3.3]])
y = Tensor([[-3.8],[-21.2],[-18.5]])

def matmul():
    return round(float((x @ y).data[0,0]), 2)

def add():
    return round(float((x + y).data[1,2]), 2)

def test_matmul():
    assert matmul() == round(-7.389, 2)
    assert add() == round(-17.900002, 2)
