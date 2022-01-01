import os
import sys

os.chdir(os.path.realpath(os.path.dirname(__name__)))
sys.path.append(".")

from gradman import Tensor  # noqa


def test_sum():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([9, 8, 7], requires_grad=True)

    r1 = a.sum()
    r1.backward()
    assert a.grad.data.tolist() == [1, 1, 1]

    r2 = b.sum()
    r2.backward(Tensor(43))
    assert b.grad.data.tolist() == [43, 43, 43]
