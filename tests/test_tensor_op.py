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


def test_add():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([9, 8, 7], requires_grad=True)

    r1 = a + b
    result = [-10, 20, -30]
    grad = result
    r1.backward(Tensor(grad))
    assert a.grad.data.tolist() == result and b.grad.data.tolist() == result

    a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor([9, 8, 7], requires_grad=True)
    r2 = a + b
    r2.backward(Tensor([[1, 2, 3], [4, 5, 6]]))
    assert a.grad.data.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert b.grad.data.tolist() == [5, 7, 9]

    a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor([[9, 8, 7]], requires_grad=True)
    r2 = a + b
    r2.backward(Tensor([[1, 2, 3], [4, 5, 6]]))
    assert a.grad.data.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert b.grad.data.tolist() == [[5, 7, 9]]
