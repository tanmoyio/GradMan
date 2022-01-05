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


def test_mul():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([9, 8, 7], requires_grad=True)
    r1 = a * b
    r1.backward(Tensor([-2, -3, 5]))
    assert a.grad.data.tolist() == [-18, -24, 35] and b.grad.data.tolist() == [
        -2,
        -6,
        15,
    ]

    a = Tensor([[1, 2, 3], [4, 5, 7]], requires_grad=True)
    b = Tensor([9, 8, 7], requires_grad=True)
    r2 = a * b
    r2.backward(Tensor([[-1, 2, -3], [-1, 2, -3]]))
    assert a.grad.data.tolist() == [
        [-9, 16, -21],
        [-9, 16, -21],
    ] and b.grad.data.tolist() == [-5, 14, -30]


def test_neg():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = -a
    b.backward(Tensor([4, -5, 6]))
    assert a.grad.data.tolist() == [-4, 5, -6]


def test_sub():
    a = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    b = Tensor([7, 8, 9], requires_grad=True)
    c = a - b
    c.backward(Tensor([[2, 3, 4], [2, 3, 4]]))
    assert a.grad.data.tolist() == [[2, 3, 4], [2, 3, 4]] and b.grad.data.tolist() == [
        -4,
        -6,
        -8,
    ]
