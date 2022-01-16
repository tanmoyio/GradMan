from typing import List, Optional, Union

import numpy as np

from gradman.ctg import ContextGraph
from gradman.ops import add_, matmul_, mul_, neg_, slice_, sum_


class Tensor:

    Tensorable = Union[int, float, list, np.ndarray]

    def __init__(
        self,
        data: Tensorable,
        requires_grad: bool = False,
        _ctx: List[ContextGraph] = None,
    ) -> None:

        self._data = self.UnTensored(data)
        self.requires_grad = requires_grad
        self._ctx = _ctx or []
        self.grad: Optional["Tensor"] = None
        self.shape = self.data.shape

        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, t: np.ndarray):
        self._data = t
        self.grad = None

    def UnTensored(self, t: Tensorable) -> np.ndarray:
        if isinstance(t, np.ndarray):
            return t
        else:
            return np.array(t)

    def __repr__(self) -> str:
        return f"<Tensor ({self.data}, requires_grad={self.requires_grad})>"

    def __getitem__(self, idx) -> "Tensor":
        o, requires_grad, _ctx = slice_(self, idx)
        return Tensor(o, requires_grad, _ctx)

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))

    def backward(self, grad: "Tensor" = None) -> None:
        assert self.requires_grad, "Called backward() on non-requires-grad Tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("`grad` must be specified for non 0 tensor")

        self.grad.data += grad.data

        for c in self._ctx:
            c.tensor.backward(Tensor(c.grad_fn(grad.data)))

    def sum(self) -> "Tensor":
        """Sum of elements of a tensor.

        Returns
        -------
        "Tensor"

        """
        o, requires_grad, _ctx = sum_(self)
        return Tensor(o, requires_grad, _ctx)

    def add(t1: "Tensor", t2: "Tensor") -> "Tensor":
        """Addition of two tensors"""
        o, requires_grad, _ctx = add_(t1, t2)
        return Tensor(o, requires_grad, _ctx)

    __add__ = add

    def mul(t1: "Tensor", t2: "Tensor") -> "Tensor":
        """Multiplication of two tensors"""
        o, requires_grad, _ctx = mul_(t1, t2)
        return Tensor(o, requires_grad, _ctx)

    __mul__ = mul

    def neg(self) -> "Tensor":
        """Negative of a tensor"""
        o, requires_grad, _ctx = neg_(self)
        return Tensor(o, requires_grad, _ctx)

    __neg__ = neg

    def sub(t1: "Tensor", t2: "Tensor") -> "Tensor":
        """Subtraction of two tensors"""
        return t1 + (-t2)

    __sub__ = sub

    def matmul(t1: "Tensor", t2: "Tensor") -> "Tensor":
        """Tensor Matrix Multiplication"""
        o, requires_grad, _ctx = matmul_(t1, t2)
        return Tensor(o, requires_grad, _ctx)

    __matmul__ = matmul
