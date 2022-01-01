from typing import Callable, List, NamedTuple, Optional, Union

import numpy as np

Tensorable = Union[int, float, list, np.ndarray]


def UnTensored(t: Tensorable) -> np.ndarray:
    if isinstance(t, np.ndarray):
        return t
    else:
        return np.array(t)


class ContextGraph(NamedTuple):
    tensor: "Tensor"
    grad_fn: Callable[[np.ndarray], np.ndarray]


class Tensor:
    def __init__(
        self,
        data: Tensorable,
        requires_grad: bool = False,
        _ctx: List[ContextGraph] = None,
    ) -> None:

        self.data = UnTensored(data)
        self.requires_grad = requires_grad
        self._ctx = _ctx or []
        self.grad: Optional["Tensor"] = None
        self.shape = self.data.shape

        if self.requires_grad:
            self.zero_grad()

    def __repr__(self) -> str:
        return f"<Tensor ({self.data}, requires_grad={self.requires_grad})>"

    def zero_grad(self) -> None:
        self.grad = Tensor(np.zeros_like(self.data))

    def backward(self, grad: "Tensor" = None) -> None:
        assert self.requires_grad, "Called backward() on non-requires-grad Tensor"

        if grad is None:
            if self.shape == ():
                grad = Tensor(1)
            else:
                raise RuntimeError("`grad` must be specified for non 0 tensor")

        self.grad.data += grad.data

        for c in self._ctx:
            c.tensor.backward(Tensor(c.grad_fn(grad.data)))

    def sum(self) -> "Tensor":
        def _sum(t: Tensor) -> Tensor:
            data = t.data.sum()
            requires_grad = t.requires_grad

            if requires_grad:

                def grad_fn(grad: np.ndarray) -> np.ndarray:
                    return grad * np.ones_like(t.data)

                _ctx = [ContextGraph(t, grad_fn)]
            else:
                _ctx = []

            return Tensor(data, requires_grad, _ctx)

        return _sum(self)
