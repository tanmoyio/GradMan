from typing import List, Tuple

import numpy as np

from gradman.ctg import ContextGraph

"""Collection of all Tensor supported operations"""


def sum_(t: "Tensor") -> Tuple[np.ndarray, bool, List[ContextGraph]]:
    """Sum of elements of a Tensor

    Parameters
    ----------
    t : 'Tensor'
        t

    Returns
    -------
    Tuple[np.ndarray, bool, List[ContextGraph]]

    """
    o = t.data.sum()
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(t.data)

        _ctx = [ContextGraph(t, grad_fn)]
    else:
        _ctx = []

    return o, requires_grad, _ctx


def add_(t1: "Tensor", t2: "Tensor") -> Tuple[np.ndarray, bool, List[ContextGraph]]:
    """Addition of two tensors

    Parameters
    ----------
    t1 : "Tensor"
        t1
    t2 : 'Tensor'
        t2

    Returns
    -------
    Tuple[np.ndarray, bool, List[ContextGraph]]

    """

    o = t1.data + t2.data

    requires_grad = t1.requires_grad or t2.requires_grad
    _ctx: List[ContextGraph] = []

    if t1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:

            for _ in range(grad.ndim - t1.data.ndim):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        _ctx.append(ContextGraph(t1, grad_fn1))

    if t2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:

            for _ in range(grad.ndim - t2.data.ndim):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        _ctx.append(ContextGraph(t2, grad_fn2))

    return o, requires_grad, _ctx


def mul_(t1: "Tensor", t2: "Tensor") -> Tuple[np.ndarray, bool, List[ContextGraph]]:
    o = t1.data * t2.data

    requires_grad = t1.requires_grad or t2.requires_grad
    _ctx: List[ContextGraph] = []

    if t1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            grad = grad * t2.data

            for _ in range(grad.ndim - t1.data.ndim):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t1.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        _ctx.append(ContextGraph(t1, grad_fn1))

    if t2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            grad = grad * t1.data

            for _ in range(grad.ndim - t2.data.ndim):
                grad = grad.sum(axis=0)

            for i, dim in enumerate(t2.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)

            return grad

        _ctx.append(ContextGraph(t2, grad_fn2))

    return o, requires_grad, _ctx


def neg_(t: "Tensor") -> Tuple[np.ndarray, bool, List[ContextGraph]]:

    o = -t.data
    requires_grad = t.requires_grad
    if requires_grad:
        _ctx = [ContextGraph(t, lambda x: -x)]
    else:
        _ctx = []

    return o, requires_grad, _ctx


def matmul_(t1: "Tensor", t2: "Tensor") -> Tuple[np.ndarray, bool, List[ContextGraph]]:

    o = t1.data @ t2.data

    requires_grad = t1.requires_grad or t2.requires_grad
    _ctx: List[ContextGraph] = []

    if t1.requires_grad:

        def grad_fn1(grad: np.ndarray) -> np.ndarray:
            return grad @ t2.data.T

        _ctx.append(ContextGraph(t1, grad_fn1))

    if t2.requires_grad:

        def grad_fn2(grad: np.ndarray) -> np.ndarray:
            return t1.data.T @ grad

        _ctx.append(ContextGraph(t2, grad_fn2))

    return o, requires_grad, _ctx


def slice_(t: "Tensor", idx) -> Tuple[np.ndarray, bool, List[ContextGraph]]:

    o = t.data[idx]
    requires_grad = t.requires_grad

    if requires_grad:

        def grad_fn(grad: np.ndarray) -> np.ndarray:
            partial_grad = np.zeros_like(o)
            partial_grad[idx] = grad
            return partial_grad

        _ctx.append(ContextGraph(t, grad_fn))

    else:
        _ctx = []

    return o, requires_grad, _ctx
