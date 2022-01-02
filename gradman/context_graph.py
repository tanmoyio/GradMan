from typing import Callable, NamedTuple

import numpy as np


class ContextGraph(NamedTuple):
    tensor: "Tensor"  # noqa: F821
    grad_fn: Callable[[np.ndarray], np.ndarray]
