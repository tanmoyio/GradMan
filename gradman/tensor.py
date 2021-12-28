import numpy as np

class Tensor:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.float32) if not isinstance(data, np.ndarray) else data

    def __repr__(self):
        return f"<Tensor {self.data!r}>"

    @property
    def shape(self): return self.data.shape

    @classmethod
    def eye(cls, dim, **kwargs):
        return cls(np.eye(dim).astype(np.float32), **kwargs)

    def matmul(self, w):
        return Tensor(np.matmul(self.data,w.data))
    dot = matmul
