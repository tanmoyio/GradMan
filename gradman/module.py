from gradman import Tensor

class Module:
    
    def _optimize(self,i,optimizer):
        for att in i.__dict__:
            if att in i.params: 
                n = getattr(i, att)
                v = Tensor(optimizer(n.data, n.grad))
                setattr(i, att, v)

    def optimize(self, optimizer):
        objs = [getattr(self, i) for i in self.__dict__]
        for i in objs:
            if isinstance(i, Module): self._optimize(i, optimizer) 

    def _forward(self,i):
        return self.forward(i)
    __call__ = _forward
