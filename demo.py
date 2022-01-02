from gradman import Tensor

a = Tensor([1.0, 0.5, 0.8], requires_grad=True)
b = a.sum()
b.backward()
print(a.grad)
