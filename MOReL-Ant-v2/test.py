import torch

a = torch.tensor([1, 2])
b = torch.tensor([3, 4])
c = torch.cat((a, b), -1)
print(c)