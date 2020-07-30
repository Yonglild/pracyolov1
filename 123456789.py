import torch

a = torch.tensor([[1, 2, 3], [4, 5, 6]])
b = torch.tensor([[11, 22, 33], [44, 55, 66]])

c = torch.stack([a, b], dim=0)      # c = [a, b], size([2, 2, 3])
print(a.size())
print(c)
print(c.size())


# 对于torch.stack来说，会先将原始数据维度扩展一维，然后再按照维度进行拼接;
# d = torch.stack([a, b], dim=2)
# print(d)
# print(d.size())

# 对于torch.cat来说， 直接在维度上叠加
d = torch.cat([a, b], dim=0)
print(d)
print(d.size())