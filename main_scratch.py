import torch



t = torch.randn(*(2,3,3,2))
t = t.unsqueeze(0)
print(t.shape)
print(t.index_select(dim=2,index=torch.tensor([0])))
print(t.index_select(dim=2,index=torch.tensor([0])).shape)

t = t.permute([2,1,0,3,4])
# t = t.split(3,dim=3)
print(t.shape)
print(t.index_select(dim=0,index=torch.tensor([0])))
print(t.index_select(dim=0,index=torch.tensor([0])).shape)

