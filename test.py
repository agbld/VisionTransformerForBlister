#%%
import torch
import torch.nn as nn
from utils.losses import TripletLoss_fix

#%%
t1 = (torch.randn(3, 1) * 10).to(dtype=torch.float16).cuda()
print(t1)
t2 = (torch.randn(3, 1) * 10).to(dtype=torch.float16).cuda()
print(t2)
t3 = (torch.randn(3, 1) * 10).to(dtype=torch.float16).cuda()

#%%
dist = torch.cdist(t1.unsqueeze(1), t2.unsqueeze(1), p=2)
print(dist)

#%%
loss_fn = TripletLoss_fix().to(dtype=torch.float16)

loss = loss_fn(t1, t1, t3)
print(loss)

loss = loss_fn(t1, t2, t3)
print(loss)

loss = loss_fn(t1, t2, t1)
print(loss)

# %%
