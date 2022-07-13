#%%
import torch
import torch.nn as nn
from utils.losses import TripletLoss, TripletLoss_fix

# initialize t1, t2, t3
batch_size = 128
dim = 8
t1 = (torch.randn(batch_size, dim) * 10)
t2 = (torch.randn(batch_size, dim) * 10)
t3 = (torch.randn(batch_size, dim) * 10)
print('t1.shape: ', t1.shape)
print('t2.shape: ', t2.shape)
print('t3.shape: ', t3.shape)

# Problem demonstration

# original triplet loss function
print('\noriginal loss function:')
loss_fn = TripletLoss()
loss = loss_fn(t1, t1, t3)
print('loss_fn(t1, t1, t3) = ', loss) # must be lowest (but doesn't!!)
loss = loss_fn(t1, t2, t3)  
print('loss_fn(t1, t2, t3) = ', loss)
loss = loss_fn(t1, t2, t1)
print('loss_fn(t1, t2, t1) = ', loss) # must be highest (but doesn't!!)

# fixed triplet loss function
print('\nfixed loss function:')
loss_fn = TripletLoss_fix()
loss = loss_fn(t1, t1, t3)
print('loss_fn(t1, t1, t3) = ', loss)  # must be lowest
loss = loss_fn(t1, t2, t3)  
print('loss_fn(t1, t2, t3) = ', loss)
loss = loss_fn(t1, t2, t1)
print('loss_fn(t1, t2, t1) = ', loss)  # must be highest

# Reason
print('\nReason:')
# we expect a (batch_size, 1) tensor of cdist output, representing the distance between each pair of samples
# however, the original cdist arguments unsqueeze in wrong dimension, resulting a (1, batch_size, batch_size) tensor

# cdist arguments used in original triplet loss function
dist_original = torch.cdist(t1.unsqueeze(0), t2.unsqueeze(0))   # this output a matrix of interlaced distances (element-wise) between t1 and t2
print('dist_original.shape: ', dist_original.shape)

# cdist arguments used in fixed triplet loss function
dist_fixed = torch.cdist(t1.unsqueeze(1), t2.unsqueeze(1))   # this output a matrix of pair-wise distances between t1 and t2, which is more reasonable for training.
print('dist_fixed.shape: ', dist_fixed.shape)

# %%
from utils.dataset import TripletBlister_Dataset
