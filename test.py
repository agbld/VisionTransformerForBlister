#%%
# Explaination of TripletLoss_fix

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
from utils.dataset import TripletBlister_Dataset, Prototype_Dataset
import torchvision
import PIL
import matplotlib.pyplot as plt
import numpy as np
import torch

TRAIN_PATH    = './data/train_50'
IMG_SIZE = 416

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop(512),
    torchvision.transforms.Resize([IMG_SIZE, IMG_SIZE]),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(10, resample=PIL.Image.BILINEAR),
    torchvision.transforms.RandomAffine(8, translate=(.15, .15)),
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                 std=[0.229, 0.224, 0.225]), 
    # torchvision.transforms.Normalize(mean=[0.4590, 0.4521, 0.4120], # training set mean (r, g, b)
    #                                 std=[0.2468, 0.2397, 0.2307]),  # training set std (r, g, b)
    ])
ps1_ps2_train_dataset = torchvision.datasets.ImageFolder(
    root=TRAIN_PATH, transform=data_transform)
triplet_train_dataset = TripletBlister_Dataset(
    ps1_ps2_train_dataset)  # Returns triplets of images
prototype_train_dataset = Prototype_Dataset(ps1_ps2_train_dataset, len(triplet_train_dataset.labels_set), list(range(48)))

mean_list = []
std_list = []
for cls_idx in range(1):
    imgs = torch.stack(prototype_train_dataset[cls_idx])
    # imgs = imgs.permute(1, 0, 2, 3)
    # mean_list.append(imgs.mean(dim=(1, 2, 3)))
    # std_list.append(imgs.std(dim=(1, 2, 3)))
    first_img = imgs[0]
    first_img: torch.Tensor
    # first_img[0] = first_img[0] + 0.4590
    # first_img[1] = first_img[1] + 0.4521
    # first_img[2] = first_img[2] + 0.4120
    print(cls_idx, first_img.shape)
    # print(first_img)
    plt.imshow((first_img.permute(1, 2, 0)))

# %%
dest_folder = './data/full_50'
src_folder = './data/train_50'
import os
import shutil
for cls_idx in range(51):
    src_path = os.path.join(src_folder, str(cls_idx))
    dest_path = os.path.join(dest_folder, str(cls_idx))
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    for img_name in os.listdir(src_path):
        src_img_path = os.path.join(src_path, img_name)
        dest_img_path = os.path.join(dest_path, img_name)
        shutil.copy(src_img_path, dest_img_path)
# %%
tmp_dict = {'a': 1, 'b': 2, 'c': 3}
print(tmp_dict)
tmp_dict2 = {i: v for i, v in enumerate(list(tmp_dict.keys()))}
print(tmp_dict2)
# %%
