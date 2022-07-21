import numpy as np
from PIL import Image
import random
from torch import negative
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder

# Triplet dataset class for torchvision.datasets.ImageFolder, the latest version
class TripletDataset(Dataset):
    def __init__(self, img_folder_dataset: ImageFolder, num_pos: int=10, neg_ratio: int=1):
        self.img_folder_dataset = img_folder_dataset # __getitem__() -> (img, label)
        self.transform = self.img_folder_dataset.transform
        
        self.imgs_path = [item[0] for item in self.img_folder_dataset.imgs]
        self.img_idxs = list(range(len(self.imgs_path)))
        self.labels = [item[1] for item in self.img_folder_dataset.imgs]
        self.labels_set = set(self.labels)
        
        self.anchor_idxs = []
        self.positive_idxs = []
        self.negatives_idx = []
        for anchor_idx in self.img_idxs:
            # sample positive index with respect to the anchor
            positive_label = self.labels[anchor_idx]
            all_positive_idxs = [idx for idx in self.img_idxs if self.labels[idx] == positive_label]
            cropped_num_pos = min(num_pos, len(all_positive_idxs))
            positive_idxs = list(np.random.choice(all_positive_idxs, size=cropped_num_pos, replace=False))
            for i in range(neg_ratio - 1):
                positive_idxs += positive_idxs
            
            # sample negative index with respect to the anchor
            negative_labels = self.labels_set - {positive_label}
            all_negative_idxs = list(set(self.img_idxs) - set(all_positive_idxs))
            cropped_num_neg = min(len(all_negative_idxs), cropped_num_pos * neg_ratio)
            negative_idxs = list(np.random.choice(all_negative_idxs, size=cropped_num_neg, replace=False))
            
            # align anchor index with positive and negative index
            anchor_idxs = list(np.repeat(anchor_idx, len(negative_idxs)))
            
            self.anchor_idxs += anchor_idxs
            self.positive_idxs += positive_idxs
            self.negatives_idx += negative_idxs
        
    def __getitem__(self, index):
        anchor_img_path = self.imgs_path[self.anchor_idxs[index]]
        positive_img_path = self.imgs_path[self.positive_idxs[index]]
        negative_img_path = self.imgs_path[self.negatives_idx[index]]
        anchor_img = Image.open(anchor_img_path)
        positive_img = Image.open(positive_img_path)
        negative_img = Image.open(negative_img_path)
        
        if self.transform is not None:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
        return (anchor_img, positive_img, negative_img)
    
    def __len__(self):
        return len(self.anchor_idxs)

class TripletBlister_Dataset(Dataset):
    def __init__(self, blister_dataset):
        self.blister_dataset = blister_dataset
        self.transform = self.blister_dataset.transform

        self.train_paths = [item[0] for item in self.blister_dataset.imgs]
        self.train_labels = np.array([item[1] for item in self.blister_dataset.imgs])
        self.labels_set = set(self.train_labels)
        self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                    for label in self.labels_set}

    def __getitem__(self, index):
        # 取 anchor
        label1 = self.train_labels[index].item()
        img1 = Image.open(self.train_paths[index])
        # 取 positive (確保不會取到 anchor)
        positive_index = index
        while positive_index == index:
            positive_index = np.random.choice(self.label_to_indices[label1])
        img2 = Image.open(self.train_paths[positive_index])
        # 取 negative
        negative_label = np.random.choice(list(self.labels_set - set([label1]))) # 隨機取得負樣本標記
        negative_index = np.random.choice(self.label_to_indices[negative_label]) # 利用標記取到影像
        img3 = Image.open(self.train_paths[negative_index])
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)
        return (img1, img2, img3), []
    def __len__(self):
        return len(self.blister_dataset)

class TripletBlister_Dataset_mod(TripletBlister_Dataset):
    def __init__(self, blister_dataset, neg_ratio:int=1):
        super().__init__(blister_dataset)
        self.neg_ratio = int(neg_ratio)

    def __getitem__(self, index):
        actual_index = index % len(self.blister_dataset)
        return super().__getitem__(actual_index)

    def __len__(self):
        return super().__len__() * self.neg_ratio

class Prototype_Dataset(Dataset):

    def __init__(self, dataset, class_num, prototype_index = None):
        
        self.transform = dataset.transform
        self.train_paths = [item[0] for item in dataset.imgs]
        self.train_labels = np.array([item[1] for item in dataset.imgs])
        self.labels_set = set(self.train_labels)
        self.label_to_indices = {label: np.where(self.train_labels == label)[0]
                                for label in self.labels_set}
        self.prototype_index = prototype_index
        self.class_num = class_num

    def __getitem__(self, index):

        image_index = self.label_to_indices[index]
        all_prototype_img = []
        
        if self.prototype_index != None:
            image_index = image_index[self.prototype_index]
        for each_index in image_index:
            all_prototype_img.append(self.transform(Image.open(self.train_paths[each_index])))
        
        return all_prototype_img

    def __len__(self):
        return self.class_num

        