import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler


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


class Prototype_Dataset(Dataset):

    def __init__(self, dataset, class_num, prototype_index):
        
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
        
        for each_index in image_index[self.prototype_index]:
            all_prototype_img.append(self.transform(Image.open(self.train_paths[each_index])))
        
        return all_prototype_img

    def __len__(self):
        return self.class_num

        