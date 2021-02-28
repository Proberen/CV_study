# -*- coding: utf-8 -*-

import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import cv2



# random.seed(1)
#
#
# rmb_label = {"1": 0, "100": 1}
#
# train_transform = transforms.Compose([
#     transforms.Resize((32, 32)),
#     # transforms.RandomCrop(32, padding=4),
#     transforms.ToTensor(),
#     # transforms.Normalize(norm_mean, norm_std),
# ])

class RMBDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.label_name = {"1": 0, "100": 1}
        self.data_info = self.get_img_info(data_dir)  # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.transform = transform

    def __getitem__(self, index):
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255

        if self.transform is not None:
            img = self.transform(img)   # 在这里做transform，转为tensor等等
        return img, label

    def __len__(self):
        return len(self.data_info)

    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        rmb_label = {"1": 0, "100": 1}
        for root, dirs, _ in os.walk(data_dir): #
            # 遍历类别
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))

                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = rmb_label[sub_dir]
                    data_info.append((path_img, int(label)))

        return data_info


# train_dir='./RMB_data/rmb_split/train/'
#
# train_data = RMBDataset(data_dir=train_dir, transform=train_transform)
# # 构建DataLoder
# train_loader = DataLoader(dataset=train_data, batch_size=16)
#
#
# for epoch in range(10):
#     for i, data in enumerate(train_loader):
#         pass




# list_train = [[1,2,3,4],[5,6,7,8],[9,1,2,3]]
# for i, data in enumerate(list_train):
#     pass


# dir = "./data/"  #100张图片
#
# def 挤牙膏操作(dir,batch_size):
#     读取文件夹(dir)
#     image ,label=按照batch_size进行数据的拆分(batch_size)
#     image=数据预处理(image)
#     return image ,label
#
# for i in range(5): #5个epoch
#     for j in range(10):
#         image, label = 挤牙膏操作(dir, batch_size=10)


#         pre_label = 神经网络(image)
#         损失函数(pre_label,label)
#




#
# class MYDataset(Dataset):
#     def __init__(self, data_dir, transform=None):
#        ..
#
#     def __getitem__(self, index):
#         ..
#         return  ..,..
#
#     def __len__(self):
#         return len(..)
#
#
# train_dir='./'
#
# train_data = MYDataset(data_dir=train_dir)
# # 构建DataLoder
# train_loader = DataLoader(dataset=train_data, batch_size=16)
#
#
# for epoch in range(10):
#     for i, data in enumerate(train_loader):
#         pass























