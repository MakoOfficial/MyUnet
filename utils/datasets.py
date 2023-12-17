import os
import torch.utils.data as data
from PIL import Image, ImageOps
import numpy as np
import torch


class resize:
    """resize the pic, and remain the ratio,use 0 padding """

    def __init__(self, reshape_size=224):
        self.reshape_size = reshape_size
        pass

    def __call__(self, img):
        w, h = img.size
        long = max(w, h)
        w, h = int(w / long * self.reshape_size), int(h / long * self.reshape_size)
        img = img.resize((w, h))
        delta_w, delta_h = self.reshape_size - w, self.reshape_size - h
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img = ImageOps.expand(img, padding)
        return img


class UnetDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.images = os.listdir(self.root_dir)  # 目录里的所有文件
        self.list = []
        for i in range(len(self.images)):
            print(i)
            self.list.append(self.read_a_pic(i))
        # tuple_1= tuple(self.list)
        # self.imglist = torch.stack(tuple_1)

    def __len__(self):
        return len(self.images)

    def read_a_pic(self, index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        # label_path = os.path.join(self.root_dir, "label", image_index)

        img = Image.open(img_path)
        # label = Image.open(label_path)
        # return (self.transform(img), self.transform(label))
        return self.transform(img)

    def __getitem__(self, index):
        return self.list[index]


class ClassDataset(data.Dataset):
    def __init__(self, df, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.images = os.listdir(self.root_dir)  # 目录里的所有文件
        self.df = df    # load the dataframe from cvd file
        self.image = []
        self.age = []
        self.male = []
        print(type(self.df['id'][0]))
        for i in range(len(self.images)):
            print(i)
            self.image.append(self.read_a_pic(i))
            age, male = self.get_label(i)
            self.age.append(age)
            self.male.append(male)
        # tuple_1= tuple(self.list)
        # self.imglist = torch.stack(tuple_1)

    def __len__(self):
        return len(self.images)

    def read_a_pic(self, index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        # label_path = os.path.join(self.root_dir, "label", image_index)

        img = Image.open(img_path)
        # label = Image.open(label_path)
        # return (self.transform(img), self.transform(label))
        return self.transform(img)

    def get_label(self, index):
        image_index = self.images[index]
        image_id = image_index.split('.')[0]
        row = self.df[self.df['id'] == int(image_id)]
        boneage = np.array(row['boneage'])
        male = np.array(row['male'].astype('float32'))
        return torch.Tensor(boneage), torch.Tensor(male)


    def __getitem__(self, index):
        return self.image[index], self.age[index], self.male[index]