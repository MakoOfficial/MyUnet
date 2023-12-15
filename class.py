import torch
import model
import os
from torch.utils import data
from PIL import Image

checkpoint_ori = torch.load("./checkpoint/server/checkpoint_ori_200.pth")
checkpoint_canny = torch.load("./checkpoint/server/checkpoint_canny_200.pth")

classifer = model.classifer(checkpoint_ori, checkpoint_canny)

# print(classifer)
for name,parameters in classifer.named_parameters():
        print(name,':',parameters.size())
        print('-->grad_requirs:', parameters.requires_grad)

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


def main():
    return