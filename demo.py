import torch
from torch.utils import data
import numpy as np
import os
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

torch.manual_seed(10)


class resize:

    def __init__(self, reshape_size=224):
        self.reshape_size = reshape_size
        pass

    def __call__(self, img):
        w, h = img.size
        long = max(w, h)
        # 按比例缩放成512
        w, h = int(w / long * self.reshape_size), int(h / long * self.reshape_size)
        # 压缩并插值
        img = img.resize((w, h))
        # 然后是给短边扩充，使用ImageOps.expand
        delta_w, delta_h = self.reshape_size - w, self.reshape_size - h
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        img = ImageOps.expand(img, padding)
        return img


def showImg(img, binary=True, fName=''):
    """
  show image from given numpy image
  """
    img = img[0, :, :] * 255

    # if binary:
    #   img = img > 0.5

    img = Image.fromarray(np.uint8(img), mode='L')
    img.show()


class UnetDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.root_dir = root_dir  # 文件目录
        self.transform = transform  # 变换
        self.images = os.listdir(self.root_dir)  # 目录里的所有文件
        self.list = []
        for i in range(5):
            print(i)
            self.list.append(self.read_a_pic(i))
        self.imglist = torch.stack(tuple(self.list))

    def __len__(self):
        return len(self.images)

    def read_a_pic(self, index):
        image_imdex = self.images[index]
        img_path = os.path.join(self.root_dir, image_imdex)
        img = Image.open(img_path)
        return torch.repeat_interleave(self.transform(img), 1, dim=0)
        # return torch.repeat_interleave(self.transform(img), 3, dim=0)

    def __getitem__(self, index):
        return self.imglist[index]


def show(images, y, savename):
    def show_image(image, title=''):
        plt.imshow(image, cmap='Greys_r')
        plt.title(title, fontsize=16)
        plt.axis('off')
        return

    # plt.rcParams['figure.figsize'] = [24, 24]
    plt.figure(dpi=450)
    plt.subplot(2, 5, 1)
    show_image(images[0], "origin")
    plt.subplot(2, 5, 2)
    show_image(images[1], " ")
    plt.subplot(2, 5, 3)
    show_image(images[2], " ")
    plt.subplot(2, 5, 4)
    show_image(images[3], " ")
    plt.subplot(2, 5, 5)
    show_image(images[4], " ")

    plt.subplot(2, 5, 6)
    show_image(y[0], "output")
    plt.subplot(2, 5, 7)
    show_image(y[1], " ")
    plt.subplot(2, 5, 8)
    show_image(y[2], " ")
    plt.subplot(2, 5, 9)
    show_image(y[3], " ")
    plt.subplot(2, 5, 10)
    show_image(y[4], " ")

    plt.savefig(savename)

    # plt.show()



def main(args):
    model = torch.load(args.chkpt, map_location='cpu')
    model.eval()
    print(f"model :{model}")

    trans = transforms.Compose([
        resize(args.input_size),
        # normal(),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    dataset = UnetDataset(args.data_path, transform=trans)

    input = dataset.__getitem__([0, 1, 2, 3, 4])

    y_hat = model(input)
    y_hat = y_hat.detach()
    loss_fn = nn.MSELoss()
    print(f"model loss is: {loss_fn(input, y_hat)}")


    y_hat = y_hat.permute(0, 2, 3, 1)
    images = input.permute(0, 2, 3, 1)

    save_dirs = os.path.dirname(args.save_path)
    if not os.path.exists(save_dirs):
        print(f"create dirs {save_dirs}")
        os.makedirs(save_dirs)

    show(images, y_hat, args.save_path)

from setting import  get_demo_args

args = get_demo_args()
main(args)
