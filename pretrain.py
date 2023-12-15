import torch.utils.data as data
import os
from model import UNet
import torch.optim as optim
from torchvision import transforms
import time
import torch
import setting

from PIL import Image, ImageOps
from torch.optim.lr_scheduler import StepLR

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


def train(args):
    net = UNet()
    net.cuda()
    net.train()
    # 在多个GPU上设置模型

    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        resize(args.input_size),
        # normal(),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    dataset = UnetDataset(args.data_path, transform=trans)
    print(f"dataset is {dataset}")
    sampler = torch.utils.data.RandomSampler(data_source=dataset)
    print(f"sampler is {sampler}")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True
    )

    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    for epo in range(args.epochs):
        batch_idx = 0
        total_loss = 0
        start_time = time.time()
        for idx, data in enumerate(dataloader):
            batch_idx += 1
            optimizer.zero_grad()
            data = data.float().cuda()
            # print(data)
            out = net(data)
            loss = criterion(out, data)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        end_time = time.time()
        print('This epoch {epo}, the loss is {loss}, cost time is {time}, lr={lr}'.format(epo=epo,
                                                                                          loss=total_loss / batch_idx,
                                                                                          time=end_time - start_time,
                                                                                          lr=optimizer.param_groups[0][
                                                                                              "lr"]))
        scheduler.step()
    torch.save(net, args.save_path)

#=====================================================================
args = setting.get_args()
train(args)