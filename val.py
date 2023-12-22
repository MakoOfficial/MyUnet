import torch
import model
import os
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from torchvision import transforms
from torch import nn
from torch.optim import Adam
from utils import datasets
from utils.setting import get_class_args
from utils.func import print, eval_func, normalize_age


import numpy as np
import random

def setup_seed(seed=3407):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True # 选择确定性算法
    torch.backends.cudnn.benchmark = False # if benchmark=True, deterministic will be False
    torch.backends.cudnn.enabled = False

# setup_seed(seed=3047)
setup_seed(seed=0)
net = torch.load('../../autodl-tmp/classifer_200.pth')
# net = torch.load('CHECKPOINT_ori.pth')

df = pd.read_csv('../archive/boneage-training-dataset.csv')
ori_dir = '../masked_1K_train/ori'
canny_dir = '../masked_1K_train/canny'
val_trans = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Grayscale(),
    transforms.ToTensor(),
])
dataset = datasets.ClassDataset(df=df, ori_dir=ori_dir, canny_dir=canny_dir, transform=val_trans)
sampler = torch.utils.data.RandomSampler(data_source=dataset)
# loader = data.dataloader.DataLoader(
#     dataset=dataset,
#     batch_size=80,
#     shuffle=False,
#     drop_last=True
# )
loader = data.dataloader.DataLoader(
        dataset=dataset,
        batch_size=80,
        # sampler=sampler,
        shuffle=False,
        drop_last=True
    )
net = net.cuda()
net.eval()
loss_func = nn.L1Loss(reduction='sum')
val_length = 0.
total_loss = 0.
with torch.no_grad():
    for idx, batch in enumerate(loader):
        val_length += batch[0].shape[0]
        images = batch[0].cuda()
        cannys = batch[1].cuda()
        boneage = batch[2].cuda()
        male = batch[3].cuda()
        output = net(images, cannys, male)
        # output = net(images, male)
        output = torch.squeeze(output)
        boneage = torch.squeeze(boneage)
        print(f"pred: {output}, \nlabel: {boneage}")
        loss = loss_func(output, boneage)
        total_loss += loss.item()

print(f'length is {val_length}')    
print(f'loss is {round(total_loss/val_length, 3)}')

