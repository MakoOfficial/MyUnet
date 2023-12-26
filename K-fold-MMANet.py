import csv

import torch
import cv2
import os
from torch.utils import data
from torch.optim.lr_scheduler import StepLR
import pandas as pd
from torchvision import transforms
from torch import nn
from torch.optim import Adam
from utils import datasets, MMANet
from utils.setting import get_class_args
from utils.func import print, eval_func, normalize_age, L1_regular, eval_func_MMANet
import numpy as np
import random
from sklearn.model_selection import KFold
import time

from albumentations.augmentations.transforms import Lambda, Normalize, RandomBrightnessContrast
from albumentations.augmentations.geometric.transforms import ShiftScaleRotate, HorizontalFlip
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.augmentations.crops.transforms import RandomResizedCrop
from albumentations import Compose, OneOrOther

import torch.nn.functional as F

norm_mean = [0.143]  # 0.458971
norm_std = [0.144]  # 0.225609

#   数据增强用的（本质还是正则化，增强鲁棒性），随机删除一个图片上的像素，p为执行概率，scale擦除部分占据图片比例的范围，ratio擦除部分的长宽比范围
RandomErasing = transforms.RandomErasing(scale=(0.02, 0.08), ratio=(0.5, 2), p=0.8)


def randomErase(image, **kwargs):
    return RandomErasing(image)


def sample_normalize(image, **kwargs):
    image = image / 255
    channel = image.shape[2]
    #   平均值和标准差的计算，先将通道提出来，分别计算三个通道的平均值和方差
    mean, std = image.reshape((-1, channel)).mean(axis=0), image.reshape((-1, channel)).std(axis=0)
    return (image - mean) / (std + 1e-3)  # 1e-3:0.001,1乘以10的-3次方


#   利用compose将多个步骤合到一起
transform_train = Compose([
    # RandomBrightnessContrast(p = 0.8),
    RandomResizedCrop(512, 512, (0.5, 1.0), p=0.5),  # 512为调整后的图片大小，（0.5,1.0）为scale剪切的占比范围，概率p为0.5
    # ShiftScaleRotate操作：仿射变换，shift为平移，scale为缩放比率，rotate为旋转角度范围，border_mode用于外推法的标记，value即为padding_value，前者用到的，p为概率
    ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, border_mode=cv2.BORDER_CONSTANT, value=0.0,
                     p=0.8),
    # HorizontalFlip(p = 0.5),

    # ShiftScaleRotate(shift_limit = 0.2, scale_limit = 0.2, rotate_limit=20, p = 0.8),
    HorizontalFlip(p=0.5),  # 水平翻转
    RandomBrightnessContrast(p=0.8, contrast_limit=(-0.3, 0.2)),  # 概率调整图片的对比度
    Lambda(image=sample_normalize),  # 标准化
    ToTensorV2(),  # 将图片转化为tensor类型
    Lambda(image=randomErase)  # 做随机擦除

])

#   对验证集和测试集只进行了简单的normalize，然后转化为tensor
transform_val = Compose([
    Lambda(image=sample_normalize),
    ToTensorV2(),
])


def L1_penalty(net, alpha):
    l1_penalty = torch.nn.L1Loss(size_average=False)
    loss = 0
    for param in net.module.fc.parameters():
        loss += torch.sum(torch.abs(param))

    return alpha * loss


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_fold(args, train_set, val_set, k):
    gpus = [0, 1]
    torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    classifer = MMANet.BAA_New(32, *MMANet.get_My_resnet50())
    classifer = nn.DataParallel(classifer.cuda(), device_ids=gpus, output_device=gpus[0])
    print(f'number of training params: {sum(p.numel() for p in classifer.parameters() if p.requires_grad) / 1e6} M')

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    loss_func = nn.L1Loss(reduction="sum")

    epochs = args.epochs
    optimizer = Adam(classifer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(epochs):
        total_loss = 0.
        train_length = 0.
        classifer.train()
        start_time = time.time()
        for idx, data in enumerate(train_loader):
            image, gender = data[0]
            image, gender = image.cuda(), gender.cuda()

            batch_size = len(data[1])
            label = data[1].cuda()
            optimizer.zero_grad()

            _, _, _, y_pred = classifer(image, gender)

            y_pred = torch.squeeze(y_pred)
            label = torch.squeeze(label)

            assert label.shape == y_pred.shape, "pred and output isn't the same shape"

            loss = loss_func(y_pred, label)
            capital_loss = loss + L1_penalty(classifer, 1e-5)
            capital_loss.backward()
            optimizer.step()
            train_length += batch_size
            total_loss += loss.item()
        end_time = time.time()
        print(f'epoch {epoch + 1}: training loss: {round(total_loss / train_length, 3)}, '
              f'valid loss: {round(eval_func_MMANet(classifer, val_loader, boneage_mean, boneage_div), 3)}, '
              f'lr:{optimizer.param_groups[0]["lr"]}'
              f'cost time is {end_time - start_time}')
        scheduler.step()
    save_path = os.path.join(args.save_path, f"MMANet_{k}Fold.pth")
    torch.save(classifer, save_path)

    with torch.no_grad():
        train_record = [['label', 'pred']]
        train_record_path = os.path.join(args.save_path, f"train{k}.csv")
        train_length = 0.
        total_loss = 0.
        classifer.eval()
        for idx, data in enumerate(train_loader):
            image, gender = data[0]
            image, gender = image.cuda(), gender.cuda()

            batch_size = len(data[1])
            label = data[1].cuda()

            _, _, _, y_pred = classifer(image, gender)

            output = (y_pred.cpu() * boneage_div) + boneage_mean
            label = (label.cpu() * boneage_div) + boneage_mean

            output = torch.squeeze(output)
            label = torch.squeeze(label)
            for i in range(output.shape[0]):
                train_record.append([label[i].item(), round(output[i].item(), 2)])
            assert output.shape == label.shape, "pred and output isn't the same shape"

            total_loss += F.l1_loss(output, label, reduction='sum').item()
            train_length += batch_size
        print(f"length :{train_length}")
        print(f'{k} fold final training loss: {round(total_loss / train_length, 3)}')
        with open(train_record_path, 'w', newline='') as csvfile:
            writer_train = csv.writer(csvfile)
            for row in train_record:
                writer_train.writerow(row)

    with torch.no_grad():
        val_record = [['label', 'pred']]
        val_record_path = os.path.join(args.save_path, f"val{k}.csv")
        val_length = 0.
        val_loss = 0.
        classifer.eval()
        for idx, data in enumerate(val_loader):
            image, gender = data[0]
            image, gender = image.cuda(), gender.cuda()

            batch_size = len(data[1])
            label = data[1].cuda()

            _, _, _, y_pred = classifer(image, gender)

            output = (y_pred.cpu() * boneage_div) + boneage_mean
            label = (label.cpu() * boneage_div) + boneage_mean

            output = torch.squeeze(output)
            label = torch.squeeze(label)
            for i in range(output.shape[0]):
                val_record.append([label[i].item(), round(output[i].item(), 2)])
            assert output.shape == label.shape, "pred and output isn't the same shape"

            val_loss += F.l1_loss(output, label, reduction='sum').item()
            val_length += batch_size
        print(f"length :{val_length}")
        print(f'{k} fold final val loss: {round(val_loss / val_length, 3)}')
        with open(val_record_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in val_record:
                writer.writerow(row)


if __name__ == "__main__":
    args = get_class_args()
    print(args)
    seed_everything(args.seed)
    print(f'Set manual random seed: {args.seed}')

    df = pd.read_csv(args.csv_path)
    df, boneage_mean, boneage_div = normalize_age(df)
    train_ori_dir = args.ori_train_path

    train_dataset = datasets.MMANetDataset(df=df, data_dir=train_ori_dir)
    print(f'Training dataset info:\n{train_dataset}')
    data_len = train_dataset.__len__()
    X = torch.randn(data_len, 1)
    kf = KFold(n_splits=5, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=X)):
        print(f"Fold {fold + 1}/5")
        ids, zscore, male = train_dataset[train_idx]
        train_set = datasets.Kfold_MMANet_Dataset(ids, zscore, male, train_ori_dir, transforms=transform_train)
        ids1, zscore1, male1 = train_dataset[val_idx]
        val_set = datasets.Kfold_MMANet_Dataset(ids1, zscore1, male1, train_ori_dir, transforms=transform_val)
        torch.set_default_tensor_type('torch.FloatTensor')

        run_fold(args, train_set, val_set, fold + 1)


