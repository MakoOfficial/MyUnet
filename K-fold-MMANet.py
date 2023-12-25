import csv

import torch
import model
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

def setup_seed(seed=3407):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED']  = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True  # 选择确定性算法
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.enabled = False


def run_fold(args, train_set, val_set, k):
    classifer = MMANet.BAA_New(32, *MMANet.get_My_resnet50())
    print(f'number of training params: {sum(p.numel() for p in classifer.parameters() if p.requires_grad) / 1e6} M')

    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False
    )

    val_loader = data.DataLoader(
        dataset=val_set,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    loss_func = nn.L1Loss(reduction="sum")

    epochs = args.epochs
    optimizer = Adam(classifer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Use step level LR & WD scheduler!")
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print(f'Scheduler:\n{scheduler}')
    classifer.cuda()

    for epoch in range(epochs):
        classifer.train()
        total_loss = 0.
        train_length = 0.
        classifer.train()
        start_time = time.time()
        for idx, batch in enumerate(train_loader):
            images = batch[0].cuda()
            boneage = batch[2].cuda()
            male = batch[3].cuda()
            optimizer.zero_grad()

            output = classifer(images, male)

            output = torch.squeeze(output)
            boneage = torch.squeeze(boneage)

            assert output.shape == boneage.shape, "pred and output isn't the same shape"

            loss = loss_func(output, boneage) + L1_regular(classifer, 1e-4)
            loss.backward()
            optimizer.step()
            train_length += batch[0].shape[0]
            total_loss += loss.item()
        end_time = time.time()
        print(f'epoch {epoch + 1}: training loss: {round(total_loss / train_length, 3)}, '
              f'valid loss: {round(eval_func_MMANet(classifer, val_loader), 3)}, '
              f'lr:{optimizer.param_groups[0]["lr"]}'
              f'cost time is {end_time - start_time}')
        scheduler.step()

    with torch.no_grad():
        train_record = [['label', 'pred']]
        train_record_path = os.path.join(args.save_path, f"train{k}.csv")
        train_length = 0.
        total_loss = 0.
        classifer.eval()
        for idx, patch in enumerate(train_loader):
            train_length += patch[0].shape[0]
            images = patch[0].cuda()
            boneage = patch[2].cuda()
            male = patch[3].cuda()

            output = classifer(images, male)

            output = torch.squeeze(output)
            boneage = torch.squeeze(boneage)
            for i in range(output.shape[0]):
                train_record.append([boneage[i].item(), round(output[i].item(), 2)])
            assert output.shape == boneage.shape, "pred and output isn't the same shape"

            loss = loss_func(output, boneage)
            total_loss += loss.item()
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
        for idx, patch in enumerate(val_loader):
            val_length += patch[0].shape[0]
            images = patch[0].cuda()
            boneage = patch[2].cuda()
            male = patch[3].cuda()

            output = classifer(images, male)

            output = torch.squeeze(output)
            boneage = torch.squeeze(boneage)
            for i in range(output.shape[0]):
                val_record.append([boneage[i].item(), round(output[i].item(), 2)])
            assert output.shape == boneage.shape, "pred and output isn't the same shape"

            loss = loss_func(output, boneage)
            val_loss += loss.item()
        print(f"length :{val_length}")
        print(f'{k} fold final val loss: {round(val_loss / val_length, 3)}')
        with open(val_record_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in val_record:
                writer.writerow(row)
    save_path = os.path.join(args.save_path, f"MMANet_{k}Fold.pth")
    torch.save(classifer, save_path)


def main(args):
    print(args)
    setup_seed(args.seed)
    print(f'Set manual random seed: {args.seed}')

    df = pd.read_csv(args.csv_path)
    df, boneage_mean, boneage_div = normalize_age(df)
    train_ori_dir = args.ori_train_path
    train_canny_dir = args.canny_train_path
    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    train_dataset = datasets.ClassDataset(df=df, ori_dir=train_ori_dir, canny_dir=train_canny_dir,
                                          transform=train_trans)
    print(f'Training dataset info:\n{train_dataset}')
    data_len = train_dataset.__len__()
    X = torch.randn(data_len, 2)
    kf = KFold(n_splits=5, shuffle=True)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X=X)):
        print(f"Fold {fold + 1}/5")
        ori, canny, age, male = train_dataset[train_idx]
        train_set = datasets.KfoldDataset(ori, canny, age, male)
        print(train_set)
        ori1, canny1, age1, male1 = train_dataset[val_idx]
        val_set = datasets.KfoldDataset(ori1, canny1, age1, male1)
        print(val_set)

        run_fold(args, train_set, val_set, fold+1)
    return None


opt = get_class_args()
main(opt)
