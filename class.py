import numpy as np
import torch
import model
import os
from torch.utils import data
from PIL import Image
import pandas as pd
from torchvision import transforms
from PIL import ImageOps
from torch import nn
from torch.optim import Adam
from utils import datasets

torch.manual_seed(0)


def main():
    checkpoint_ori = torch.load("./checkpoint/masked_1K/masked_1K_ori_200.pth")
    checkpoint_canny = torch.load("./checkpoint/masked_1K/masked_1K_canny_200.pth")

    classifer = model.classifer(checkpoint_ori, checkpoint_canny)

    df = pd.read_csv('../archive/boneage-training-dataset.csv')
    ori_dir = "../archive/masked_1K_train/ori"
    canny_dir = "../archive/masked_1K_train/canny"
    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        # datasets.resize(512),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ClassDataset(df=df, ori_dir=ori_dir, canny_dir=canny_dir, transform=trans)
    print(train_dataset)
    sampler = torch.utils.data.RandomSampler(data_source=train_dataset)
    print(sampler)

    train_loader = data.dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=2,
        sampler=sampler,
        drop_last=True
    )


    loss_func = nn.L1Loss(reduction="sum")
    total_loss = 0.
    length = train_dataset.__len__()
    epochs = 2
    optimizer = Adam(classifer.parameters(), lr=1e-4, weight_decay=1e-5)
    for epoch in range(epochs):
        classifer.cuda()
        for idx, patch in enumerate(train_loader):
            images = patch[0].cuda()
            cannys = patch[1].cuda()
            boneage = patch[2].cuda()
            male = patch[3].cuda()
            optimizer.zero_grad()

            output = classifer(images, cannys, male)
            loss = loss_func(output, boneage)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'epoch {epoch+1}: loss is {total_loss/length}')

    return None


main()