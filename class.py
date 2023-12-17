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

from torchkeras import summary

torch.manual_seed(0)

def main():
    checkpoint_ori = torch.load("./checkpoint/shortcut/checkpoint_ori_200.pth")
    checkpoint_canny = torch.load("./checkpoint/shortcut/checkpoint_canny_200.pth")

    classifer = model.classifer(checkpoint_ori, checkpoint_canny)
    print(summary(classifer, input_shape=(15,)))

    df = pd.read_csv('../archive/boneage-training-dataset.csv')
    data_dir = "../archive/masked_crop/train/1"
    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        datasets.resize(512),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ClassDataset(df, data_dir, trans)
    train_loader = data.dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=5
    )

    loss_func = nn.L1Loss(reduction="sum")
    total_loss = 0.
    length = train_dataset.__len__()
    epochs = 10
    optimizer = Adam(classifer.parameters(), lr=1e-4, weight_decay=1e-5)
    for epoch in range(epochs):
        classifer.cuda()
        for idx, patch in enumerate(train_loader):
            images = patch[0].cuda()
            boneage = patch[1].cuda()
            male = patch[2].cuda()
            optimizer.zero_grad()

        #     output = classifer(images, male)
        #     loss = loss_func(output, boneage)
        #     loss.backward()
        #     optimizer.step()
        #     total_loss += loss.item()
        # print(f'epoch {epoch+1}: loss is {total_loss/length}')

    return None

main()