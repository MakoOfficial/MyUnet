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


def main(args):
    torch.manual_seed(args.seed)
    checkpoint_ori = torch.load(args.ori_ckpt_path)
    checkpoint_canny = torch.load(args.canny_ckpt_path)

    classifer = model.classifer(checkpoint_ori, checkpoint_canny)

    df = pd.read_csv(args.csv_path)
    train_ori_dir = args.ori_train_path
    train_canny_dir = args.canny_train_path
    train_trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    val_ori_dir = args.ori_val_path
    val_canny_dir = args.canny_val_path
    val_trans = transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ClassDataset(df=df, ori_dir=train_ori_dir, canny_dir=train_canny_dir, transform=train_trans)
    print(train_dataset)
    sampler = torch.utils.data.RandomSampler(data_source=train_dataset)
    print(sampler)

    val_dataset = datasets.ClassDataset(df=df, ori_dir=val_ori_dir, canny_dir=val_canny_dir, transform=val_trans)
    print(val_dataset)

    train_loader = data.dataloader.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        drop_last=True
    )

    val_loader = data.dataloader.DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False
    )

    loss_func = nn.L1Loss(reduction="sum")
    train_length = train_dataset.__len__()
    epochs = args.epochs
    optimizer = Adam(classifer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    for epoch in range(epochs):
        classifer.cuda()
        total_loss = 0.
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

        # valid process
        classifer.eval()
        val_loss = 0.
        val_length = 0.
        with torch.no_grad():
            for idx, patch in enumerate(val_loader):
                patch_len = patch[0].shape[0]
                images = patch[0].cuda()
                cannys = patch[1].cuda()
                boneage = patch[2].cuda()
                male = patch[3].cuda()
                output = classifer(images, cannys, male)

                loss = loss_func(output, boneage)
                val_loss += loss.item()
                val_length += patch_len

        print(f'epoch {epoch+1}: training loss: {round(total_loss/train_length, 3)}, '
              f'valid loss: {round(val_loss/val_length, 3)}, lr:{optimizer.param_groups[0]["lr"]}')
        scheduler.step()

        if int((epoch+1) % args.save_ckpt_freq) == 0:
            save_name = os.path.join(args.save_path, args.save_name)
            torch.save(classifer, f'{save_name}_{epoch+1}.pth')
    return None


opt = get_class_args()
main(opt)
