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


def main(args):
    print(args)
    torch.manual_seed(args.seed)
    print(f'Set manual random seed: {args.seed}')
    checkpoint_ori = torch.load(args.ori_ckpt_path)
    print(f'Load MyUnet_Ori from {args.ori_ckpt_path}')
    checkpoint_canny = torch.load(args.canny_ckpt_path)
    print(f'Load MyUnet_Canny from {args.canny_ckpt_path}')

    classifer = model.classifer(checkpoint_ori, checkpoint_canny)
    print(f'Model:\n{classifer}')
    print(f'number of training params: {sum(p.numel() for p in classifer.parameters() if p.requires_grad) / 1e6} M')

    df = pd.read_csv(args.csv_path)
    df, boneage_mean, boneage_div = normalize_age(df)
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
    print(f'Training dataset info:\n{train_dataset}')
    sampler = torch.utils.data.RandomSampler(data_source=train_dataset)

    val_dataset = datasets.ClassDataset(df=df, ori_dir=val_ori_dir, canny_dir=val_canny_dir, transform=val_trans)
    print(f'Valid dataset info:\n{val_dataset}')

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

    num_training_steps_per_epoch = len(train_dataset) // args.batch_size

    print(f"LR = {args.lr}")
    print(f"Batch size = {args.batch_size}")
    print(f"Number of training steps = {num_training_steps_per_epoch}")
    print(f"Number of training examples per epoch = {num_training_steps_per_epoch*args.batch_size}")

    loss_func = nn.L1Loss(reduction="sum")
    train_length = train_dataset.__len__()
    epochs = args.epochs
    optimizer = Adam(classifer.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("Use step level LR & WD scheduler!")
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    print(f'Scheduler:\n{scheduler}')


    for epoch in range(epochs):
        classifer.cuda()
        classifer.train()
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

        print(f'epoch {epoch+1}: training loss: {round(total_loss/train_length, 3)}, '
              f'valid loss: {round(eval_func(classifer, val_loader, boneage_mean, boneage_div), 3)}, '
              f'lr:{optimizer.param_groups[0]["lr"]}')
        scheduler.step()

        if int((epoch+1) % args.save_ckpt_freq) == 0:
            save_name = os.path.join(args.save_path, args.save_name)
            torch.save(classifer, f'{save_name}_{epoch+1}.pth')
    return None


opt = get_class_args()
main(opt)
