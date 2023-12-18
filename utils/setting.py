import argparse

def get_args():
    parser = argparse.ArgumentParser('Unet pre-training script', add_help=False)
    parser.add_argument('--use_canny', default=0, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--save_ckpt_freq', default=200, type=int)

    # Model parameters
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--step_size', type=int, default=50,
                        help='learning rate reduce step (default: 50)')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='learning rate reduce ratio (default: 0.7)')
     
    # Dataset parameters
    parser.add_argument('--data_path', default='../archive/masked_crop/train/1', type=str,
                        help='dataset path')
    parser.add_argument('--save_path', default="checkpoint_ori_200.pth",
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    
    return parser.parse_args()


def get_demo_args():
    parser = argparse.ArgumentParser('Unet pre-training script', add_help=False)
    parser.add_argument('--data_path', default="../archive/masked_crop/canny/val", type=str)
    parser.add_argument('--save_path', default="./output/masked_canny.png", type=str)
    parser.add_argument('--chkpt', default="checkpoint\checkpoint_ori_200.pth", type=str)
    parser.add_argument('--input_size', default=512, type=int,
                        help='images input size for backbone')

    return parser.parse_args()


def get_class_args():
    parser = argparse.ArgumentParser('Unet classify script', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)

    # Model parameters
    parser.add_argument('--input_size', default=512, type=int,
                        help='images input size for backbone')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--step_size', type=int, default=50,
                        help='learning rate reduce step (default: 50)')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='learning rate reduce ratio (default: 0.7)')

    # Dataset parameters
    parser.add_argument('--data_path', default='../archive/masked_crop/train/1', type=str,
                        help='dataset path')
    parser.add_argument('--save_path', default="checkpoint_ori_200.pth",
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)

    return parser.parse_args()