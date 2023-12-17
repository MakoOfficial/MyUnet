import torch.utils.data as data
import torch.optim as optim
from torchvision import transforms
import time
import torch

from torch.optim.lr_scheduler import StepLR

from utils import datasets, func, setting


def train(args):

    torch.manual_seed(args.seed)

    net_Ori, net_Canny = func.get_align()
    net_Ori.cuda()
    net_Canny.cuda()
    net_Ori.eval()
    net_Canny.eval()


    trans = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        datasets.resize(args.input_size),
        # normal(),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])

    dataset = datasets.UnetDataset(args.data_path, transform=trans)
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


# =====================================================================
opts = setting.get_args()
train(opts)
