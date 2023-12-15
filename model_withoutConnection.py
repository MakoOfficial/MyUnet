import torch
import torch.nn as nn
import torch.nn.functional as functional


def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU(),
        nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.ReLU()
    )

def upsample(ch_coarse, ch_fine):
    return nn.Sequential(
        nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
        nn.ReLU()
    )

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        self.conv1 = add_conv_stage(1, 32)
        self.conv2 = add_conv_stage(32, 64)
        self.conv3 = add_conv_stage(64, 128)
        self.conv4 = add_conv_stage(128, 256)
        self.conv5 = add_conv_stage(256, 512)

        self.output = nn.Sequential(
          nn.Conv2d(32, 1, 3, 1, 1),
          # nn.Sigmoid()
          # nn.ReLU()
        )

        self.max_pool = nn.MaxPool2d(2)

        self.upsample5 = upsample(512, 256)
        self.upsample4 = upsample(256, 128)
        self.upsample3 = upsample(128, 64)
        self.upsample2 = upsample(64, 32)


    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))
        conv4_out = self.conv4(self.max_pool(conv3_out))
        conv5_out = self.conv5(self.max_pool(conv4_out))

        upsam_out5 = self.upsample5(conv5_out)
        upsam_out4 = self.upsample4(upsam_out5)
        upsam_out3 = self.upsample3(upsam_out4)
        upsam_out2 = self.upsample2(upsam_out3)


        return self.output(upsam_out2)

if __name__ == '__main__':
    net = UNet()
    total = sum([param.nelement() for param in net.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    x = torch.ones([1, 3, 224, 224])
    out = net(x)
    print(out)
    # print(net)