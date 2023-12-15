import torch
import torch.nn as nn


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

        self.conv4m = add_conv_stage(512, 256)
        self.conv3m = add_conv_stage(256, 128)
        self.conv2m = add_conv_stage(128, 64)
        self.conv1m = add_conv_stage(64, 32)

        self.conv0 = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            # nn.Sigmoid()
            # nn.ReLU()
        )

        self.max_pool = nn.MaxPool2d(2)

        self.upsample54 = upsample(512, 256)
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128, 64)
        self.upsample21 = upsample(64, 32)

    def forward(self, x):
        conv1_out = self.conv1(x)  # 32, 512, 512
        conv2_out = self.conv2(self.max_pool(conv1_out))  # 64, 256, 256
        conv3_out = self.conv3(self.max_pool(conv2_out))  # 128, 128, 128
        conv4_out = self.conv4(self.max_pool(conv3_out))  # 256, 64, 64
        conv5_out = self.conv5(self.max_pool(conv4_out))  # 512, 32, 32

        conv5m_out = torch.cat((self.upsample54(conv5_out), conv4_out), 1)  # 512, 64, 64
        conv4m_out = self.conv4m(conv5m_out)  # 256, 64, 64
        conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)    # 256, 128, 128
        conv3m_out = self.conv3m(conv4m_out_)  # 128, 128, 128
        conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)    # 128, 256, 256
        conv2m_out = self.conv2m(conv3m_out_)   # 64, 256, 256
        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)    # 64, 512, 512
        conv1m_out = self.conv1m(conv2m_out_)   # 32, 512, 512
        conv0_out = self.conv0(conv1m_out)  # 1, 512, 512

        return conv0_out

    def fusion(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(self.max_pool(conv1_out))
        conv3_out = self.conv3(self.max_pool(conv2_out))
        conv4_out = self.conv4(self.max_pool(conv3_out))
        conv5_out = self.conv5(self.max_pool(conv4_out))

        return conv5_out

class UNet_ori(nn.Module):
    def __init__(self):
        super(UNet_ori, self).__init__()

        self.conv1 = add_conv_stage(1, 32)
        self.conv2 = add_conv_stage(32, 64)
        self.conv3 = add_conv_stage(64, 128)
        self.conv4 = add_conv_stage(128, 256)
        self.conv5 = add_conv_stage(256, 512)

        self.conv4m = add_conv_stage(512, 256)
        self.conv3m = add_conv_stage(256, 128)
        # self.conv2m = add_conv_stage(128, 64)
        # self.conv1m = add_conv_stage(64, 32)

        self.conv0 = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            # nn.Sigmoid()
            # nn.ReLU()
        )

        self.max_pool = nn.MaxPool2d(2)

        self.upsample54 = upsample(512, 256)
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128, 64)
        self.upsample21 = upsample(64, 32)

    def forward(self, x):
        conv1_out = self.conv1(x)   # 32, 512, 512
        conv2_out = self.conv2(self.max_pool(conv1_out))    # 64, 256, 256
        conv3_out = self.conv3(self.max_pool(conv2_out))    # 128, 128, 128
        conv4_out = self.conv4(self.max_pool(conv3_out))    # 256, 64, 64
        conv5_out = self.conv5(self.max_pool(conv4_out))    # 512, 32, 32

        conv5m_out = torch.cat((self.upsample54(conv5_out), conv4_out), 1)  # 512, 64, 64
        conv4m_out = self.conv4m(conv5m_out)    # 256, 64, 64
        # conv3m_out = self.upsample43(conv4m_out) # 128, 128, 128
        conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)    # 256, 128, 128
        conv3m_out = self.conv3m(conv4m_out_)   # 128, 128, 128
        #   ori, delete last 2 joint.
        conv3m_out_ = self.upsample32(conv3m_out)  # 64, 256, 256
        conv2m_out_ = self.upsample21(conv3m_out_)  # 32, 512, 512
        conv0_out = self.conv0(conv2m_out_)  # 1, 512, 512

        return conv0_out


class UNet_canny(nn.Module):
    def __init__(self):
        super(UNet_canny, self).__init__()

        self.conv1 = add_conv_stage(1, 32)
        self.conv2 = add_conv_stage(32, 64)
        self.conv3 = add_conv_stage(64, 128)
        self.conv4 = add_conv_stage(128, 256)
        self.conv5 = add_conv_stage(256, 512)

        # self.conv4m = add_conv_stage(512, 256)
        # self.conv3m = add_conv_stage(256, 128)
        # self.conv2m = add_conv_stage(128, 64)
        self.conv1m = add_conv_stage(64, 32)

        self.conv0 = nn.Sequential(
            nn.Conv2d(32, 1, 3, 1, 1),
            # nn.Sigmoid()
            # nn.ReLU()
        )

        self.max_pool = nn.MaxPool2d(2)

        self.upsample54 = upsample(512, 256)
        self.upsample43 = upsample(256, 128)
        self.upsample32 = upsample(128, 64)
        self.upsample21 = upsample(64, 32)

    def forward(self, x):
        conv1_out = self.conv1(x)  # 32, 512, 512
        conv2_out = self.conv2(self.max_pool(conv1_out))  # 64, 256, 256
        conv3_out = self.conv3(self.max_pool(conv2_out))  # 128, 128, 128
        conv4_out = self.conv4(self.max_pool(conv3_out))  # 256, 64, 64
        conv5_out = self.conv5(self.max_pool(conv4_out))  # 512, 32, 32

        conv4_us = self.upsample54(conv5_out)  # 256, 64, 64
        conv3_us = self.upsample43(conv4_us)    # 128, 128, 128
        conv2us = self.upsample32(conv3_us) # 64, 256, 256
        # conv3m_out_ = torch.cat((self.upsample32(conv3_us), conv2_out), 1)    # 128, 256, 256
        # conv2m_out = self.conv2m(conv3m_out_)   # 64, 256, 256
        conv2m_out_ = torch.cat((self.upsample21(conv2us), conv1_out), 1)    # 64, 512, 512
        conv1m_out = self.conv1m(conv2m_out_)   # 32, 512, 512
        conv0_out = self.conv0(conv1m_out)  # 1, 512, 512

        return conv0_out



class classifer(nn.Module):
    def __init__(self, backbone_ori, backbone_canny):
        super(classifer, self).__init__()
        self.feature_extract_ori = nn.ModuleList()
        self.feature_extract_ori.append(backbone_ori.conv1)
        self.feature_extract_ori.append(backbone_ori.conv2)
        self.feature_extract_ori.append(backbone_ori.conv3)
        self.feature_extract_ori.append(backbone_ori.conv4)
        self.feature_extract_ori.append(backbone_ori.conv5)

        self.feature_extract_canny = nn.ModuleList()
        self.feature_extract_canny.append(backbone_canny.conv1)
        self.feature_extract_canny.append(backbone_canny.conv2)
        self.feature_extract_canny.append(backbone_canny.conv3)
        self.feature_extract_canny.append(backbone_canny.conv4)
        self.feature_extract_canny.append(backbone_canny.conv5)

        for param in self.feature_extract_ori.parameters():
            param.requires_grad = False
        for param in self.feature_extract_canny.parameters():
            param.requires_grad = False

        self.fusion = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.gender_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.MLP = nn.Sequential(
            nn.Linear(1024+32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, image, gender):
        feature_ori = self.feature_extract_ori(image)
        feature_canny = self.feature_extract_canny(image)

        gender_encode = self.gender_encoder(gender)

        feature_fusion = torch.cat((feature_ori, feature_canny, gender_encode), dim=1)  # 512+512+32
        return self.MLP(feature_fusion)