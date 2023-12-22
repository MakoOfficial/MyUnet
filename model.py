import torch
import torch.nn as nn
import torch.nn.functional as F
from vit import Vit


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
        # self.conv3m = add_conv_stage(256, 128)
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
        conv3m_out = self.upsample43(conv4m_out) # 128, 128, 128
        # conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)    # 256, 128, 128
        # conv3m_out = self.conv3m(conv4m_out_)   # 128, 128, 128
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

        conv4_us = self.upsample54(conv5_out)  # 256, 64, 64
        conv3_us = self.upsample43(conv4_us)    # 128, 128, 128
        # conv2us = self.upsample32(conv3_us) # 64, 256, 256
        conv3m_out_ = torch.cat((self.upsample32(conv3_us), conv2_out), 1)    # 128, 256, 256
        conv2m_out = self.conv2m(conv3m_out_)   # 64, 256, 256
        conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)    # 64, 512, 512
        conv1m_out = self.conv1m(conv2m_out_)   # 32, 512, 512
        conv0_out = self.conv0(conv1m_out)  # 1, 512, 512

        return conv0_out





class Ori_Embedding(nn.Module):

    def __init__(self, backbone):
        super(Ori_Embedding, self).__init__()
        self.feature_extract = nn.ModuleList([])
        self.feature_extract.append(backbone.conv1)
        self.feature_extract.append(backbone.max_pool)
        self.feature_extract.append(backbone.conv2)
        self.feature_extract.append(backbone.max_pool)
        self.feature_extract.append(backbone.conv3)
        self.feature_extract.append(backbone.max_pool)
        self.feature_extract.append(backbone.conv4)
        self.feature_extract.append(backbone.max_pool)
        self.feature_extract.append(backbone.conv5)

        for param in self.feature_extract.parameters():
            param.requires_grad = False

        # add the patch embedding
        self.downSample = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, stride=2),
            # nn.BatchNorm2d(256),
            # nn.LayerNorm((16, 16)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.LayerNorm((16, 16)),
            nn.ReLU(),
            nn.Conv2d(256, 1024, kernel_size=3, padding=1),
            # nn.BatchNorm2d(1024)
            # nn.LayerNorm((16, 16))
        )

        self.res = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=2),
            # nn.BatchNorm2d(1024)
            # nn.LayerNorm((16, 16))
        )


    def forward(self, input):
        x = input
        for module in self.feature_extract:
            x = module(x)
        # print(f"After Ori_pretrain's shape: {x.shape}, and it's required [B, 512, 32, 32]")
        # x.shape = [B, 512, 32, 32]
        low_module = x
        # low_module = self.Patch_Embding(low_module)
        low_module = F.relu(self.downSample(low_module)+self.res(low_module))
        low_module = F.adaptive_avg_pool2d(low_module, 1)
        low_module = torch.squeeze(low_module)
        # print(f"After Ori_Embed's shape: {low_module.shape}, and it's required [B, 1024]")
        # low_module.shape = [B, 1024]
        return low_module


class BTNK1(nn.Module):

    def __init__(self, C_in, C_out, S) -> None:
        super(BTNK1, self).__init__()
        self.inner_C = int(C_out/4)
        self.downSample = nn.Sequential(
            nn.Conv2d(C_in, self.inner_C, kernel_size=1, stride=S),
            # nn.BatchNorm2d(256),
            # nn.LayerNorm((16, 16)),
            nn.ReLU(),
            nn.Conv2d(self.inner_C, self.inner_C, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.LayerNorm((16, 16)),
            nn.ReLU(),
            nn.Conv2d(self.inner_C, C_out, kernel_size=1),
            # nn.BatchNorm2d(1024)
            # nn.LayerNorm((16, 16))
        )

        self.res = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size=1, stride=S),
            # nn.BatchNorm2d(1024)
            # nn.LayerNorm((16, 16))
        )

    def forward(self, x):
        return F.relu((self.downSample(x)+self.res(x)))


class BTNK2(nn.Module):

    def __init__(self, C) -> None:
        super(BTNK2, self).__init__()
        self.inner_channels = int(C/4)
        self.innerLayer = nn.Sequential(
            nn.Conv2d(C, self.inner_channels, kernel_size=1),
            # nn.BatchNorm2d(256),
            # nn.LayerNorm((16, 16)),
            nn.ReLU(),
            nn.Conv2d(self.inner_channels, self.inner_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(256),
            # nn.LayerNorm((16, 16)),
            nn.ReLU(),
            nn.Conv2d(self.inner_channels, C, kernel_size=1),
            # nn.BatchNorm2d(1024)
            # nn.LayerNorm((16, 16))
        )

    def forward(self, x):
        return F.relu(self.innerLayer(x)+x)


class Ori_Embedding2(nn.Module):

    def __init__(self, backbone):
        super(Ori_Embedding2, self).__init__()
        self.feature_extract = nn.ModuleList([])
        self.feature_extract.append(backbone.conv1)
        self.feature_extract.append(backbone.max_pool)
        self.feature_extract.append(backbone.conv2)
        self.feature_extract.append(backbone.max_pool)
        self.feature_extract.append(backbone.conv3)
        self.feature_extract.append(backbone.max_pool)
        self.feature_extract.append(backbone.conv4)
        self.feature_extract.append(backbone.max_pool)
        self.feature_extract.append(backbone.conv5)

        for param in self.feature_extract.parameters():
            param.requires_grad = False

        self.residual = nn.Sequential(
            BTNK1(512, 1024, 2),
            BTNK2(1024),
            BTNK2(1024),
            BTNK2(1024)
        )


    def forward(self, input):
        x = input
        for module in self.feature_extract:
            x = module(x)
        # print(f"After Ori_pretrain's shape: {x.shape}, and it's required [B, 512, 32, 32]")
        # x.shape = [B, 512, 32, 32]
        # low_module = self.Patch_Embding(low_module)
        low_module = self.residual(x)
        low_module = F.adaptive_avg_pool2d(low_module, 1)
        low_module = torch.squeeze(low_module)
        # print(f"After Ori_Embed's shape: {low_module.shape}, and it's required [B, 1024]")
        # low_module.shape = [B, 1024]
        return low_module


class Canny_Embedding(nn.Module):

    def __init__(self, backbone):
        super(Canny_Embedding, self).__init__()
        self.feature_extract = nn.ModuleList([])
        self.feature_extract.append(backbone.conv1)
        self.feature_extract.append(backbone.max_pool)
        self.feature_extract.append(backbone.conv2)

        for param in self.feature_extract.parameters():
            param.requires_grad = False

        self.vit = Vit(input_size=256, patch_size=32, in_chans=64, embed_dim=1024, depth=6,
                       num_heads=12)

    def forward(self, input):
        feature = input
        for module in self.feature_extract:
            feature = module(feature)

        high_module = feature
        # print(f"After Canny_pretrain's shape: {high_module.shape}, and it's required [B, 64, 256, 256]")
        high_module = self.vit(high_module)
        # high_module.shape = [B, 1024]
        # print(f"After Canny_Embed's shape: {high_module.shape}, and it's required [B, 1024]")
        return high_module


class Canny_Embedding2(nn.Module):

    def __init__(self, backbone):
        super(Canny_Embedding2, self).__init__()
        self.feature_extract = nn.ModuleList([])
        self.feature_extract.append(backbone.conv1)
        self.feature_extract.append(backbone.max_pool)
        self.feature_extract.append(backbone.conv2)

        for param in self.feature_extract.parameters():
            param.requires_grad = False

        self.stage1 = nn.Sequential(
            BTNK1(64, 256, 2),
            BTNK2(256),
            BTNK2(256)
        )

        self.stage2 = nn.Sequential(
            BTNK1(256, 512, 2),
            BTNK2(512),
            BTNK2(512),
            BTNK2(512)
        )

        self.stage3 = nn.Sequential(
            BTNK1(512, 1024, 2),
            BTNK2(1024),
            BTNK2(1024),
            BTNK2(1024),
            BTNK2(1024),
            BTNK2(1024)
        )
        #
        # self.stage4 = nn.Sequential(
        #     BTNK1(1024, 2048, 2),
        #     BTNK2(2048),
        #     BTNK2(2048)
        # )


    def forward(self, input):
        feature = input
        for module in self.feature_extract:
            feature = module(feature)
        # print(f"After Canny_pretrain's shape: {feature.shape}, and it's required [B, 64, 256, 256]")

        # resnet
        high_module = self.stage1(feature)
        high_module = self.stage2(high_module)
        high_module = self.stage3(high_module)
        # high_module = self.stage4(high_module)

        high_module = F.adaptive_avg_pool2d(high_module, 1)
        high_module = torch.squeeze(high_module)

        # high_module.shape = [B, 1024]
        # print(f"After Canny_Embed's shape: {high_module.shape}, and it's required [B, 1024]")
        return high_module


class classifer(nn.Module):
    def __init__(self, backbone_ori, backbone_canny):
        super(classifer, self).__init__()
        self.feature_extract_ori = Ori_Embedding(backbone_ori)
        self.feature_extract_canny = Canny_Embedding(backbone_canny)

        self.gender_encoder = nn.Sequential(
            nn.Linear(1, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.MLP = nn.Sequential(
            nn.Linear(1024+1024+32, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, image, canny, gender):
        feature_ori = self.feature_extract_ori(image)
        feature_canny = self.feature_extract_canny(canny)

        gender_encode = self.gender_encoder(gender)

        # print(feature_ori.shape, feature_canny.shape, gender_encode.shape)
        feature_fusion = torch.cat((feature_ori, feature_canny, gender_encode), dim=1)  # 512+512+32
        # print(feature_fusion.shape)
        return self.MLP(feature_fusion)


class Align(nn.Module):
    def __init__(self, backbone_ori, backbone_canny):
        super(Align, self).__init__()
        self.feature_extract_ori = Ori_Embedding(backbone_ori)
        self.feature_extract_canny = Canny_Embedding(backbone_canny)


    def forward(self, image, canny, gender):
        feature_ori = image.clone()
        feature_canny = canny.clone()
        feature_ori = self.feature_extract_ori(feature_ori)
        feature_canny = self.feature_extract_canny(feature_canny)

        gender_encode = self.gender_encoder(gender)

        # print(feature_ori.shape, feature_canny.shape, gender_encode.shape)
        feature_fusion = torch.cat((feature_ori, feature_canny, gender_encode), dim=1)  # 512+512+32
        # print(feature_fusion.shape)
        return self.MLP(feature_fusion)


class distillation(nn.Module):
    def __init__(self, backbone):
        super(distillation, self).__init__()
        # ori branch
        self.feature_extract_ori = Ori_Embedding(backbone)

        self.gender_encoder = nn.Sequential(
            nn.Linear(1, 32),
            # nn.BatchNorm1d(32),
            # nn.LayerNorm(32),
            nn.ReLU()
        )

        self.MLP = nn.Sequential(
            nn.Linear(1024 + 32, 512),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            # nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x, gender):
        feature = self.feature_extract_ori(x)

        gender_encode = self.gender_encoder(gender)

        fusion = torch.cat((feature, gender_encode), dim=1)

        return self.MLP(fusion)


class distillation_canny(nn.Module):
    def __init__(self, backbone):
        super(distillation_canny, self).__init__()
        # ori branch
        self.feature_extract_canny = Canny_Embedding(backbone)

        self.gender_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.MLP = nn.Sequential(
            nn.Linear(1024 + 32, 512),
            # nn.BatchNorm1d(1024),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, x, gender):
        feature = self.feature_extract_canny(x)

        gender_encode = self.gender_encoder(gender)

        fusion = torch.cat((feature, gender_encode), dim=1)

        return self.MLP(fusion)

class classifer2(nn.Module):
    def __init__(self, backbone_ori, backbone_canny):
        super(classifer2, self).__init__()
        self.feature_extract_ori = Ori_Embedding2(backbone_ori)
        self.feature_extract_canny = Canny_Embedding2(backbone_canny)

        self.gender_encoder = nn.Sequential(
            nn.Linear(1, 32),
            # nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.MLP = nn.Sequential(
            nn.Linear(1024+1024+32, 1024),
            # nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            # nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, image, canny, gender):
        feature_ori = self.feature_extract_ori(image)
        feature_canny = self.feature_extract_canny(canny)

        gender_encode = self.gender_encoder(gender)

        # print(feature_ori.shape, feature_canny.shape, gender_encode.shape)
        feature_fusion = torch.cat((feature_ori, feature_canny, gender_encode), dim=1)  # 1024+2048+32
        # print(feature_fusion.shape)
        return self.MLP(feature_fusion)
