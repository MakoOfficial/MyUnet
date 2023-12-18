import torch
from torch import nn
import torch.nn.functional as F
import numpy as np



def get_sinusoid_encoding_table(n_position, d_hid):
    ''' Sinusoid position encoding table '''
    # TODO: make it with torch instead of numpy
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)



class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        self.q_bias = None
        self.v_bias = None

        self.proj = nn.Linear(all_head_dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)


    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patchd Embedding
    """
    def __init__(self, input_size=256, patch_size=32, in_chans=64, embed_dim=1024):
        super().__init__()
        num_patches = (input_size // patch_size) ** 2
        self.patch_shape = input_size // patch_size
        self.input_size = input_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class Vit(nn.Module):
    def __init__(self, input_size=256, patch_size=32, in_chans=64, embed_dim=1024, depth=6,
                 num_heads=12, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_size = input_size
        self.patch_embed = PatchEmbed(
            input_size=input_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches

        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        self.blocks = nn.ModuleList(
            [Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=4) for i in range(depth)]
        )

        self.norm = norm_layer(embed_dim)

        self.to_latent = nn.Linear(embed_dim, embed_dim, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, x):
        print(f"2:{x.shape}")
        x = self.patch_embed(x)
        print(f"3:{x.shape}")
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        for blk in self.blocks:
            x = blk(x)
        x = x.mean(dim=1)

        return self.to_latent(x)

