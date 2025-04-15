"""
Most neural networks used in the paper are implemented here.
Note that the ConvNeXt code is based on the official implementation: https://github.com/facebookresearch/ConvNeXt
"""


import torch
import torch.nn as nn
from torch import autograd
from timm.models.layers import DropPath, trunc_normal_


class ConCodeTrans(nn.Module):
    def __init__(self, hid_dim, num_fea):
        super(ConCodeTrans, self).__init__()

        self.nets = nn.Sequential(
            nn.Linear(num_fea, hid_dim//16),
            nn.LeakyReLU(),
            nn.Linear(hid_dim//16, hid_dim//4),
            nn.LeakyReLU(),
            nn.Linear(hid_dim//4, hid_dim)
        )

    def forward(self, z, fea):
        return z + self.nets(fea)


class Discriminator(nn.Module):
    def __init__(self, channel_base):
        super(Discriminator, self).__init__()

        self.encoder_net = ConvNeXt(channel_base=channel_base)

    def forward(self, x):
        # (N, 4, 80, 128, 128)
        x = self.encoder_net(x)
        # (N, 1)
        return x


def calc_gradient_penalty(d_model, real_data, fake_data):
    if len(real_data.size()) == 5:
        alpha = torch.rand(real_data.size(0), 1, 1, 1, 1).cuda()
        alpha = alpha.expand(real_data.size())
    else:
        alpha = torch.rand(real_data.size(0), 1).cuda()
        alpha = alpha.expand(real_data.size())

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    disc_interpolates = d_model(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates,
                              inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True,
                              retain_graph=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gradient_penalty


class Generator(nn.Module):
    def __init__(self, channel_base):
        super(Generator, self).__init__()
        self.decoder_net = InvertConvNeXt(channel_base=channel_base)
        self.act = nn.Softmax(dim=1)

    def forward(self, z):
        # (N, hidden_dim)
        y_hat = self.decoder_net(z)
        # (N, 4, 80, 128, 128)

        y_hat = self.act(y_hat)
        # (N, 4, 80, 128, 128)
        return y_hat


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, data_format='channels_last'):
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.data_format = data_format
        if self.data_format not in ['channels_first', 'channels_last']:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == 'channels_last':
            return nn.functional.layer_norm(x, self.normalized_shape, self.weight, self.bias, 1e-6)
        elif self.data_format == 'channels_first':
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + 1e-6)
            if len(x.size()) == 4:
                x = self.weight[:, None, None] * x + self.bias[:, None, None]
            elif len(x.size()) == 5:
                x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
            else:
                raise NotImplementedError
            return x


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=(3, 5, 5), padding=(1, 2, 2), groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input_x = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 4, 1)  # (N, C, H, W, D) -> (N, H, W, D, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 4, 1, 2, 3)  # (N, H, W, D, C) -> (N, C, H, W, D)

        x = input_x + self.drop_path(x)
        return x


class InvertConvNeXt(nn.Module):
    def __init__(self, channel_base=32, out_channel=4, depths=(1, 1, 3, 1),
                 drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()

        hidden_dim = channel_base * 32
        dims = [channel_base * 32, channel_base * 8, channel_base * 2, out_channel]

        self.upsample_layers = nn.ModuleList()

        tmp_stem = nn.Sequential(
            nn.ConvTranspose3d(hidden_dim, dims[0], kernel_size=(10, 16, 16), groups=dims[0]),
            LayerNorm(dims[0], data_format="channels_first"),
        )
        self.upsample_layers.append(tmp_stem)

        for i in range(3):
            upsample_layer = nn.Sequential(
                LayerNorm(dims[i], data_format="channels_first"),
                nn.Upsample(scale_factor=2, mode='trilinear'),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=(3, 5, 5), padding=(1, 2, 2), groups=dims[i+1])
            )
            self.upsample_layers.append(upsample_layer)

        self.stages = nn.ModuleList()

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.upsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):

        # (N, hidden_dim)
        x = x.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        # (N, hidden_dim, 1, 1, 1)
        x = self.forward_features(x)
        # (N, 4, 80, 128, 128)

        return x


class ConvNeXt(nn.Module):
    def __init__(self, channel_base=32, in_channel=4, depths=(1, 1, 3, 1),
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_dim=1):
        super().__init__()

        dims = [channel_base // 2, channel_base * 2, channel_base * 8, channel_base * 32]

        self.downsample_layers = nn.ModuleList()

        stem = nn.Sequential(
            nn.Conv3d(in_channel, dims[0], kernel_size=(3, 5, 5), stride=2, padding=(1, 2, 2), groups=in_channel),
            LayerNorm(dims[0], data_format="channels_first")
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], data_format="channels_first"),
                nn.Conv3d(dims[i], dims[i+1], kernel_size=(3, 5, 5), stride=2, padding=(1, 2, 2), groups=dims[i])
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()

        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Sequential(
            nn.Linear(dims[-1], 4 * dims[-1]),
            nn.GELU(),
            nn.Linear(4 * dims[-1], out_dim)
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        # (N, in_channel, 80, 128, 128)
        x = self.forward_features(x)
        # (N, channel_base * 32, 5, 8, 8)
        x = x.mean([-3, -2, -1])
        # (N, channel_base * 32)
        x = self.norm(x)
        # (N, channel_base * 32)
        x = self.head(x)
        # (N, 1)
        return x


class AlphaEncoder(nn.Module):
    def __init__(self, channel_base):
        super(AlphaEncoder, self).__init__()

        self.encoder_net = ConvNeXt(channel_base=channel_base, out_dim=channel_base * 32)

    def forward(self, x):
        # (N, 4, 80, 128, 128)
        x = self.encoder_net(x)
        # (N, 1024)
        return x


class CodeDiscriminator(nn.Module):
    def __init__(self, channel_base):
        super().__init__()

        hidden_dim = channel_base * 32
        self.lin = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, 1),
        )

    def forward(self, z):
        return self.lin(z)
