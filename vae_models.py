"""
Models for VAE and CVAE
"""


import torch
import torch.nn as nn
from torch import FloatTensor
import torch.distributions as dists
from models import InvertConvNeXt, ConvNeXt


class VaeConvNeXt(ConvNeXt):
    def __init__(self, channel_base=32):
        super().__init__(channel_base)

        hidden_dim = channel_base * 32
        self.head_mean = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )
        self.head_log_var = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

    def forward(self, x):
        # (N, in_channel, 80, 128, 128)
        x = self.forward_features(x)
        # (N, channel_base * 32, 5, 8, 8)
        x = x.mean([-3, -2, -1])
        # (N, channel_base * 32)
        x = self.norm(x)
        # (N, channel_base * 32)

        z_loc = self.head_mean(x)
        # (N, channel_base * 32)
        z_std = torch.exp(self.head_log_var(x)).pow(0.5) + 1e-16
        # (N, channel_base * 32)
        z_dist = dists.normal.Normal(z_loc, z_std)
        return z_dist


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
        y_hat = torch.clamp(y_hat, 0., 1.)
        return y_hat


class BaseEncoder(nn.Module):
    def __init__(self, channel_base: int):
        super(BaseEncoder, self).__init__()
        self.encoder_net = VaeConvNeXt(channel_base=channel_base)

    def con_forward(self, x):
        # (N, in_channel, 80, 128, 128)
        x = self.encoder_net.forward_features(x)
        # (N, channel_base * 32, 5, 8, 8)
        x = x.mean([-3, -2, -1])
        # (N, channel_base * 32)
        x = self.encoder_net.norm(x)
        # (N, channel_base * 32)
        return x

    def con_dist(self, x):
        z_loc = self.encoder_net.head_mean(x)
        # (N, channel_base * 32)
        z_std = torch.exp(self.encoder_net.head_log_var(x)).pow(0.5) + 1e-16
        # (N, channel_base * 32)
        return z_loc, z_std

    def forward(self, x):
        # (N, 4, 80, 128, 128)
        z_dist = self.encoder_net(x)
        return z_dist


class BaseAE(nn.Module):
    def __init__(self, channel_base: int):
        super(BaseAE, self).__init__()
        self.encoder = BaseEncoder(channel_base=channel_base)
        self.decoder = Generator(channel_base=channel_base)

    def forward(self, mris: FloatTensor):
        # (N, 4, 80, 128, 128)
        z_dist = self.encoder(mris)
        z_hat = z_dist.rsample()
        # (N, channel_base * 32)

        y_hat = self.decoder(z_hat)
        # (N, 4, 80, 128, 128)

        target_dist = dists.normal.Normal(torch.zeros_like(z_hat), torch.ones_like(z_hat))
        kl_loss = dists.kl.kl_divergence(z_dist, target_dist)

        return kl_loss, y_hat


class Prior(nn.Module):
    def __init__(self, num_fea, channel_base):
        super(Prior, self).__init__()
        hid_dim = channel_base * 32

        self.head_mean = nn.Sequential(
            nn.Linear(num_fea, hid_dim//16),
            nn.LeakyReLU(),
            nn.Linear(hid_dim//16, hid_dim//4),
            nn.LeakyReLU(),
            nn.Linear(hid_dim//4, hid_dim)
        )

        self.head_log_var = nn.Sequential(
            nn.Linear(num_fea, hid_dim // 16),
            nn.LeakyReLU(),
            nn.Linear(hid_dim // 16, hid_dim // 4),
            nn.LeakyReLU(),
            nn.Linear(hid_dim // 4, hid_dim)
        )

    def forward(self, fea):
        z_loc = self.head_mean(fea)
        # (N, channel_base * 32)
        z_std = torch.exp(self.head_log_var(fea)).pow(0.5) + 1e-16
        # (N, channel_base * 32)

        return z_loc, z_std

