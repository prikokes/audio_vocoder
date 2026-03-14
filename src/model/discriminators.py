import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm, spectral_norm


class DiscriminatorR(nn.Module):
    def __init__(self, resolution, channels=32, max_channels=512):
        super().__init__()
        self.resolution = resolution  # (n_fft, hop_length, win_length)
        n_fft, hop_length, win_length = resolution
        self.register_buffer("window", torch.hann_window(win_length))

        spec_channels = 2

        self.convs = nn.ModuleList()

        self.convs.append(
            weight_norm(nn.Conv2d(spec_channels, channels, kernel_size=(3, 9), padding=(1, 4)))
        )

        ch_in = channels
        for i in range(3):
            ch_out = min(ch_in * 2, max_channels)
            self.convs.append(
                weight_norm(nn.Conv2d(ch_in, ch_out, kernel_size=(3, 9), stride=(1, 2), padding=(1, 4)))
            )
            ch_in = ch_out

        ch_out = min(ch_in * 2, max_channels)
        self.convs.append(
            weight_norm(nn.Conv2d(ch_in, ch_out, kernel_size=(3, 3), padding=(1, 1)))
        )
        ch_in = ch_out

        self.conv_post = weight_norm(nn.Conv2d(ch_in, 1, kernel_size=(3, 3), padding=(1, 1)))

    def spectrogram(self, x):
        # x: (B, 1, T) -> (B, T)
        x = x.squeeze(1)
        n_fft, hop_length, win_length = self.resolution

        pad_amount = (n_fft - hop_length) // 2
        x = F.pad(x, (pad_amount, pad_amount), mode='reflect')

        spec = torch.stft(
            x, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            window=self.window, return_complex=True,
        )
        # (B, freq, time) -> split real/imag -> (B, 2, freq, time)
        spec = torch.stack([spec.real, spec.imag], dim=1)
        return spec

    def forward(self, x):
        fmaps = []
        x = self.spectrogram(x)

        for conv in self.convs:
            x = conv(x)
            x = F.leaky_relu(x, 0.1)
            fmaps.append(x)

        x = self.conv_post(x)
        fmaps.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmaps


class MultiResolutionDiscriminator(nn.Module):
    def __init__(self, resolutions=None, channels=32, max_channels=512):
        super().__init__()

        if resolutions is None:
            # (n_fft, hop_length, win_length)
            resolutions = [
                (1024, 120, 600),
                (2048, 240, 1200),
                (512, 50, 240),
            ]

        self.discriminators = nn.ModuleList([
            DiscriminatorR(res, channels=channels, max_channels=max_channels)
            for res in resolutions
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []

        for disc in self.discriminators:
            y_d_r, fmap_r = disc(y)
            y_d_g, fmap_g = disc(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True),
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i-1](y)
                y_hat = self.meanpools[i-1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
