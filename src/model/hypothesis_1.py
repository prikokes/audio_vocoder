import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm, remove_weight_norm
from librosa.filters import mel as librosa_mel_fn

from src.model.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator


LRELU_SLOPE = 0.1


class PseudoInverseMelFilter(nn.Module):
    def __init__(self, sr=22050, n_fft=1024, n_mels=80, fmin=0.0, fmax=8000.0):
        super().__init__()
        mel_basis = librosa_mel_fn(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        mel_basis_pinv = np.linalg.pinv(mel_basis)
        self.register_buffer("mel_basis_pinv", torch.FloatTensor(mel_basis_pinv))

    def forward(self, mel):
        mel_linear = torch.exp(mel)
        magnitude = torch.matmul(self.mel_basis_pinv, mel_linear)
        magnitude = torch.clamp(magnitude, min=1e-7)
        return magnitude


class InitialSignalReconstruction(nn.Module):
    def __init__(self, n_fft=1024, hop_length=256, win_length=1024):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, magnitude):
        random_phase = 2 * np.pi * torch.rand_like(magnitude) - np.pi
        complex_spec = magnitude * torch.exp(1j * random_phase)

        waveform = torch.istft(
            complex_spec, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
        )

        stft_out = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window, return_complex=True,
        )

        mag = stft_out.abs()
        phase = torch.angle(stft_out)
        return mag, phase


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for d in dilation:
            self.convs1.append(
                weight_norm(nn.Conv1d(channels, channels, kernel_size,
                    dilation=d, padding=(kernel_size * d - d) // 2))
            )
            self.convs2.append(
                weight_norm(nn.Conv1d(channels, channels, kernel_size,
                    dilation=1, padding=(kernel_size - 1) // 2))
            )

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        for c in self.convs1:
            remove_weight_norm(c)
        for c in self.convs2:
            remove_weight_norm(c)


class ISTFTNetModule(nn.Module):
    def __init__(self, n_fft=16, hop_length=4):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, magnitude):
        zero_phase = torch.zeros_like(magnitude)
        complex_spec_init = magnitude * torch.exp(1j * zero_phase)

        waveform_intermediate = torch.istft(
            complex_spec_init,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
        )

        stft_out = torch.stft(
            waveform_intermediate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            return_complex=True,
        )

        extracted_phase = torch.angle(stft_out)

        T = min(magnitude.shape[-1], extracted_phase.shape[-1])
        magnitude = magnitude[..., :T]
        extracted_phase = extracted_phase[..., :T]

        complex_spec_final = magnitude * torch.exp(1j * extracted_phase)

        waveform = torch.istft(
            complex_spec_final,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
        )

        return waveform
    

class FreeVGeneratorH1(nn.Module):
    def __init__(
        self, sr=22050, n_fft=1024, hop_length=256, win_length=1024,
        n_mels=80, fmin=0.0, fmax=8000.0, upsample_initial_channel=512,
        upsample_rates=(8, 8), upsample_kernel_sizes=(16, 16),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        istft_n_fft=16, istft_hop_length=4,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.istft_n_fft = istft_n_fft
        self.istft_hop_length = istft_hop_length
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.pimf = PseudoInverseMelFilter(
            sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax
        )
        self.isr = InitialSignalReconstruction(
            n_fft=n_fft, hop_length=hop_length, win_length=win_length
        )

        spec_channels = n_fft // 2 + 1
        input_channels = spec_channels * 2

        self.conv_pre = weight_norm(
            nn.Conv1d(input_channels, upsample_initial_channel, 7, padding=3)
        )

        self.ups = nn.ModuleList()
        ch = upsample_initial_channel
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(nn.ConvTranspose1d(ch, ch // 2, k, stride=u, padding=(k - u) // 2))
            )
            ch = ch // 2

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch_cur = upsample_initial_channel // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch_cur, k, d))

        post_channels = ch_cur
        istft_freq_bins = istft_n_fft // 2 + 1

        self.mag_conv = weight_norm(
            nn.Conv1d(post_channels, istft_freq_bins, 7, padding=3)
        )

        self.istft_module = ISTFTNetModule(
            n_fft=istft_n_fft, hop_length=istft_hop_length
        )

    def forward(self, mel):
        magnitude = self.pimf(mel)
        mag, phase = self.isr(magnitude)

        T = min(mag.shape[-1], mel.shape[-1])
        mag = mag[..., :T]
        phase = phase[..., :T]

        x = torch.cat([mag, phase], dim=1)
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels

        x = F.leaky_relu(x)

        pred_mag = torch.exp(self.mag_conv(x))

        audio = self.istft_module(pred_mag)

        audio = audio.unsqueeze(1)
        audio = torch.clamp(audio, -1.0, 1.0)
        return audio

    def remove_weight_norm(self):
        for up in self.ups:
            remove_weight_norm(up)
        for block in self.resblocks:
            block.remove_weight_norm()
        remove_weight_norm(self.conv_pre)
        remove_weight_norm(self.mag_conv)


class FreeVH1(nn.Module):
    """FreeV с Hypothesis 1: Indirect Phase Prediction."""
    def __init__(self, generator_params, mpd_params=None, mrd_params=None):
        super().__init__()
        self.generator = FreeVGeneratorH1(**generator_params)
        self.mpd = MultiPeriodDiscriminator(**(mpd_params or {}))
        self.mrd = MultiResolutionDiscriminator(**(mrd_params or {}))
        self.discriminators = nn.ModuleList([self.mpd, self.mrd])

    def forward(self, mel=None, audio_real=None, **kwargs):
        if mel is None:
            if 'mel' in kwargs:
                mel = kwargs['mel']
            else:
                raise ValueError("Mel spectrogram input is required")

        audio_generated = self.generator(mel)

        if audio_real is not None:
            mpd_real, mpd_fake, mpd_real_fmaps, mpd_fake_fmaps = self.mpd(audio_real, audio_generated)
            mrd_real, mrd_fake, mrd_real_fmaps, mrd_fake_fmaps = self.mrd(audio_real, audio_generated)

            return {
                'audio_generated': audio_generated,
                'mpd_real': mpd_real,
                'mpd_fake': mpd_fake,
                'mrd_real': mrd_real,
                'mrd_fake': mrd_fake,
                'mpd_real_fmaps': mpd_real_fmaps,
                'mpd_fake_fmaps': mpd_fake_fmaps,
                'mrd_real_fmaps': mrd_real_fmaps,
                'mrd_fake_fmaps': mrd_fake_fmaps,
            }
        else:
            return {'audio_generated': audio_generated}

    def inference(self, mel):
        return self.forward(mel=mel)

    def remove_weight_norm(self):
        self.generator.remove_weight_norm()

    @property
    def discriminator(self):
        return self.discriminators
