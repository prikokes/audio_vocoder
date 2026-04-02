import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import weight_norm, remove_weight_norm

from src.model.freev import PseudoInverseMelFilter, InitialSignalReconstruction, ResBlock
from src.model.discriminators import MultiResolutionDiscriminator, MultiPeriodDiscriminator

LRELU_SLOPE = 0.1

class FreeVGeneratorH1(nn.Module):
    def __init__(
        self, sr=22050, n_fft=1024, hop_length=256, win_length=1024,
        n_mels=80, fmin=0.0, fmax=8000.0, upsample_initial_channel=512,
        upsample_rates=(8, 8), upsample_kernel_sizes=(16, 16),
        resblock_kernel_sizes=(3, 7, 11),
        resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        istft_n_fft=128, istft_hop_length=4,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.istft_n_fft = istft_n_fft
        self.istft_hop_length = istft_hop_length
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.pimf = PseudoInverseMelFilter(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
        self.isr = InitialSignalReconstruction(n_fft=n_fft, hop_length=hop_length, win_length=win_length)

        spec_channels = n_fft // 2 + 1
        input_channels = spec_channels * 3

        self.conv_pre = weight_norm(nn.Conv1d(input_channels, upsample_initial_channel, 7, padding=3))

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

        self.mag_conv = weight_norm(nn.Conv1d(post_channels, istft_freq_bins, 7, padding=3))
        
        self.inter_real_conv = weight_norm(nn.Conv1d(post_channels, istft_freq_bins, 7, padding=3))
        self.inter_imag_conv = weight_norm(nn.Conv1d(post_channels, istft_freq_bins, 7, padding=3))
        
        self.register_buffer("istft_window", torch.hann_window(istft_n_fft))

    def forward(self, mel):
        magnitude = self.pimf(mel)
        mag, phase = self.isr(magnitude)

        T = min(mag.shape[-1], mel.shape[-1])
        mag = mag[..., :T]
        phase = phase[..., :T]

        x = torch.cat([mag, torch.cos(phase), torch.sin(phase)], dim=1)
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

        mag_logits = torch.clamp(self.mag_conv(x), max=10.0)
        pred_mag = torch.exp(mag_logits)

        inter_real = self.inter_real_conv(x)
        inter_imag = self.inter_imag_conv(x)
        inter_complex = torch.complex(inter_real, inter_imag)

        T_out = pred_mag.shape[-1]
        expected_audio_length = (T_out - 1) * self.istft_hop_length

        inter_audio = torch.istft(
            inter_complex, n_fft=self.istft_n_fft, hop_length=self.istft_hop_length,
            win_length=self.istft_n_fft, window=self.istft_window,
            center=True, return_complex=False, length=expected_audio_length
        )

        inter_stft = torch.stft(
            inter_audio, n_fft=self.istft_n_fft, hop_length=self.istft_hop_length,
            win_length=self.istft_n_fft, window=self.istft_window, 
            center=True, return_complex=True, pad_mode='reflect'
        )
        
        inter_stft = inter_stft[..., :T_out]

        inter_mag = torch.clamp(torch.abs(inter_stft), min=1e-5)
        implicit_phase_complex = inter_stft / inter_mag

        final_complex_spec = pred_mag * implicit_phase_complex

        audio = torch.istft(
            final_complex_spec, n_fft=self.istft_n_fft, hop_length=self.istft_hop_length,
            win_length=self.istft_n_fft, window=self.istft_window,
            center=True
        )

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
        remove_weight_norm(self.inter_real_conv)
        remove_weight_norm(self.inter_imag_conv)


class FreeVH1(nn.Module):
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

        gen_output = self.generator(mel)
        
        if isinstance(gen_output, tuple):
            audio_fake = gen_output[-1] 
        else:
            audio_fake = gen_output

        output_dict = {
            'audio_fake': audio_fake,
        }

        if audio_real is not None:
            output_dict['audio_real'] = audio_real
            
            mpd_real, mpd_fake, mpd_real_fmaps, mpd_fake_fmaps = self.mpd(audio_real, audio_fake)
            mrd_real, mrd_fake, mrd_real_fmaps, mrd_fake_fmaps = self.mrd(audio_real, audio_fake)

            output_dict.update({
                'mpd_real': mpd_real,
                'mpd_fake': mpd_fake,
                'mrd_real': mrd_real,
                'mrd_fake': mrd_fake,
                'mpd_fmap_real': mpd_real_fmaps,
                'mpd_fmap_fake': mpd_fake_fmaps,
                'mrd_fmap_real': mrd_real_fmaps,
                'mrd_fmap_fake': mrd_fake_fmaps,
            })

        return output_dict

    def inference(self, mel):
        return self.forward(mel=mel)

    def remove_weight_norm(self):
        if hasattr(self.generator, 'remove_weight_norm'):
            self.generator.remove_weight_norm()

    @property
    def discriminator(self):
        return self.discriminators
