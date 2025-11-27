import torch
import torchaudio
import torchaudio.functional as F
import numpy as np
from torch.nn import functional as torch_functional


class AudioToMelSpectrogram:
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256,
                 win_length=1024, n_mels=80, f_min=0.0, f_max=8000.0):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=1.0,
            normalized=False
        )

    def __call__(self, audio):
        mel = self.mel_transform(audio)

        mel = torch.log(torch.clamp(mel, min=1e-5))

        return mel


class AudioSegment:
    def __init__(self, segment_size=8192):
        self.segment_size = segment_size

    def __call__(self, audio):
        if audio.shape[1] >= self.segment_size:
            max_start = audio.shape[1] - self.segment_size
            start = torch.randint(0, max_start + 1, (1,)).item()
            segment = audio[:, start:start + self.segment_size]
        else:
            pad_size = self.segment_size - audio.shape[1]
            segment = torch_functional.pad(audio, (0, pad_size), mode='reflect')

        return segment


class AudioNormalize:
    def __init__(self, eps=1e-8):
        self.eps = eps

    def __call__(self, audio):
        max_val = torch.max(torch.abs(audio)) + self.eps
        return audio / max_val


class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for transform in self.transforms:
            x = transform(x)
        return x