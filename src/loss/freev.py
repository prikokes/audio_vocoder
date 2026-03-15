import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import melscale_fbanks


def anti_wrapping_function(x):
    return x - torch.round(x / (2 * torch.pi)) * 2 * torch.pi


def compute_phase_from_audio(audio, n_fft, hop_length, win_length, window):
    if audio.dim() == 3:
        audio = audio.squeeze(1)
    stft_out = torch.stft(
        audio, n_fft=n_fft, hop_length=hop_length,
        win_length=win_length, window=window,
        return_complex=True, center=True,
    )
    magnitude = stft_out.abs()
    phase = torch.angle(stft_out)
    return magnitude, phase


class FreeVLoss(nn.Module):
    def __init__(
        self,
        lambda_A=45.0,
        lambda_P=100.0,
        lambda_S=45.0,
        lambda_W=1.0,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        sample_rate=22050,
        f_min=0.0,
        f_max=8000.0,
    ):
        super().__init__()
        self.lambda_A = lambda_A
        self.lambda_P = lambda_P
        self.lambda_S = lambda_S
        self.lambda_W = lambda_W

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        mel_basis = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        self.register_buffer("mel_basis", mel_basis.T)

    def _find_disc_names(self, model_output):
        names = set()
        for key in model_output:
            if key.endswith("_real") and key != "audio_real":
                name = key[:-5] 
                if (f"{name}_fake" in model_output
                        and f"{name}_fmap_real" in model_output
                        and f"{name}_fmap_fake" in model_output):
                    names.add(name)
        return list(names)

    def forward(self, model_output, optimizer_idx):
        if optimizer_idx == 0:
            return self._generator_loss(model_output)
        else:
            return self._discriminator_loss(model_output)

    def _generator_loss(self, model_output):
        losses = {}
        disc_names = self._find_disc_names(model_output)

        audio_real = model_output["audio_real"]
        audio_fake = model_output["audio_fake"]

        adv_loss = 0.0
        adv_count = 0
        for name in disc_names:
            for d_fake in model_output[f"{name}_fake"]:
                adv_loss += -torch.mean(d_fake)
                adv_count += 1
        adv_loss = adv_loss / max(adv_count, 1)
        losses["loss_adv"] = adv_loss

        fm_loss = 0.0
        fm_count = 0
        for name in disc_names:
            fmap_real_key = f"{name}_fmap_real"
            fmap_fake_key = f"{name}_fmap_fake"
            if fmap_real_key not in model_output:
                continue
            for real_fmaps, fake_fmaps in zip(
                model_output[fmap_real_key], model_output[fmap_fake_key]
            ):
                for real_f, fake_f in zip(real_fmaps, fake_fmaps):
                    real_f = real_f.detach()
                    slices = [slice(None)] * real_f.dim()
                    for d in range(2, real_f.dim()):
                        min_size = min(real_f.shape[d], fake_f.shape[d])
                        slices[d] = slice(None, min_size)
                    real_f = real_f[tuple(slices)]
                    fake_f = fake_f[tuple(slices)]
                    fm_loss += self.l1_loss(fake_f, real_f)
                    fm_count += 1
        fm_loss = fm_loss / max(fm_count, 1)
        losses["loss_fm"] = fm_loss

        mel_loss = self._mel_spectrogram_loss(audio_real, audio_fake)
        losses["loss_mel"] = mel_loss

        amp_loss = self._amplitude_loss(audio_real, audio_fake)
        losses["loss_amplitude"] = amp_loss

        phase_loss = self._phase_loss(audio_real, audio_fake)
        losses["loss_phase"] = phase_loss

        stft_loss = self._stft_consistency_loss(audio_fake)
        losses["loss_stft"] = stft_loss

        losses["loss_g"] = (
            self.lambda_A * amp_loss
            + self.lambda_P * phase_loss
            + self.lambda_S * stft_loss
            + self.lambda_W * (mel_loss + fm_loss + adv_loss)
        )
        return losses


    def _discriminator_loss(self, model_output):
        losses = {}
        disc_names = self._find_disc_names(model_output)

        disc_loss = 0.0
        disc_count = 0
        for name in disc_names:
            for d_real, d_fake in zip(
                model_output[f"{name}_real"], model_output[f"{name}_fake"]
            ):
                disc_loss += torch.mean(F.relu(1.0 - d_real))
                disc_loss += torch.mean(F.relu(1.0 + d_fake.detach()))
                disc_count += 1
        disc_loss = disc_loss / max(disc_count, 1)
        losses["loss_d"] = disc_loss
        return losses

    def _get_window(self, device):
        return torch.hann_window(self.win_length, device=device)

    def _amplitude_loss(self, audio_real, audio_fake):
        min_len = min(audio_real.shape[-1], audio_fake.shape[-1])
        audio_real = audio_real[..., :min_len]
        audio_fake = audio_fake[..., :min_len]

        window = self._get_window(audio_real.device)
        mag_real, _ = compute_phase_from_audio(
            audio_real, self.n_fft, self.hop_length, self.win_length, window
        )
        mag_fake, _ = compute_phase_from_audio(
            audio_fake, self.n_fft, self.hop_length, self.win_length, window
        )
        return self.mse_loss(mag_fake, mag_real)

    def _phase_loss(self, audio_real, audio_fake):
        min_len = min(audio_real.shape[-1], audio_fake.shape[-1])
        audio_real = audio_real[..., :min_len]
        audio_fake = audio_fake[..., :min_len]

        window = self._get_window(audio_real.device)
        _, phase_real = compute_phase_from_audio(
            audio_real, self.n_fft, self.hop_length, self.win_length, window
        )
        _, phase_fake = compute_phase_from_audio(
            audio_fake, self.n_fft, self.hop_length, self.win_length, window
        )

        ip_loss = torch.mean(torch.abs(
            anti_wrapping_function(phase_fake - phase_real)
        ))

        gd_loss = torch.mean(torch.abs(
            anti_wrapping_function(
                torch.diff(phase_fake, dim=-2) - torch.diff(phase_real, dim=-2)
            )
        ))

        ptd_loss = torch.mean(torch.abs(
            anti_wrapping_function(
                torch.diff(phase_fake, dim=-1) - torch.diff(phase_real, dim=-1)
            )
        ))

        return ip_loss + gd_loss + ptd_loss

    def _stft_consistency_loss(self, audio_fake):
        if audio_fake.dim() == 3:
            audio_fake = audio_fake.squeeze(1)

        window = self._get_window(audio_fake.device)

        stft_fake = torch.stft(
            audio_fake, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window,
            return_complex=True, center=True,
        )

        waveform_reconstructed = torch.istft(
            stft_fake, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window,
        )
        stft_reconstructed = torch.stft(
            waveform_reconstructed, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window,
            return_complex=True, center=True,
        )

        T = min(stft_fake.shape[-1], stft_reconstructed.shape[-1])
        stft_fake = stft_fake[..., :T]
        stft_reconstructed = stft_reconstructed[..., :T]

        return (
            self.l1_loss(stft_reconstructed.abs(), stft_fake.abs())
            + self.l1_loss(stft_reconstructed.real, stft_fake.real)
            + self.l1_loss(stft_reconstructed.imag, stft_fake.imag)
        )

    def _mel_spectrogram_loss(self, audio_real, audio_fake):
        min_length = min(audio_real.shape[-1], audio_fake.shape[-1])
        audio_real = audio_real[..., :min_length]
        audio_fake = audio_fake[..., :min_length]
        return self.l1_loss(self._compute_mel(audio_fake), self._compute_mel(audio_real))

    def _compute_mel(self, audio):
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        window = self._get_window(audio.device)
        stft = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=window,
            return_complex=True, center=True,
        )
        magnitude = torch.abs(stft)
        mel = torch.einsum("mf,bft->bmt", self.mel_basis.to(audio.device), magnitude)
        return torch.log(torch.clamp(mel, min=1e-5))