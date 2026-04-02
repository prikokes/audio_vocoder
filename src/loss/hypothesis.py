import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import melscale_fbanks

def anti_wrapping_function(x):
    return x - torch.round(x / (2 * torch.pi)) * 2 * torch.pi

class HypothesisLoss(nn.Module):
    def __init__(
        self,
        lambda_A=45.0,
        lambda_P=100.0,
        lambda_W=1.0,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        sample_rate=22050,
        f_min=0.0,
        f_max=8000.0,
        **kwargs,
    ):
        super().__init__()
        self.lambda_A = lambda_A
        self.lambda_P = lambda_P
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
        self.register_buffer("window", torch.hann_window(win_length))

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

    def _compute_stft_components(self, audio):
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        stft_out = torch.stft(
            audio, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.win_length, window=self.window,
            return_complex=True, center=True,
        )
        mag = torch.abs(stft_out)
        phase = torch.angle(stft_out)
        
        mel = torch.einsum("mf,bft->bmt", self.mel_basis, mag)
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        
        return mag, phase, log_mel

    def _generator_loss(self, model_output):
        losses = {}
        disc_names = self._find_disc_names(model_output)

        audio_real = model_output["audio_real"]
        audio_fake = model_output["audio_fake"]
        
        min_len = min(audio_real.shape[-1], audio_fake.shape[-1])
        audio_real = audio_real[..., :min_len]
        audio_fake = audio_fake[..., :min_len]

        mag_real, phase_real, mel_real = self._compute_stft_components(audio_real)
        mag_fake, phase_fake, mel_fake = self._compute_stft_components(audio_fake)

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
            for real_fmaps, fake_fmaps in zip(model_output[fmap_real_key], model_output[fmap_fake_key]):
                for real_f, fake_f in zip(real_fmaps, fake_fmaps):
                    real_f = real_f.detach()
                    if real_f.shape != fake_f.shape:
                        min_t = min(real_f.shape[-1], fake_f.shape[-1])
                        real_f, fake_f = real_f[..., :min_t], fake_f[..., :min_t]
                    fm_loss += self.l1_loss(fake_f, real_f)
                    fm_count += 1
        fm_loss = fm_loss / max(fm_count, 1)
        losses["loss_fm"] = fm_loss

        mel_loss = self.l1_loss(mel_fake, mel_real)
        losses["loss_mel"] = mel_loss

        amp_loss = self.mse_loss(mag_fake, mag_real)
        losses["loss_amplitude"] = amp_loss

        ip_loss = torch.mean(torch.abs(anti_wrapping_function(phase_fake - phase_real)))
        gd_loss = torch.mean(torch.abs(anti_wrapping_function(torch.diff(phase_fake, dim=-2) - torch.diff(phase_real, dim=-2))))
        ptd_loss = torch.mean(torch.abs(anti_wrapping_function(torch.diff(phase_fake, dim=-1) - torch.diff(phase_real, dim=-1))))
        phase_loss = ip_loss + gd_loss + ptd_loss
        losses["loss_phase"] = phase_loss

        losses["loss_g"] = (
            self.lambda_A * amp_loss
            + self.lambda_P * phase_loss
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