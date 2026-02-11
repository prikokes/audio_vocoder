import torch
import torch.nn as nn
from torchaudio.functional import melscale_fbanks


class HiFiGANLoss(nn.Module):
    def __init__(self, lambda_adv=1.0, lambda_fm=2.0, lambda_mel=45.0,
                 n_fft=1024, hop_length=256, win_length=1024,
                 n_mels=80, sample_rate=22050, f_min=0.0, f_max=8000.0):
        super(HiFiGANLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        mel_basis = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=sample_rate
        )
        print(f"mel_basis shape from melscale_fbanks: {mel_basis.shape}")
        self.register_buffer('mel_basis', mel_basis.T)

    def forward(self, audio_real, audio_fake, mpd_real, mpd_fake, msd_real, msd_fake,
                mpd_real_fmaps, mpd_fake_fmaps, msd_real_fmaps, msd_fake_fmaps,
                model, optimizer_idx):

        if optimizer_idx == 0:
            return self._generator_loss(
                audio_real, audio_fake, mpd_fake, msd_fake,
                mpd_real_fmaps, mpd_fake_fmaps, msd_real_fmaps, msd_fake_fmaps
            )
        else:
            return self._discriminator_loss(mpd_real, mpd_fake, msd_real, msd_fake)

    def _generator_loss(self, audio_real, audio_fake, mpd_fake, msd_fake,
                        mpd_real_fmaps, mpd_fake_fmaps, msd_real_fmaps, msd_fake_fmaps):
        losses = {}

        adv_loss = 0.0
        for d_fake in mpd_fake:
            adv_loss += self.mse_loss(d_fake, torch.ones_like(d_fake))
        for d_fake in msd_fake:
            adv_loss += self.mse_loss(d_fake, torch.ones_like(d_fake))
        adv_loss = adv_loss / (len(mpd_fake) + len(msd_fake))
        losses['loss_adv'] = adv_loss

        fm_loss = 0.0
        fm_count = 0

        for real_fmaps, fake_fmaps in [(mpd_real_fmaps, mpd_fake_fmaps),
                                       (msd_real_fmaps, msd_fake_fmaps)]:
            for i in range(len(real_fmaps)):
                for j in range(len(real_fmaps[i])):
                    real_fmap = real_fmaps[i][j].detach()
                    fake_fmap = fake_fmaps[i][j]

                    min_len = min(real_fmap.shape[-1], fake_fmap.shape[-1])
                    real_fmap = real_fmap[..., :min_len]
                    fake_fmap = fake_fmap[..., :min_len]

                    fm_loss += self.l1_loss(fake_fmap, real_fmap)
                    fm_count += 1

        fm_loss = fm_loss / fm_count if fm_count > 0 else 0.0
        losses['loss_fm'] = fm_loss

        mel_loss = self._mel_spectrogram_loss(audio_real, audio_fake)
        losses['loss_mel'] = mel_loss

        total_loss = (self.lambda_adv * adv_loss +
                      self.lambda_fm * fm_loss +
                      self.lambda_mel * mel_loss)
        losses['loss_g'] = total_loss

        return losses

    def _discriminator_loss(self, mpd_real, mpd_fake, msd_real, msd_fake):
        losses = {}
        disc_loss = 0.0

        for d_real, d_fake in zip(mpd_real, mpd_fake):
            disc_loss += self.mse_loss(d_real, torch.ones_like(d_real))
            disc_loss += self.mse_loss(d_fake.detach(), torch.zeros_like(d_fake))

        for d_real, d_fake in zip(msd_real, msd_fake):
            disc_loss += self.mse_loss(d_real, torch.ones_like(d_real))
            disc_loss += self.mse_loss(d_fake.detach(), torch.zeros_like(d_fake))

        disc_loss = disc_loss / (2 * (len(mpd_real) + len(msd_real)))
        losses['loss_d'] = disc_loss

        return losses

    def _mel_spectrogram_loss(self, audio_real, audio_fake):
        min_length = min(audio_real.shape[-1], audio_fake.shape[-1])
        audio_real = audio_real[..., :min_length]
        audio_fake = audio_fake[..., :min_length]

        mel_real = self._compute_mel(audio_real)
        mel_fake = self._compute_mel(audio_fake)

        return self.l1_loss(mel_fake, mel_real)

    def _compute_mel(self, audio):
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        window = torch.hann_window(self.win_length, device=audio.device)
        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
            center=True
        )

        magnitude = torch.abs(stft)  # [B, 513, T']
        mel_basis = self.mel_basis.to(audio.device)  # [80, 513]

        mel = torch.einsum('mf,bft->bmt', mel_basis, magnitude)

        mel = torch.log(torch.clamp(mel, min=1e-5))

        return mel
