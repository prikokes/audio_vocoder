import torch
import torch.nn as nn
import torch.nn.functional as F


class HiFiGANLoss(nn.Module):
    def __init__(self, lambda_adv=1.0, lambda_fm=2.0, lambda_mel=45.0):
        super(HiFiGANLoss, self).__init__()
        self.lambda_adv = lambda_adv
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

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

        # Feature matching loss with shape matching
        fm_loss = 0.0
        fm_count = 0

        for i in range(len(mpd_real_fmaps)):
            for j in range(len(mpd_real_fmaps[i])):
                real_fmap = mpd_real_fmaps[i][j].detach()
                fake_fmap = mpd_fake_fmaps[i][j]

                # Match the dimensions
                fake_fmap = self._match_feature_dims(fake_fmap, real_fmap)

                fm_loss += self.l1_loss(fake_fmap, real_fmap)
                fm_count += 1

        for i in range(len(msd_real_fmaps)):
            for j in range(len(msd_real_fmaps[i])):
                real_fmap = msd_real_fmaps[i][j].detach()
                fake_fmap = msd_fake_fmaps[i][j]

                # Match the dimensions
                fake_fmap = self._match_feature_dims(fake_fmap, real_fmap)

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
            disc_loss += self.mse_loss(d_fake, torch.zeros_like(d_fake))

        for d_real, d_fake in zip(msd_real, msd_fake):
            disc_loss += self.mse_loss(d_real, torch.ones_like(d_real))
            disc_loss += self.mse_loss(d_fake, torch.zeros_like(d_fake))

        disc_loss = disc_loss / (len(mpd_real) + len(msd_real) + len(mpd_fake) + len(msd_fake))
        losses['loss_d'] = disc_loss

        return losses

    def _match_feature_dims(self, fake_fmap, real_fmap):
        """
        Match the dimensions of fake feature map to real feature map.
        Handles different temporal/spatial dimensions.
        """
        if fake_fmap.shape == real_fmap.shape:
            return fake_fmap

        # Match the last dimension (temporal/spatial)
        if fake_fmap.shape[-1] != real_fmap.shape[-1]:
            min_len = min(fake_fmap.shape[-1], real_fmap.shape[-1])
            fake_fmap = fake_fmap[..., :min_len]

        # Match 2D case if needed (for 4D tensors)
        if len(fake_fmap.shape) == 4 and fake_fmap.shape[-2] != real_fmap.shape[-2]:
            min_h = min(fake_fmap.shape[-2], real_fmap.shape[-2])
            fake_fmap = fake_fmap[..., :min_h, :]

        return fake_fmap

    def _mel_spectrogram_loss(self, audio_real, audio_fake):

        min_length = min(audio_real.shape[-1], audio_fake.shape[-1])
        audio_real = audio_real[..., :min_length]
        audio_fake = audio_fake[..., :min_length]

        mel_loss = self.l1_loss(audio_fake, audio_real)

        return mel_loss