import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchaudio.functional import spectrogram, melscale_fbanks


class MelSpectrogramDistance:
    def __init__(self, sample_rate=22050, n_fft=1024, hop_length=256,
                 win_length=1024, n_mels=80, f_min=0.0, f_max=8000.0, name=""):
        self.name = name
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max

        self.mel_fb = melscale_fbanks(
            n_freqs=n_fft // 2 + 1,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            sample_rate=sample_rate
        )

    def __call__(self, audio_real, audio_fake):
        # Обрабатываем батч: вычисляем метрики для каждого примера и усредняем
        batch_size = audio_real.shape[0]
        total_l1 = 0.0
        total_l2 = 0.0

        for i in range(batch_size):
            audio_real_i = audio_real[i:i + 1]  # сохраняем размерность батча
            audio_fake_i = audio_fake[i:i + 1]

            min_length = min(audio_real_i.shape[-1], audio_fake_i.shape[-1])
            audio_real_i = audio_real_i[..., :min_length]
            audio_fake_i = audio_fake_i[..., :min_length]

            spec_real = self._compute_mel_spectrogram(audio_real_i)
            spec_fake = self._compute_mel_spectrogram(audio_fake_i)

            total_l1 += F.l1_loss(spec_fake, spec_real).item()
            total_l2 += F.mse_loss(spec_fake, spec_real).item()

        return {
            f'{self.name}_mel_l1': total_l1 / batch_size,
            f'{self.name}_mel_l2': total_l2 / batch_size
        }

    def _compute_mel_spectrogram(self, audio):
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        stft = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length).to(audio.device),
            return_complex=True,
            center=True
        )

        magnitude = torch.abs(stft)
        mel_fb = self.mel_fb.to(audio.device)
        mel_spec = torch.matmul(mel_fb, magnitude)
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))

        return mel_spec


class STFTDistance:
    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512],
                 win_sizes=[512, 1024, 2048], name=""):
        self.name = name
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def __call__(self, audio_real, audio_fake):
        batch_size = audio_real.shape[0]
        total_sc_loss = 0.0
        total_mag_loss = 0.0
        total_combined = 0.0

        for i in range(batch_size):
            audio_real_i = audio_real[i:i + 1]
            audio_fake_i = audio_fake[i:i + 1]

            min_length = min(audio_real_i.shape[-1], audio_fake_i.shape[-1])
            audio_real_i = audio_real_i[..., :min_length]
            audio_fake_i = audio_fake_i[..., :min_length]

            batch_sc_loss = 0.0
            batch_mag_loss = 0.0

            for n_fft, hop_length, win_length in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
                sc_loss, mag_loss = self._stft_loss(audio_real_i, audio_fake_i, n_fft, hop_length, win_length)
                batch_sc_loss += sc_loss
                batch_mag_loss += mag_loss

            batch_sc_loss = batch_sc_loss / len(self.fft_sizes)
            batch_mag_loss = batch_mag_loss / len(self.fft_sizes)

            total_sc_loss += batch_sc_loss.item()
            total_mag_loss += batch_mag_loss.item()
            total_combined += (batch_sc_loss + batch_mag_loss).item()

        return {
            f'{self.name}_stft_sc_loss': total_sc_loss / batch_size,
            f'{self.name}_stft_mag_loss': total_mag_loss / batch_size,
            f'{self.name}_stft_total_loss': total_combined / batch_size
        }

    def _stft_loss(self, audio_real, audio_fake, n_fft, hop_length, win_length):
        stft_real = self._stft(audio_real, n_fft, hop_length, win_length)
        stft_fake = self._stft(audio_fake, n_fft, hop_length, win_length)

        sc_loss = torch.norm(stft_real - stft_fake, p='fro') / torch.norm(stft_real, p='fro')

        mag_real = torch.log(torch.clamp(torch.abs(stft_real), min=1e-5))
        mag_fake = torch.log(torch.clamp(torch.abs(stft_fake), min=1e-5))
        mag_loss = F.l1_loss(mag_fake, mag_real)

        return sc_loss, mag_loss

    def _stft(self, audio, n_fft, hop_length, win_length):
        if audio.dim() == 3:
            audio = audio.squeeze(1)

        stft = torch.stft(
            audio,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=torch.hann_window(win_length).to(audio.device),
            return_complex=True,
            center=True
        )
        return stft


class PESQScore:
    def __init__(self, sample_rate=22050, mode='wb', name=""):
        self.name = name
        self.sample_rate = sample_rate
        self.mode = mode

    def __call__(self, audio_real, audio_fake):
        try:
            from pesq import pesq
        except ImportError:
            return {f'{self.name}_pesq': 0.0}

        # Вычисляем PESQ только для первого примера в батче (дорогая операция)
        audio_real_cpu = audio_real[0].squeeze().cpu().numpy()
        audio_fake_cpu = audio_fake[0].squeeze().cpu().numpy()

        min_len = min(len(audio_real_cpu), len(audio_fake_cpu))
        audio_real_cpu = audio_real_cpu[:min_len]
        audio_fake_cpu = audio_fake_cpu[:min_len]

        try:
            if self.sample_rate == 16000:
                pesq_score = pesq(16000, audio_real_cpu, audio_fake_cpu, 'wb')
            elif self.sample_rate == 8000:
                pesq_score = pesq(8000, audio_real_cpu, audio_fake_cpu, 'nb')
            else:
                import librosa
                audio_real_16k = librosa.resample(audio_real_cpu, orig_sr=self.sample_rate, target_sr=16000)
                audio_fake_16k = librosa.resample(audio_fake_cpu, orig_sr=self.sample_rate, target_sr=16000)
                pesq_score = pesq(16000, audio_real_16k, audio_fake_16k, 'wb')

            return {f'{self.name}_pesq': pesq_score}
        except Exception as e:
            self.logger.warning(f"PESQ computation failed: {e}")
            return {f'{self.name}_pesq': 0.0}


class AudioMetrics:
    def __init__(self, sample_rate=22050, name=""):
        self.name = name
        self.mel_metric = MelSpectrogramDistance(sample_rate=sample_rate)
        self.stft_metric = STFTDistance()
        self.pesq_metric = PESQScore(sample_rate=sample_rate)

    def __call__(self, audio_real, audio_fake):
        metrics = {}

        mel_metrics = self.mel_metric(audio_real, audio_fake)
        metrics.update(mel_metrics)

        stft_metrics = self.stft_metric(audio_real, audio_fake)
        metrics.update(stft_metrics)

        if np.random.random() < 0.1:
            pesq_metrics = self.pesq_metric(audio_real, audio_fake)
            metrics.update(pesq_metrics)

        return metrics


class GANMetrics:
    def __init__(self):
        pass

    def __call__(self, mpd_real, mpd_fake, msd_real, msd_fake):
        metrics = {}

        mpd_real_acc = torch.mean((torch.cat(mpd_real) > 0.5).float())
        mpd_fake_acc = torch.mean((torch.cat(mpd_fake) < 0.5).float())
        msd_real_acc = torch.mean((torch.cat(msd_real) > 0.5).float())
        msd_fake_acc = torch.mean((torch.cat(msd_fake) < 0.5).float())

        metrics.update({
            'mpd_real_acc': mpd_real_acc.item(),
            'mpd_fake_acc': mpd_fake_acc.item(),
            'msd_real_acc': msd_real_acc.item(),
            'msd_fake_acc': msd_fake_acc.item(),
            'disc_real_acc': (mpd_real_acc + msd_real_acc).item() / 2,
            'disc_fake_acc': (mpd_fake_acc + msd_fake_acc).item() / 2,
        })

        return metrics