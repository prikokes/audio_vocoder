from abc import abstractmethod

import torch
import numpy as np
from numpy import inf
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from src.datasets.data_utils import inf_loop
from src.metrics.tracker import MetricTracker
from src.utils.io_utils import ROOT_PATH
from src.trainer.base_trainer import BaseTrainer


class HiFiGANTrainer(BaseTrainer):
    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer_g,
            optimizer_d,
            lr_scheduler_g,
            lr_scheduler_d,
            config,
            device,
            dataloaders,
            logger,
            writer,
            epoch_len=None,
            skip_oom=True,
            batch_transforms=None,
    ):

        super().__init__(
            model=model,
            criterion=criterion,
            metrics=metrics,
            optimizer=optimizer_g,
            lr_scheduler=lr_scheduler_g,
            config=config,
            device=device,
            dataloaders=dataloaders,
            logger=logger,
            writer=writer,
            epoch_len=epoch_len,
            skip_oom=skip_oom,
            batch_transforms=batch_transforms,
        )

        self.optimizer_g = optimizer_g
        self.optimizer_d = optimizer_d
        self.lr_scheduler_g = lr_scheduler_g
        self.lr_scheduler_d = lr_scheduler_d

        self.d_steps = config.trainer.get("d_steps", 1)
        self.g_steps = config.trainer.get("g_steps", 1)
        self.segment_size = config.trainer.get("segment_size", 8192)

    def process_batch(self, batch, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]

        mel = batch["mel"]
        audio_real = batch["audio"]

        if self.is_train:
            d_losses = self._train_discriminator(mel, audio_real)
            batch.update(d_losses)

            g_losses = self._train_generator(mel, audio_real)
            batch.update(g_losses)

            with torch.no_grad():
                audio_fake = self.model.generator(mel)
            batch["audio_fake"] = audio_fake
        else:
            with torch.no_grad():
                audio_fake = self.model.generator(mel)

                min_len = min(audio_real.shape[-1], audio_fake.shape[-1])
                audio_real_matched = audio_real[..., :min_len]
                audio_fake_matched = audio_fake[..., :min_len]

                mpd_real, mpd_fake, mpd_real_fmaps, mpd_fake_fmaps = self.model.mpd(
                    audio_real_matched, audio_fake_matched
                )
                msd_real, msd_fake, msd_real_fmaps, msd_fake_fmaps = self.model.msd(
                    audio_real_matched, audio_fake_matched
                )

                g_losses = self.criterion(
                    audio_real=audio_real_matched,
                    audio_fake=audio_fake_matched,
                    mpd_real=mpd_real,
                    mpd_fake=mpd_fake,
                    msd_real=msd_real,
                    msd_fake=msd_fake,
                    mpd_real_fmaps=mpd_real_fmaps,
                    mpd_fake_fmaps=mpd_fake_fmaps,
                    msd_real_fmaps=msd_real_fmaps,
                    msd_fake_fmaps=msd_fake_fmaps,
                    model=self.model,
                    optimizer_idx=0
                )
                batch.update(g_losses)

                d_losses = self.criterion(
                    audio_real=audio_real_matched,
                    audio_fake=audio_fake_matched,
                    mpd_real=mpd_real,
                    mpd_fake=mpd_fake,
                    msd_real=msd_real,
                    msd_fake=msd_fake,
                    mpd_real_fmaps=mpd_real_fmaps,
                    mpd_fake_fmaps=mpd_fake_fmaps,
                    msd_real_fmaps=msd_real_fmaps,
                    msd_fake_fmaps=msd_fake_fmaps,
                    model=self.model,
                    optimizer_idx=1
                )
                batch.update(d_losses)

            batch["audio_fake"] = audio_fake

        batch["audio_real"] = audio_real

        for loss_name in self.config.writer.loss_names:
            if loss_name in batch and batch[loss_name] is not None:
                loss_value = batch[loss_name]
                if hasattr(loss_value, 'item'):
                    metrics.update(loss_name, loss_value.item())

        for met in metric_funcs:
            try:
                metric_value = met(batch["audio_real"], batch["audio_fake"])
                metrics.update(met.name, metric_value)
            except Exception as e:
                self.logger.warning(f"Could not compute metric {met.name}: {e}")

        if 'loss' not in batch:
            if 'loss_g' in batch:
                batch['loss'] = batch['loss_g']
            elif 'loss_d' in batch:
                batch['loss'] = batch['loss_d']

        return batch
    
    def _log_audio_samples(self, batch, mode):
        import traceback
        import numpy as np

        audio_real = batch.get("audio")
        audio_fake = batch.get("audio_fake")

        if audio_real is None or audio_fake is None:
            return

        try:
            idx = 0
            real = audio_real[idx].detach().cpu().numpy().squeeze()
            fake = audio_fake[idx].detach().cpu().numpy().squeeze()
            real = np.clip(real, -1.0, 1.0).astype(np.float32)
            fake = np.clip(fake, -1.0, 1.0).astype(np.float32)

            self.writer.add_audio(f"{mode}/audio_real", real, sample_rate=22050)
            self.writer.add_audio(f"{mode}/audio_fake", fake, sample_rate=22050)
            self.logger.info("Audio logged successfully")
        except Exception as e:
            self.logger.warning(f"Failed to log audio: {e}")
            self.logger.warning(traceback.format_exc())

        try:
            mel = batch.get("mel")
            if mel is not None:
                mel_np = mel[0].detach().cpu().numpy()
                self.writer.add_image(f"{mode}/mel_spectrogram", mel_np)
                self.logger.info("Mel image logged successfully")
        except Exception as e:
            self.logger.warning(f"Failed to log mel: {e}")
            self.logger.warning(traceback.format_exc())

    def _train_epoch(self, epoch):
        self.is_train = True
        self.model.train()
        self.train_metrics.reset()
        self.writer.set_step((epoch - 1) * self.epoch_len)
        self.writer.add_scalar("epoch", epoch)

        last_batch = None

        for batch_idx, batch in enumerate(
            tqdm(self.train_dataloader, desc="train", total=self.epoch_len)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    metrics=self.train_metrics,
                )
                last_batch = batch
            except torch.cuda.OutOfMemoryError as e:
                if self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            self.train_metrics.update("grad_norm", self._get_grad_norm())

            if batch_idx + 1 >= self.epoch_len:
                break

        self.writer.set_step(epoch * self.epoch_len)
        self.writer.add_scalar("learning_rate_g", self.lr_scheduler_g.get_last_lr()[0])
        self.writer.add_scalar("learning_rate_d", self.lr_scheduler_d.get_last_lr()[0])
        self._log_scalars(self.train_metrics)

        if last_batch is not None:
            self._log_audio_samples(last_batch, "train")

        last_train_metrics = self.train_metrics.result()

        self.logger.info(
            "Train Epoch: {} | loss_g: {:.4f} | loss_d: {:.4f}".format(
                epoch,
                last_train_metrics.get("loss_g", 0.0),
                last_train_metrics.get("loss_d", 0.0),
            )
        )

        if self.lr_scheduler_g is not None:
            self.lr_scheduler_g.step()
        if self.lr_scheduler_d is not None:
            self.lr_scheduler_d.step()

        logs = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_logs = self._evaluation_epoch(epoch, part, dataloader)
            logs.update(**{f"{part}_{name}": value for name, value in val_logs.items()})

        return logs

    def _train_generator(self, mel, audio_real):
        self.optimizer_g.zero_grad()

        audio_fake = self.model.generator(mel)

        min_len = min(audio_real.shape[-1], audio_fake.shape[-1])
        audio_real_matched = audio_real[..., :min_len]
        audio_fake_matched = audio_fake[..., :min_len]

        mpd_real, mpd_fake, mpd_real_fmaps, mpd_fake_fmaps = self.model.mpd(
            audio_real_matched, audio_fake_matched
        )
        msd_real, msd_fake, msd_real_fmaps, msd_fake_fmaps = self.model.msd(
            audio_real_matched, audio_fake_matched
        )

        losses = self.criterion(
            audio_real=audio_real_matched,
            audio_fake=audio_fake_matched,
            mpd_real=mpd_real,
            mpd_fake=mpd_fake,
            msd_real=msd_real,
            msd_fake=msd_fake,
            mpd_real_fmaps=mpd_real_fmaps,
            mpd_fake_fmaps=mpd_fake_fmaps,
            msd_real_fmaps=msd_real_fmaps,
            msd_fake_fmaps=msd_fake_fmaps,
            model=self.model,
            optimizer_idx=0
        )

        if self.is_train:
            losses["loss_g"].backward()
            self._clip_grad_norm_g()
            self.optimizer_g.step()

        losses["audio_fake"] = audio_fake.detach()

        return losses

    def _train_discriminator(self, mel, audio_real):
        self.optimizer_d.zero_grad()

        with torch.no_grad():
            audio_fake = self.model.generator(mel)

        min_len = min(audio_real.shape[-1], audio_fake.shape[-1])
        audio_real_matched = audio_real[..., :min_len]
        audio_fake_matched = audio_fake[..., :min_len]

        mpd_real, mpd_fake, mpd_real_fmaps, mpd_fake_fmaps = self.model.mpd(
            audio_real_matched, audio_fake_matched
        )
        msd_real, msd_fake, msd_real_fmaps, msd_fake_fmaps = self.model.msd(
            audio_real_matched, audio_fake_matched
        )

        losses = self.criterion(
            audio_real=audio_real_matched,
            audio_fake=audio_fake_matched,
            mpd_real=mpd_real,
            mpd_fake=mpd_fake,
            msd_real=msd_real,
            msd_fake=msd_fake,
            mpd_real_fmaps=mpd_real_fmaps,
            msd_real_fmaps=msd_real_fmaps,
            msd_fake_fmaps=msd_fake_fmaps,
            mpd_fake_fmaps=mpd_fake_fmaps,
            model=self.model,
            optimizer_idx=1
        )

        if self.is_train:
            losses["loss_d"].backward()
            self._clip_grad_norm_d()
            self.optimizer_d.step()

        return losses

    def _clip_grad_norm_g(self):
        if self.config["trainer"].get("max_grad_norm_g", None) is not None:
            clip_grad_norm_(
                self.model.generator.parameters(),
                self.config["trainer"]["max_grad_norm_g"]
            )

    def _clip_grad_norm_d(self):
        if self.config["trainer"].get("max_grad_norm_d", None) is not None:
            d_params = []
            d_params.extend(self.model.mpd.parameters())
            d_params.extend(self.model.msd.parameters())

            clip_grad_norm_(
                d_params,
                self.config["trainer"]["max_grad_norm_d"]
            )

    @torch.no_grad()
    def _get_grad_norm(self, norm_type=2):
        g_parameters = [p for p in self.model.generator.parameters() if p.grad is not None]
        g_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in g_parameters]),
            norm_type,
        ).item() if g_parameters else 0.0

        d_parameters = []
        d_parameters.extend([p for p in self.model.mpd.parameters() if p.grad is not None])
        d_parameters.extend([p for p in self.model.msd.parameters() if p.grad is not None])
        d_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in d_parameters]),
            norm_type,
        ).item() if d_parameters else 0.0

        return g_norm + d_norm

    def _log_batch(self, batch_idx, batch, mode="train"):
        pass

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer_g": self.optimizer_g.state_dict(),
            "optimizer_d": self.optimizer_d.state_dict(),
            "lr_scheduler_g": self.lr_scheduler_g.state_dict() if self.lr_scheduler_g else None,
            "lr_scheduler_d": self.lr_scheduler_d.state_dict() if self.lr_scheduler_d else None,
            "monitor_best": self.mnt_best,
            "config": self.config,
        }

        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")

        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in config file is different from checkpoint."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        if (checkpoint["config"]["optimizer_g"] != self.config["optimizer_g"] or
                checkpoint["config"]["optimizer_d"] != self.config["optimizer_d"]):
            self.logger.warning(
                "Warning: Optimizer configuration different from checkpoint. Not resuming optimizers."
            )
        else:
            self.optimizer_g.load_state_dict(checkpoint["optimizer_g"])
            self.optimizer_d.load_state_dict(checkpoint["optimizer_d"])

        if (self.lr_scheduler_g and checkpoint["lr_scheduler_g"] and
                checkpoint["config"]["lr_scheduler_g"] == self.config["lr_scheduler_g"]):
            self.lr_scheduler_g.load_state_dict(checkpoint["lr_scheduler_g"])

        if (self.lr_scheduler_d and checkpoint["lr_scheduler_d"] and
                checkpoint["config"]["lr_scheduler_d"] == self.config["lr_scheduler_d"]):
            self.lr_scheduler_d.load_state_dict(checkpoint["lr_scheduler_d"])

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")
