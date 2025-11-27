import logging
import random
from typing import List
import os

import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd
import sys
import soundfile


logger = logging.getLogger(__name__)


class LJSpeechDataset(Dataset):
    def __init__(
            self,
            root_dir: str,
            metadata_file: str = "metadata.csv",
            limit: int = None,
            offset: int = 0,
            shuffle_index: bool = False,
            instance_transforms: dict = None,
            split_symbol: str = "|",
            name: str = "lj_speech"
    ):
        self.root_dir = root_dir
        self.wavs_dir = os.path.join(root_dir, "wavs")
        self.name = name
        metadata_path = os.path.join(root_dir, metadata_file)

        # Load and parse metadata
        index = self._load_metadata(metadata_path, split_symbol)

        self._assert_index_is_valid(index)
        index = self._apply_offset_and_limit(index, offset, limit)
        index = self._shuffle_and_limit_index(index, limit, shuffle_index)
        self._index: List[dict] = index

        self.instance_transforms = instance_transforms

    def __getitem__(self, idx):
        data_dict = self._index[idx]
        audio_path = data_dict["audio_path"]
        text = data_dict["text"]

        waveform, sample_rate = self.load_audio(audio_path)

        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        data = {
            "audio": waveform,
            "text": text,
            "audio_path": audio_path,
            "sample_rate": sample_rate,
            "mel": []
        }

        for transform_name, transform in self.instance_transforms.items():
            if callable(transform):
                data[transform_name] = transform(data["audio"])

        logger.info("Some text.")
        logger.info(data.keys())
        sys.stdout.flush()

             # if "audio" in data:
                # data["mel"] = self.mel_transform(data["audio"])
                # data["mel_input"] = data["mel"]

        # except Exception as e:
        #     logger.error(f"Error loading sample {idx}: {e}")
        #     dummy_audio = torch.zeros(1, 16000)
        #
        #     data = {
        #         "audio": dummy_audio,
        #         "text": "",
        #         "audio_path": "",
        #         "sample_rate": 22050,
        #         "mel": torch.zeros(80, 64),
        #         "mel_input": torch.zeros(80, 64)
        #     }
        #
        # logger.info("Some text.")
        # logger.info(data.keys())

        return data

    def __len__(self):
        return len(self._index)

    def load_audio(self, audio_path):
        # waveform, sample_rate = torchaudio.load(audio_path)
        waveform, sample_rate = soundfile.read(audio_path)
        waveform = torch.from_numpy(waveform)
        waveform = waveform.to(torch.float32)

        return waveform, sample_rate

    def _load_metadata(self, metadata_path, split_symbol):
        index = []
        try:
            metadata_df = pd.read_csv(
                metadata_path,
                sep=split_symbol,
                header=None,
                names=["id", "text", "normalized_text"],
                quotechar='"',
                skipinitialspace=True
            )

            for _, row in metadata_df.iterrows():
                # Проверяем, что текст не NaN и конвертируем в строку
                normalized_text = row["normalized_text"]
                raw_text = row["text"]

                if pd.isna(normalized_text) or pd.isna(raw_text):
                    logger.warning(f"Skipping row with missing text: {row['id']}")
                    continue

                # Конвертируем в строку на всякий случай
                normalized_text = str(normalized_text).strip()
                raw_text = str(raw_text).strip()

                audio_path = os.path.join(self.wavs_dir, f"{row['id']}.wav")
                if os.path.exists(audio_path):
                    index.append({
                        "audio_path": audio_path,
                        "text": normalized_text,
                        "raw_text": raw_text,
                        "id": row["id"]
                    })
                else:
                    logger.warning(f"Audio file not found: {audio_path}")
        except Exception as e:
            logger.error(f"Error loading metadata from {metadata_path}: {e}")
            raise e

        return index

    def _apply_offset_and_limit(self, index, offset, limit):
        if offset > 0:
            index = index[offset:]
        if limit is not None:
            index = index[:limit]
        return index

    @staticmethod
    def _assert_index_is_valid(index):
        for entry in index:
            assert "audio_path" in entry, "Missing 'audio_path' in dataset entry"
            assert "text" in entry, "Missing 'text' in dataset entry"
            assert os.path.exists(entry["audio_path"]), \
                f"Audio file {entry['audio_path']} does not exist"

    @staticmethod
    def _shuffle_and_limit_index(index, limit, shuffle_index):
        if shuffle_index:
            random.seed(42)
            random.shuffle(index)

        if limit is not None:
            index = index[:limit]
        return index