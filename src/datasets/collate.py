import torch
import logging

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: list[dict]):
    result_batch = {}

    with open("log.txt", 'w') as file:
        file.write(f"{dataset_items[0].keys()}")

    if len(dataset_items) > 0:
        logger.info(f"Available keys in dataset items: {dataset_items[0].keys()}")

    audios = [item["audio"] for item in dataset_items]

    max_len = max(audio.shape[1] for audio in audios)

    padded_audios = []
    for audio in audios:
        pad_size = max_len - audio.shape[1]
        if pad_size > 0:
            padded_audio = torch.nn.functional.pad(audio, (0, pad_size))
        else:
            padded_audio = audio
        padded_audios.append(padded_audio)

    result_batch["audio"] = torch.stack(padded_audios)
    result_batch["text"] = [item["text"] for item in dataset_items]
    result_batch["audio_path"] = [item["audio_path"] for item in dataset_items]
    result_batch["sample_rate"] = torch.tensor([item["sample_rate"] for item in dataset_items])
    result_batch["mel"] = torch.concatenate([item["mel"] for item in dataset_items], 0)

    return result_batch