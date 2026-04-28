# Neural Audio Vocoders: Quality and Performance Evaluation

This repository contains the codebase for training, evaluating, and comparing various neural vocoder architectures. The main goal of this research is to analyze the trade-offs between audio synthesis quality, model size, and inference speed (Real-Time Factor - RTF). 

Supported models include standard baselines and modern architectures such as **HiFi-GAN (V1 & V2)**, **FreeV**, and custom **ISTFT-based models** (ISTFTWav, ISTFTWavSnake).

## Project Architecture & Configuration

This project relies heavily on **[Hydra](https://hydra.cc/)** for flexible configuration management. 
Instead of hardcoding parameters, all settings for models, datasets, dataloaders, and training loops are modularized in the `src/configs/` directory.

The entry point for the configuration is the **`baseline.yaml`** file. You can easily switch between models or hyperparameters by overriding values in the command line or creating new composition configs (e.g., `istftwav.yaml`).

In our study, we compared heavy baseline models against lightweight and fast architectures. The evaluation was conducted across two main dimensions: Synthesis Quality (STOI, PESQ, LSD, Mel-Distance) and Computational Complexity (Parameters, MACs, RTF on CPU/GPU).

## Research results: 

### 1. Audio Synthesis Quality

| Model | STOI ↑ | PESQ ↑ | LSD ↓ | Mel-Distance ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **HiFi-GAN V1** | 0.96 | 3.311 | 1.231 | **0.1359** |
| **HiFi-GAN V2** | 0.91 | 2.549 | 1.663 | 0.1996 |
| **FreeV** | **0.96** | **3.615** | **0.700** | 0.1380 |
| **Model 1** | 0.93 | 3.056 | 0.710 | 0.1829 |
| **Model 2** | 0.96 | 3.153 | 0.708 | 0.1845 |
| **Model 3** | **0.96** | 3.391 | **0.700** | 0.1863 |


### 2. Performance & Computational Complexity

| Model | Params (M) ↓ | MACs (M) ↓ | RTF (CPU) ↓ | RTF (GPU) ↓ |
| :--- | :---: | :---: | :---: | :---: |
| **HiFi-GAN V1** | 13.90 | 30818.74 | 0.07620 | 0.10522 |
| **HiFi-GAN V2** | **0.93** | 1909.58 | 0.01963 | 0.04314 |
| **FreeV** | 18.20 | 1563.27 | **0.00400** | 0.01292 |
| **Model 1** | 2.41 | 977.38 | 0.00701 | **0.01194** |
| **Model 2** | 2.41 | 977.38 | 0.00701 | **0.01194** |
| **Model 3** | 2.39 | **860.28** | 0.01044 | 0.01754 |

## Training
To train the models from scratch, run the train.py script and specify the model architecture.

```
python train.py model=hifigan

python train.py model=freev

python train.py model=istftwav
```

## Running inference

```
python inference.py model=freev
```


## License
This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.