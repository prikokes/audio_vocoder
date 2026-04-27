import warnings
import os
import time
import hydra
import torch
import numpy as np
import librosa
import pyworld as pw
import pysptk

from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from scipy.io import wavfile

from src.datasets.data_utils import get_dataloaders
from src.utils.init_utils import set_random_seed
from src.utils.io_utils import ROOT_PATH

from pesq import pesq
from pystoi import stoi
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cityblock
from sklearn.metrics import f1_score

try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("ВНИМАНИЕ: thop не установлен. Подсчет MACs будет пропущен. Установите: pip install thop")

warnings.filterwarnings("ignore")
class TTSMetrics:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr 

    def _normalize_audio(self, wav):
        max_val = np.max(np.abs(wav))
        if max_val > 0:
            return wav / max_val
        return wav

    def calculate_pesq(self, ref_wav, deg_wav, sr):
        if len(ref_wav) < sr * 0.5:
            return float('nan')
        if sr != 16000:
            ref_wav = librosa.resample(y=ref_wav, orig_sr=sr, target_sr=16000)
            deg_wav = librosa.resample(y=deg_wav, orig_sr=sr, target_sr=16000)
            sr = 16000
        min_len = min(len(ref_wav), len(deg_wav))
        try:
            return pesq(sr, ref_wav[:min_len], deg_wav[:min_len], 'wb')
        except:
            return float('nan')

    def calculate_stoi(self, ref_wav, deg_wav, sr):
        min_len = min(len(ref_wav), len(deg_wav))
        return stoi(ref_wav[:min_len], deg_wav[:min_len], sr, extended=False)

    def calculate_mcd(self, ref_wav, deg_wav, sr):
        min_len = min(len(ref_wav), len(deg_wav))
        ref_wav = self._normalize_audio(ref_wav[:min_len]).astype(np.float64)
        deg_wav = self._normalize_audio(deg_wav[:min_len]).astype(np.float64)

        f0_ref, t_ref = pw.dio(ref_wav, sr)
        f0_ref = pw.stonemask(ref_wav, f0_ref, t_ref, sr)
        sp_ref = pw.cheaptrick(ref_wav, f0_ref, t_ref, sr)

        f0_deg, t_deg = pw.dio(deg_wav, sr)
        f0_deg = pw.stonemask(deg_wav, f0_deg, t_deg, sr)
        sp_deg = pw.cheaptrick(deg_wav, f0_deg, t_deg, sr)

        alpha = pysptk.util.mcepalpha(sr)
        
        mc_ref = pysptk.sp2mc(sp_ref, order=34, alpha=alpha)
        mc_deg = pysptk.sp2mc(sp_deg, order=34, alpha=alpha)

        mc_ref = mc_ref[:, 1:]
        mc_deg = mc_deg[:, 1:]

        diff = mc_ref - mc_deg
        mcd_constant = (10.0 * np.sqrt(2.0)) / np.log(10.0)
        
        distances = np.sqrt(np.sum(diff ** 2, axis=1))
        
        return mcd_constant * np.mean(distances)

    def calculate_lsd(self, ref_wav, deg_wav, sr):
        min_len = min(len(ref_wav), len(deg_wav))
        ref_wav = self._normalize_audio(ref_wav[:min_len])
        deg_wav = self._normalize_audio(deg_wav[:min_len])

        S_ref = np.abs(librosa.stft(ref_wav))
        S_deg = np.abs(librosa.stft(deg_wav))

        log_S_ref = 10 * np.log10(np.maximum(S_ref ** 2, 1e-10))
        log_S_deg = 10 * np.log10(np.maximum(S_deg ** 2, 1e-10))

        lsd_frames = np.sqrt(np.mean((log_S_ref - log_S_deg) ** 2, axis=0))
        return np.mean(lsd_frames)

    def calculate_mel_distance(self, ref_wav, deg_wav, sr):
        min_len = min(len(ref_wav), len(deg_wav))
        ref_wav = self._normalize_audio(ref_wav[:min_len])
        deg_wav = self._normalize_audio(deg_wav[:min_len])

        mel_ref = librosa.feature.melspectrogram(y=ref_wav, sr=sr, n_mels=80)
        mel_deg = librosa.feature.melspectrogram(y=deg_wav, sr=sr, n_mels=80)

        log_mel_ref = np.log(np.maximum(mel_ref, 1e-5))
        log_mel_deg = np.log(np.maximum(mel_deg, 1e-5))

        return np.mean(np.abs(log_mel_ref - log_mel_deg))

    def _extract_f0_and_vuv(self, wav, sr):
        f0, voiced_flag, _ = librosa.pyin(
            wav, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'), 
            sr=sr,
            fill_na=0.0
        )
        return f0, voiced_flag

    def calculate_f0_metrics(self, ref_wav, deg_wav, sr):
        min_len = min(len(ref_wav), len(deg_wav))
        ref_wav = self._normalize_audio(ref_wav[:min_len])
        deg_wav = self._normalize_audio(deg_wav[:min_len])

        f0_ref, vuv_ref = self._extract_f0_and_vuv(ref_wav, sr)
        f0_deg, vuv_deg = self._extract_f0_and_vuv(deg_wav, sr)

        min_frames = min(len(f0_ref), len(f0_deg))
        f0_ref, f0_deg = f0_ref[:min_frames], f0_deg[:min_frames]
        vuv_ref, vuv_deg = vuv_ref[:min_frames], vuv_deg[:min_frames]

        mutual_voiced = (vuv_ref == True) & (vuv_deg == True)
        
        if np.any(mutual_voiced):
            f0_r = f0_ref[mutual_voiced]
            f0_d = f0_deg[mutual_voiced]
            
            valid = (f0_r > 0) & (f0_d > 0)
            
            if np.any(valid):
                cents_diff = 1200.0 * np.log2(f0_d[valid] / f0_r[valid])
                f0_rmse = np.sqrt(np.mean(cents_diff**2))
            else:
                f0_rmse = float('nan')
        else:
            f0_rmse = float('nan')

        vuv_f1 = f1_score(vuv_ref, vuv_deg, zero_division=1.0)
        return f0_rmse, vuv_f1
    
def load_model(config, device):
    model = instantiate(config.model).to(device)
    model.eval()
    
    checkpoint_path = config.inferencer.from_pretrained
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"WARNING: checkpoint {checkpoint_path} not found, using random weights")
    
    return model

def save_audio(audio_tensor, output_path, sample_rate):
    audio_cpu = audio_tensor.squeeze().detach().cpu()
    if audio_cpu.dim() > 1: audio_cpu = audio_cpu.squeeze()
    if audio_cpu.dim() > 1: audio_cpu = audio_cpu[0]
    
    if audio_cpu.abs().max() > 0:
        audio_save = audio_cpu / audio_cpu.abs().max() * 0.95
    else:
        audio_save = audio_cpu
    
    audio_int16 = (audio_save.numpy() * 32767).astype(np.int16)
    wavfile.write(str(output_path), sample_rate, audio_int16)
    return len(audio_int16) / sample_rate

def measure_performance_metrics(model, device, sample_rate=22050, hop_length=256, n_mels=80,
                                 audio_seconds=10, batch_size=1, use_compile=True, use_fp16=False):
    print("Оценка производительности.")
    generator = model.generator if hasattr(model, 'generator') else model

    if hasattr(generator, "remove_weight_norm"):
        try:
            generator.remove_weight_norm()
            print("weight_norm удалён для инференса")
        except Exception as e:
            print(f"remove_weight_norm пропущен: {e}")

    num_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print(f"Параметры генератора: {num_params / 1_000_000:.2f} M")

    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    frames = int(audio_seconds * sample_rate / hop_length)
    dummy_mel = torch.randn(batch_size, n_mels, frames, device=device)

    if use_fp16 and device == "cuda":
        generator = generator.half()
        dummy_mel = dummy_mel.half()

    if THOP_AVAILABLE:
        try:
            from copy import deepcopy
            macs, _ = profile(deepcopy(generator), inputs=(dummy_mel[:1, :, :int(sample_rate/hop_length)],), verbose=False)
            print(f"Сложность: {macs / 1_000_000:.2f} M MACs (для 1 сек, batch=1)")
        except Exception as e:
            print(f"Не удалось посчитать MACs: {e}")

    run_fn = generator
    if use_compile:
        try:
            run_fn = torch.compile(generator, mode="reduce-overhead", fullgraph=False)
            print("Используется torch.compile (reduce-overhead)")
        except Exception as e:
            print(f"torch.compile пропущен: {e}")
            run_fn = generator

    print(f"Вход: batch={batch_size}, длительность={audio_seconds}с, frames={frames}")

    with torch.no_grad():
        print("Прогрев (20 итераций)...")
        for _ in range(20):
            _ = run_fn(dummy_mel)
        if device == "cuda":
            torch.cuda.synchronize()

        iters = 50 if device == "cuda" else 10
        print(f"Замер RTF ({iters} итераций)...")

        if device == "cuda":
            starter = torch.cuda.Event(enable_timing=True)
            ender = torch.cuda.Event(enable_timing=True)
            starter.record()
            for _ in range(iters):
                _ = run_fn(dummy_mel)
            ender.record()
            torch.cuda.synchronize()
            total_gen_time = starter.elapsed_time(ender) / 1000.0
        else:
            start_time = time.perf_counter()
            for _ in range(iters):
                _ = run_fn(dummy_mel)
            total_gen_time = time.perf_counter() - start_time

        total_audio_duration = audio_seconds * batch_size * iters
        rtf = total_gen_time / total_audio_duration

        print(f"RTF на {device.upper()}: {rtf:.5f}x")
        print(f"Скорость генерации: {1/rtf:.2f} секунд аудио за 1 секунду работы")

def evaluate_test_set(model, dataloader, device, config):
    metrics_calc = TTSMetrics()
    sample_rate = config.inferencer.get("sample_rate", 22050)
    limit = config.inferencer.get("eval_samples_limit", 100)
    
    results = {'MCD': [], 'LSD': [], 'MelDist': [], 'F0-RMSE': [], 'V/UV F1': [], 'STOI': [], 'PESQ': []}
    
    print(f"Оценка качества на тестовой выборке (лимит: {limit} файлов)...")
    
    count = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating Metrics"):
            if count >= limit: break
                
            mel = batch["mel"].to(device)
            orig_audio_tensor = batch["audio"]
            
            gen_audio_tensor = model.generator(mel)
            
            for i in range(mel.shape[0]):
                if count >= limit: break
                
                ref_wav = orig_audio_tensor[i].squeeze().cpu().numpy()
                deg_wav = gen_audio_tensor[i].squeeze().cpu().numpy()
                
                _, trim_index = librosa.effects.trim(ref_wav, top_db=60)
                start, end = trim_index[0], trim_index[1]
                
                ref_wav = ref_wav[start:end]
                deg_wav = deg_wav[start:end] 
                
                mcd_val = metrics_calc.calculate_mcd(ref_wav, deg_wav, sample_rate)
                lsd_val = metrics_calc.calculate_lsd(ref_wav, deg_wav, sample_rate)
                mel_dist_val = metrics_calc.calculate_mel_distance(ref_wav, deg_wav, sample_rate)
                f0_rmse, vuv_f1 = metrics_calc.calculate_f0_metrics(ref_wav, deg_wav, sample_rate)
                stoi_val = metrics_calc.calculate_stoi(ref_wav, deg_wav, sample_rate)
                pesq_val = metrics_calc.calculate_pesq(ref_wav, deg_wav, sample_rate)
                
                results['MCD'].append(mcd_val)
                results['LSD'].append(lsd_val)
                results['MelDist'].append(mel_dist_val)
                results['F0-RMSE'].append(f0_rmse)
                results['V/UV F1'].append(vuv_f1)
                results['STOI'].append(stoi_val)
                results['PESQ'].append(pesq_val)
                
                count += 1

    print("Итоговые метрики:")
    print(f"MCD (↓):      {np.nanmean(results['MCD']):.3f}")
    print(f"LSD (↓):      {np.nanmean(results['LSD']):.3f}")
    print(f"MelDist (↓):  {np.nanmean(results['MelDist']):.3f}")
    print(f"F0-RMSE (↓):  {np.nanmean(results['F0-RMSE']):.2f}")
    print(f"V/UV F1 (↑):  {np.nanmean(results['V/UV F1']):.3f}")
    print(f"STOI (↑):     {np.nanmean(results['STOI']):.3f}")
    print(f"PESQ (↑):     {np.nanmean(results['PESQ']):.3f}")

def generate_clean_demos(model, dataloader, device, config, save_path, model_name, num_demos=3):
    print(f"Генерация демо (сохраняем {num_demos} цельных фраз)...")
    sample_rate = config.inferencer.get("sample_rate", 22050)
    
    count = 0
    with torch.no_grad():
        for batch in dataloader:
            if count >= num_demos:
                break
                
            mel = batch["mel"][0:1].to(device)
            audio = batch["audio"][0:1]
            
            generated_audio = model.generator(mel)
            
            gen_path = save_path / f"{model_name}_demo_phrase_{count+1}_gen.wav"
            orig_path = save_path / f"{model_name}_demo_phrase_{count+1}_orig.wav"
            
            save_audio(generated_audio, gen_path, sample_rate)
            save_audio(audio, orig_path, sample_rate)
            
            print(f"Сохранена фраза {count+1}:")
            print(f" -> {gen_path}")
            
            count += 1

@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    torch.set_num_threads(os.cpu_count())
    set_random_seed(config.inferencer.seed)
    device = "cuda" if config.inferencer.device == "auto" and torch.cuda.is_available() else config.inferencer.device

    save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    save_path.mkdir(exist_ok=True, parents=True)

    dataloaders, _ = get_dataloaders(config, device)
    
    test_dataloader = dataloaders.get("test")
    if test_dataloader is None:
        print("Внимание: Dataloader 'test' не найден, беру первый попавшийся.")
        test_dataloader = list(dataloaders.values())[0]

    model_name = config.model.get("_target_", "model").split(".")[-1].lower()
    model = load_model(config, device)

    print("=" * 60)
    print(f"Модель: {model_name} | Device: {device}")

    sample_rate = config.inferencer.get("sample_rate", 22050)
    measure_performance_metrics(
        model,
        device,
        sample_rate=sample_rate,
        audio_seconds=2,
        batch_size=100,
        use_compile=True,
        use_fp16=False,
    )

    print(f"Замер на CPU для {model_name}")
    model_cpu = load_model(config, "cpu")
    measure_performance_metrics(
        model_cpu,
        "cpu",
        sample_rate=sample_rate,
        audio_seconds=10,
        batch_size=1,
        use_compile=False,
        use_fp16=False,
    )
    del model_cpu
    
    if config.inferencer.get("run_evaluation", True):
        evaluate_test_set(model, test_dataloader, device, config)
    
    num_demos = config.inferencer.get("num_demos", 5)
    generate_clean_demos(model, test_dataloader, device, config, save_path, model_name, num_demos=num_demos)

if __name__ == "__main__":
    main()