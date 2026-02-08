# audiomae_ft_random_position.py
from __future__ import annotations
import os, math, glob, random, sys, json, hashlib, time, csv, warnings, logging
from dataclasses import dataclass, asdict, replace
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio
from torchaudio.compliance import kaldi
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from src.utils import (
    suppress_warnings,
    set_deterministic_mode,
    get_audio_info_safe,
)
from src.unified_config import UnifiedConfig

dataset_mean = UnifiedConfig().dataset_mean
dataset_std = UnifiedConfig().dataset_std

suppress_warnings()

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
@dataclass
@dataclass
class AudioMAEFTConfig:
    dataset: str = "DINOS"
    train_data_path: Optional[str] = None
    log_dir: str = "./AudioMAE_Models_FT/logs"
    checkpoint_dir: str = "./AudioMAE_Models_FT"

    # Input (1초 오디오)
    sr: int = 48000
    sample_len: int = 48000                # 1초 @ 48kHz
    target_sr: int = 16000

    # FBank / AudioMAE input shape (10초 기준)
    num_mel_bins: int = 128
    target_time_frames: int = 1024         # 10초 분량
    
    # Random positioning
    use_random_position: bool = True       # False면 항상 앞에 배치
    position_strategy: str = "uniform"     # uniform | start | end | center

    # Training
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-5
    lr_min: float = 1e-6
    wd: float = 0.05
    gradient_clip: float = 1.0
    num_workers: int = 32
    seed: int = 42

    # Model
    model_name: str = "hf_hub:gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft_as20k"
    pooling: str = "cls"
    freeze_backbone: bool = False

    # Logging
    tensorboard: bool = True
    log_interval: int = 1


def setup_logging(log_dir: str):
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "training.log"), logging.StreamHandler()],
    )
    return logging.getLogger("AudioMAE-FT-RandomPos")


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_experiment_config(cfg: AudioMAEFTConfig, save_dir: str):
    config_dict = asdict(cfg)
    config_str = json.dumps(config_dict, sort_keys=True)
    experiment_id = hashlib.md5(config_str.encode()).hexdigest()[:8]

    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    config_path = save_dir / f"config_{experiment_id}.json"

    with open(config_path, "w") as f:
        json.dump(
            {
                "experiment_id": experiment_id,
                "timestamp": datetime.now().isoformat(),
                "config": config_dict,
                "python_version": sys.version,
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            },
            f,
            indent=2,
        )
    return experiment_id


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class RobustAudioDataset(Dataset):
    """1초 오디오 데이터셋"""
    def __init__(self, wav_root: str | os.PathLike | List[str | os.PathLike], cfg: AudioMAEFTConfig,
                 clips_per_folder: int | None = None, max_retries: int = 3, use_cache: bool = True):
        super().__init__()
        self.cfg = cfg
        self.max_retries = max_retries
        self.error_count = defaultdict(int)
        self._resamplers = {}  # SR별 resampler 캐싱

        if cfg.seed is not None:
            random.seed(cfg.seed)

        if isinstance(wav_root, (list, tuple)):
            wav_roots = [Path(root) for root in wav_root]
        else:
            wav_roots = [Path(wav_root)]

        cache_key = "_".join(sorted([str(root) for root in wav_roots])) + f"__{cfg.sample_len}__{cfg.sr}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_dir = Path("./cache")
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"dataset_cache_{cache_hash}.json"

        clip_map: Dict[str, List[tuple[str, int, str]]] = defaultdict(list)

        if use_cache and cache_file.exists():
            try:
                logging.info(f"Loading dataset from cache: {cache_file}")
                with open(cache_file, "r") as f:
                    cache_data = json.load(f)
                cached_files = cache_data.get("files", [])
                if cached_files:
                    sample_size = min(10, len(cached_files))
                    samples = random.sample(cached_files, sample_size)
                    all_exist = all(os.path.exists(fp) for fp, _, _ in samples)
                    if all_exist:
                        for fp, n_clips, folder in cached_files:
                            for k in range(n_clips):
                                clip_map[folder].append((fp, k * cfg.sample_len, folder))
                        logging.info(f"Cache OK: {len(cached_files)} files, {len(clip_map)} folders")
                    else:
                        logging.warning("Cache invalid, rescan...")
                        clip_map = defaultdict(list)
            except Exception as e:
                logging.warning(f"Cache load failed: {e}, rescan...")
                clip_map = defaultdict(list)

        if not clip_map:
            logging.info(f"Scanning: {[str(r) for r in wav_roots]}")
            audio_patterns = ["**/*.wav", "**/*.WAV", "**/*.mp3", "**/*.flac"]
            audio_files = []
            for root in wav_roots:
                for p in audio_patterns:
                    audio_files.extend(glob.glob(str(root / p), recursive=True))
            logging.info(f"Found {len(audio_files)} files")

            cache_files = []
            valid_files = 0

            for fp in tqdm(audio_files, desc="Analyzing file info"):
                try:
                    info = get_audio_info_safe(fp)
                    if info is None:
                        continue
                    frames, file_sr = info
                    if frames < cfg.sample_len:
                        continue

                    valid_files += 1
                    n_clips = math.ceil(frames / cfg.sample_len)

                    fp_path = Path(fp)
                    folder = None
                    for root in wav_roots:
                        try:
                            rel_path = fp_path.relative_to(root)
                            folder = rel_path.parts[0]
                            break
                        except ValueError:
                            continue
                    if folder is None:
                        folder = fp_path.parent.name

                    cache_files.append((fp, n_clips, folder))
                    for k in range(n_clips):
                        clip_map[folder].append((fp, k * cfg.sample_len, folder))
                except Exception:
                    continue

            logging.info(f"Valid files: {valid_files}, folders: {len(clip_map)}")

            if use_cache:
                try:
                    cache_data = {
                        "roots": [str(r) for r in wav_roots],
                        "sample_len": cfg.sample_len,
                        "sr": cfg.sr,
                        "files": cache_files,
                        "timestamp": datetime.now().isoformat(),
                    }
                    with open(cache_file, "w") as f:
                        json.dump(cache_data, f)
                    logging.info(f"Cache saved: {cache_file}")
                except Exception as e:
                    logging.warning(f"Cache save failed: {e}")

        self.index_list = []
        for fld, clips in clip_map.items():
            chosen = clips if clips_per_folder is None else random.sample(clips, min(len(clips), clips_per_folder))
            self.index_list.extend(chosen)

        self.class_to_idx = {f: i for i, f in enumerate(sorted(clip_map))}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        logging.info(f"Total clips: {len(self.index_list)} | num_classes: {len(self.class_to_idx)}")

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        for retry in range(self.max_retries):
            fp = None
            try:
                fp, start, folder = self.index_list[idx]
                if not os.path.exists(fp):
                    raise FileNotFoundError(fp)

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import soundfile as sf
                    try:
                        audio, _sr = sf.read(
                            fp,
                            start=start,
                            frames=self.cfg.sample_len,
                            dtype="float32",
                            always_2d=True,
                        )
                        audio = audio.T
                    except Exception:
                        audio, _sr = sf.read(fp, dtype="float32", always_2d=True)
                        audio = audio.T
                        end_frame = start + self.cfg.sample_len
                        audio = audio[:, start:end_frame]

                # SR mismatch -> resample (캐싱)
                if _sr != self.cfg.sr:
                    if _sr not in self._resamplers:
                        self._resamplers[_sr] = torchaudio.transforms.Resample(_sr, self.cfg.sr)
                    audio_tensor = torch.from_numpy(audio)
                    audio_tensor = self._resamplers[_sr](audio_tensor)
                    audio = audio_tensor.numpy()

                # mono
                if audio.shape[0] > 1:
                    audio = audio.mean(0)
                else:
                    audio = audio[0]

                # pad/trim to sample_len
                audio = np.pad(audio, (0, max(0, self.cfg.sample_len - len(audio))))[: self.cfg.sample_len]
                waveform = torch.tensor(audio, dtype=torch.float32)
                label = self.class_to_idx[folder]
                return waveform, label

            except Exception as e:
                if fp is not None:
                    self.error_count[fp] += 1
                logging.error(f"Error loading {fp} (attempt {retry+1}/{self.max_retries}): {e}")
                if retry == self.max_retries - 1:
                    # Fallback: zero tensor
                    waveform = torch.zeros(self.cfg.sample_len, dtype=torch.float32)
                    label = random.choice(list(self.class_to_idx.values()))
                    return waveform, label
                time.sleep(0.1 * (retry + 1))


def create_dataloader(dataset, cfg: AudioMAEFTConfig, train: bool = True):
    num_workers = min(cfg.num_workers, os.cpu_count() or 1)
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn,
        drop_last=train,
    )


# ------------------------------------------------------------
# AudioMAE Preprocessor with Random Positioning
# ------------------------------------------------------------
class AudioMAEPreprocessorRandomPos(nn.Module):
    """
    1초 오디오를 10초 프레임의 임의 위치에 배치
    
    Input:  waveform_48k (B,T) 1초
    Output: fbank_norm (B,1,1024,128) 10초 분량 (대부분 zero-padded)
    """
    def __init__(self, cfg: AudioMAEFTConfig, return_on_device: bool = False, clamp_waveform: bool = False):
        super().__init__()
        self.cfg = cfg
        self.return_on_device = return_on_device
        self.clamp_waveform = clamp_waveform

        self.resampler = torchaudio.transforms.Resample(orig_freq=cfg.sr, new_freq=cfg.target_sr)

        self.register_buffer("mean", torch.tensor(float(dataset_mean), dtype=torch.float32))
        self.register_buffer("std",  torch.tensor(float(dataset_std),  dtype=torch.float32))

        # 1초 오디오 @ 16kHz
        duration_1sec = float(cfg.sample_len) / float(cfg.sr)
        self.target_len_1sec_16k = int(round(duration_1sec * cfg.target_sr))  # 16000
        
        # 10초 분량 @ 16kHz
        duration_10sec = 10.0
        self.target_len_10sec_16k = int(round(duration_10sec * cfg.target_sr))  # 160000

    def _ensure_batch_mono(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 3:
            wav = wav.mean(dim=1)
        elif wav.dim() == 2:
            pass
        else:
            raise ValueError(f"Unsupported waveform shape: {tuple(wav.shape)}")
        return wav

    def _pad_or_trim_1d(self, wav: torch.Tensor, target_len: int) -> torch.Tensor:
        T = wav.shape[-1]
        if T > target_len:
            wav = wav[..., :target_len]
        elif T < target_len:
            wav = F.pad(wav, (0, target_len - T))
        return wav

    def _get_random_position(self, max_offset: int) -> int:
        """1초 오디오를 배치할 시작 위치 결정"""
        if not self.cfg.use_random_position or not self.training:
            return 0  # 항상 앞에 배치 (inference)
        
        strategy = self.cfg.position_strategy
        
        if strategy == "uniform":
            # 0 ~ max_offset 사이 균등 분포
            return random.randint(0, max_offset)
        elif strategy == "start":
            # 앞쪽 30%에 집중
            return random.randint(0, int(max_offset * 0.3))
        elif strategy == "end":
            # 뒤쪽 30%에 집중
            return random.randint(int(max_offset * 0.7), max_offset)
        elif strategy == "center":
            # 중앙 40%에 집중
            center = max_offset // 2
            margin = int(max_offset * 0.2)
            return random.randint(max(0, center - margin), min(max_offset, center + margin))
        else:
            return random.randint(0, max_offset)

    def _random_position_padding(self, wav16_1sec: torch.Tensor) -> torch.Tensor:
        """
        1초 오디오를 10초 길이의 랜덤 위치에 배치
        wav16_1sec: (B, 16000)
        returns: (B, 160000)
        """
        B = wav16_1sec.shape[0]
        wav16_10sec = torch.zeros(B, self.target_len_10sec_16k, dtype=wav16_1sec.dtype, device=wav16_1sec.device)
        
        max_offset = self.target_len_10sec_16k - self.target_len_1sec_16k
        
        for b in range(B):
            offset = self._get_random_position(max_offset)
            wav16_10sec[b, offset:offset + self.target_len_1sec_16k] = wav16_1sec[b]
        
        return wav16_10sec

    def _pad_or_trim_time(self, fb: torch.Tensor) -> torch.Tensor:
        t = fb.shape[1]
        if t > self.cfg.target_time_frames:
            fb = fb[:, : self.cfg.target_time_frames, :]
        elif t < self.cfg.target_time_frames:
            fb = F.pad(fb, (0, 0, 0, self.cfg.target_time_frames - t))
        return fb

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (B, 48000) - 1초 @ 48kHz
        returns: (B, 1, 1024, 128) - 10초 분량
        """
        out_device = waveform.device

        wav = self._ensure_batch_mono(waveform).cpu().float()

        if self.clamp_waveform:
            wav = wav.clamp(-1.0, 1.0)

        # 48k -> 16k (1초)
        wav16_1sec = self.resampler(wav)
        wav16_1sec = self._pad_or_trim_1d(wav16_1sec, self.target_len_1sec_16k)

        # 1초 오디오를 10초 길이의 랜덤 위치에 배치
        wav16_10sec = self._random_position_padding(wav16_1sec)

        # fbank (10초 분량)
        feats = []
        for b in range(wav16_10sec.shape[0]):
            fb = kaldi.fbank(
                wav16_10sec[b].unsqueeze(0),
                htk_compat=True,
                window_type="hanning",
                num_mel_bins=self.cfg.num_mel_bins,
                sample_frequency=self.cfg.target_sr,
                frame_length=25.0,
                frame_shift=10.0,
            )
            feats.append(fb)
        fb = torch.stack(feats, dim=0)

        # time pad/trim to 1024
        fb = self._pad_or_trim_time(fb)

        # normalize
        fb = (fb - self.mean) / (self.std * 2.0)

        # shape (B,1,1024,128)
        fb = fb.contiguous().view(fb.shape[0], 1, self.cfg.target_time_frames, self.cfg.num_mel_bins)

        if self.return_on_device and fb.device != out_device:
            fb = fb.to(out_device, non_blocking=True)

        return fb


# ------------------------------------------------------------
# AudioMAE Classifier
# ------------------------------------------------------------
class AudioMAEClassifier(nn.Module):
    def __init__(self, cfg: AudioMAEFTConfig, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        self.backbone, embed_dim = self._load_backbone(cfg.model_name)

        if cfg.freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.classifier = nn.Linear(embed_dim, num_classes)

    def _load_backbone(self, model_name: str):
        try:
            import timm
        except Exception as e:
            raise RuntimeError("timm is required for AudioMAE.") from e

        candidates = [model_name]
        if not model_name.startswith("hf_hub:"):
            candidates.append(f"hf_hub:{model_name}")

        last_err = None
        for name in candidates:
            try:
                model = timm.create_model(name, pretrained=True, num_classes=0)
                embed_dim = getattr(model, "num_features", None)
                if embed_dim is None:
                    with torch.no_grad():
                        dummy = torch.zeros(1, 1, 1024, 128)
                        out = model.forward_features(dummy)
                        if isinstance(out, dict):
                            out = out.get("x", out.get("last_hidden_state"))
                        embed_dim = out.shape[-1]
                return model, int(embed_dim)
            except Exception as e:
                last_err = e

        raise RuntimeError(f"Failed to load backbone: {last_err}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.forward_features(x)
        if isinstance(feats, dict):
            feats = feats.get("x", feats.get("last_hidden_state"))

        if feats.dim() == 3:
            if self.cfg.pooling == "cls":
                pooled = feats[:, 0]
            else:
                pooled = feats[:, 1:].mean(dim=1)
        else:
            pooled = feats

        logits = self.classifier(pooled)
        return logits


# ------------------------------------------------------------
# Checkpoint Manager
# ------------------------------------------------------------
class CheckpointManager:
    def __init__(self, save_dir: str, cfg: AudioMAEFTConfig, accelerator: Optional[Accelerator] = None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.cfg = cfg
        self.accelerator = accelerator

    def save_checkpoint(self, model, optimizer, epoch, metrics: dict):
        if self.accelerator and not self.accelerator.is_main_process:
            return
        model_to_save = self.accelerator.unwrap_model(model) if self.accelerator else model
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.cfg),
        }
        filename = f"audiomae_randompos_{self.cfg.dataset}_epoch_{epoch+1:04d}.pth"
        torch.save(ckpt, self.save_dir / filename)


# ------------------------------------------------------------
# Train
# ------------------------------------------------------------
def accuracy_top1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(model, preproc, loader, optimizer, cfg: AudioMAEFTConfig, accelerator: Accelerator, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    pbar = tqdm(loader, leave=False, desc="train" if train else "eval", disable=not accelerator.is_main_process)
    for batch_idx, (waveform, label) in enumerate(pbar):
        label = label.to(accelerator.device)

        waveform = waveform.cpu()
        with torch.no_grad():
            feats = preproc(waveform)
        feats = feats.to(accelerator.device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(feats)
            loss = F.cross_entropy(logits, label)

        if train:
            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
            optimizer.step()

        acc = accuracy_top1(logits.detach(), label.detach())
        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

        pbar.set_postfix({"loss": float(loss.item()), "acc": float(acc)})

    return {
        "loss": total_loss / max(1, n_batches),
        "acc": total_acc / max(1, n_batches),
    }


# ------------------------------------------------------------
# Programmatic Entry
# ------------------------------------------------------------
def run_finetune(
    cfg: Optional[AudioMAEFTConfig] = None,
    datasets: Optional[List[str]] = None,
    device: Optional[str] = None,
    use_multi_gpu: bool = False,
):
    if device and device.startswith("cuda:") and not use_multi_gpu:
        try:
            torch.cuda.set_device(int(device.split(":")[1]))
        except Exception:
            pass
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16",
        gradient_accumulation_steps=1,
        kwargs_handlers=[ddp_kwargs],
        cpu=(device == "cpu"),
    )

    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend")

    # Dataset selection
    if datasets is None:
        if accelerator.is_main_process:
            print("\n=== Select fine-tuning dataset ===")
            print("1. DINOS")
            print("2. DCASE")
            print("3. Hybrid (DINOS+DCASE)")
            print("4. All")
            choice_str = input("Enter your choice (1-4): ")
            try:
                choice = int(choice_str)
                if choice not in [1, 2, 3, 4]:
                    print("Invalid selection.")
                    return
            except ValueError:
                print("Please enter a number.")
                return
        else:
            choice = 0

        if accelerator.num_processes > 1:
            choice_tensor = torch.tensor([choice], dtype=torch.long, device=accelerator.device)
            torch.distributed.broadcast(choice_tensor, src=0)
            choice = choice_tensor.item()

        mapping = {1: "DINOS", 2: "DCASE", 3: "Hybrid", 4: "All"}
        if choice not in mapping:
            return

        datasets_to_run = ["DINOS", "DCASE", "Hybrid"] if mapping[choice] == "All" else [mapping[choice]]
    else:
        datasets_to_run = datasets

    for dataset_name in datasets_to_run:
        if cfg is None:
            cfg_run = AudioMAEFTConfig(dataset=dataset_name)
        else:
            cfg_run = replace(cfg, dataset=dataset_name)

        if device is not None and hasattr(cfg_run, "device"):
            cfg_run.device = device

        if cfg_run.train_data_path is None:
            cfg_run.train_data_path = f"./Datasets/{cfg_run.dataset}_Pretrain"

        set_deterministic_mode(cfg_run.seed)
        logger = setup_logging(cfg_run.log_dir)

        if accelerator.is_main_process:
            exp_id = save_experiment_config(cfg_run, cfg_run.log_dir)
            logger.info(f"Experiment {exp_id} | dataset={cfg_run.dataset}")
            logger.info(f"Random Position Strategy: {cfg_run.position_strategy}")
            logger.info(f"1-sec audio positioned randomly in 10-sec frame")

        Path(cfg_run.checkpoint_dir).mkdir(exist_ok=True, parents=True)

        logger.info("Loading dataset...")
        if cfg_run.dataset == "Hybrid":
            train_paths = ["./Datasets/DCASE_Pretrain", "./Datasets/DINOS_Pretrain"]
            train_dataset = RobustAudioDataset(train_paths, cfg_run)
        else:
            train_dataset = RobustAudioDataset(cfg_run.train_data_path, cfg_run)

        num_classes = len(train_dataset.class_to_idx)
        logger.info(f"num_classes={num_classes} | train_size={len(train_dataset)}")

        train_loader = create_dataloader(train_dataset, cfg_run, train=True)

        # Synchronize after dataset creation
        accelerator.wait_for_everyone()

        preproc = AudioMAEPreprocessorRandomPos(cfg_run)
        model = AudioMAEClassifier(cfg_run, num_classes=num_classes)

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg_run.lr,
            weight_decay=cfg_run.wd,
        )

        # Initialize cosine annealing scheduler with minimum learning rate
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg_run.epochs, eta_min=cfg_run.lr_min)

        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

        # Ensure all processes are synchronized before training
        accelerator.wait_for_everyone()

        ckpt_mgr = CheckpointManager(cfg_run.checkpoint_dir, cfg_run, accelerator)

        csv_path = Path(cfg_run.log_dir) / f"ft_log_{cfg_run.dataset}.csv"
        csv_writer = None
        csvfile = None

        if accelerator.is_main_process:
            csvfile = open(csv_path, "w", newline="")
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(["epoch", "train_loss", "train_acc", "lr", "epoch_time"])

        logger.info("Starting fine-tuning with random positioning...")
        for epoch in range(cfg_run.epochs):
            epoch_start = time.time()

            train_metrics = run_epoch(
                model=model,
                preproc=preproc,
                loader=train_loader,
                optimizer=optimizer,
                cfg=cfg_run,
                accelerator=accelerator,
                train=True,
            )

            train_loss = accelerator.gather(torch.tensor(train_metrics["loss"], device=accelerator.device)).mean().item()
            train_acc = accelerator.gather(torch.tensor(train_metrics["acc"], device=accelerator.device)).mean().item()
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_time = time.time() - epoch_start

            if accelerator.is_main_process:
                logger.info(
                    f"[{cfg_run.dataset}] Epoch {epoch+1}/{cfg_run.epochs} | "
                    f"loss={train_loss:.4f} acc={train_acc:.4f} lr={current_lr:.6g} time={epoch_time:.1f}s"
                )
                csv_writer.writerow([epoch + 1, train_loss, train_acc, current_lr, epoch_time])
                csvfile.flush()

            accelerator.wait_for_everyone()
            ckpt_mgr.save_checkpoint(model, optimizer, epoch, {"train_loss": train_loss, "train_acc": train_acc})

            # Update learning rate scheduler
            scheduler.step()

        if accelerator.is_main_process and csvfile is not None:
            csvfile.close()

        logger.info("Fine-tuning completed.")

        # Synchronize after completing each dataset
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("All datasets finished.")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="AudioMAE Fine-tuning with Random Positioning")
    parser.add_argument("--dataset", type=str, default=None, choices=["DINOS", "DCASE", "Hybrid", "All"],
                       help="Dataset selection (overrides interactive prompt)")
    args = parser.parse_args()

    datasets = None
    if args.dataset:
        if args.dataset == "All":
            datasets = ["DINOS", "DCASE", "Hybrid"]
        else:
            datasets = [args.dataset]

    run_finetune(datasets=datasets)


if __name__ == "__main__":
    main()
