# audiomae_pretrain_1sec.py
from __future__ import annotations
import os, math, glob, random, sys, json, hashlib, time, csv, warnings, logging
from dataclasses import dataclass, asdict
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
class AudioMAE1SecConfig:
    dataset: str = "DCASE"
    train_data_path: Optional[str] = None
    log_dir: str = "./AudioMAE_Models_Scratch/logs"
    checkpoint_dir: str = "./AudioMAE_Models_Scratch"

    # Input (1초)
    sr: int = 48000
    sample_len: int = 48000
    target_sr: int = 48000

    # FBank (1초 기준)
    num_mel_bins: int = 128
    target_time_frames: int = 128        # 1초 ≈ 100 프레임 → 128로 패딩

    # ViT Architecture
    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 8
    num_heads: int = 12
    mlp_ratio: float = 4.0
    
    # MAE
    mask_ratio: float = 0.75
    decoder_embed_dim: int = 256
    decoder_depth: int = 4
    decoder_num_heads: int = 8

    # Training
    batch_size: int = 256
    epochs: int = 20
    warmup_epochs: int = 0
    lr: float = 1.5e-4                    # scaled by batch_size
    min_lr: float = 1e-6
    lr_min: float = 1e-6
    wd: float = 0.05
    gradient_clip: float = 3.0
    num_workers: int = 32
    seed: int = 42

    # Fine-tuning mode
    finetune: bool = False                # True면 classifier head 추가
    num_classes: int = 0
    freeze_encoder: bool = False

    # Logging
    log_interval: int = 1
    save_interval: int = 10


def setup_logging(log_dir: str):
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_dir / "training.log"), logging.StreamHandler()],
    )
    return logging.getLogger("AudioMAE-1sec")


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class AudioDataset1Sec(Dataset):
    """1초 오디오 전용 데이터셋"""
    def __init__(self, wav_root: str | os.PathLike | List[str | os.PathLike], cfg: AudioMAE1SecConfig,
                 clips_per_folder: int | None = None, max_retries: int = 3, use_cache: bool = True):
        super().__init__()
        self.cfg = cfg
        self.max_retries = max_retries
        self._resamplers = {}

        if cfg.seed is not None:
            random.seed(cfg.seed)

        if isinstance(wav_root, (list, tuple)):
            wav_roots = [Path(root) for root in wav_root]
        else:
            wav_roots = [Path(wav_root)]

        cache_key = "_".join(sorted([str(root) for root in wav_roots])) + f"|len={cfg.sample_len}"
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
                cached_sample_len = cache_data.get("sample_len")
                if cached_sample_len == cfg.sample_len and cached_files:
                    sample_size = min(10, len(cached_files))
                    samples = random.sample(cached_files, sample_size)
                    all_exist = all(os.path.exists(fp) for fp, _, _ in samples)
                    if all_exist:
                        for fp, n_clips, folder in cached_files:
                            for k in range(n_clips):
                                clip_map[folder].append((fp, k * cfg.sample_len, folder))
                        logging.info(f"Cache loaded: {len(cached_files)} files, {len(clip_map)} folders")
                    else:
                        logging.warning("Cache validation failed, rescanning...")
                        clip_map = defaultdict(list)
                else:
                    logging.warning("Cache metadata mismatch, rescanning...")
                    clip_map = defaultdict(list)
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}, rescanning...")
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
            for fp in tqdm(audio_files, desc="Analyzing"):
                try:
                    info = get_audio_info_safe(fp)
                    if info is None:
                        continue
                    frames, file_sr = info
                    if frames < cfg.sample_len:
                        continue

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

            if use_cache:
                try:
                    cache_data = {
                        "roots": [str(root) for root in wav_roots],
                        "sample_len": cfg.sample_len,
                        "files": cache_files,
                        "timestamp": datetime.now().isoformat(),
                    }
                    with open(cache_file, "w") as f:
                        json.dump(cache_data, f)
                    logging.info(f"Cache saved to {cache_file}")
                except Exception as e:
                    logging.warning(f"Failed to save cache: {e}")

        self.index_list = []
        for fld, clips in clip_map.items():
            chosen = clips if clips_per_folder is None else random.sample(clips, min(len(clips), clips_per_folder))
            self.index_list.extend(chosen)

        self.class_to_idx = {f: i for i, f in enumerate(sorted(clip_map))}
        logging.info(f"Total clips: {len(self.index_list)} | folders: {len(self.class_to_idx)}")

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        for retry in range(self.max_retries):
            fp = None
            try:
                fp, start, folder = self.index_list[idx]
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import soundfile as sf
                    audio, _sr = sf.read(fp, start=start, frames=self.cfg.sample_len, dtype="float32", always_2d=True)
                    audio = audio.T

                if _sr != self.cfg.sr:
                    if _sr not in self._resamplers:
                        self._resamplers[_sr] = torchaudio.transforms.Resample(_sr, self.cfg.sr)
                    audio_tensor = torch.from_numpy(audio)
                    audio_tensor = self._resamplers[_sr](audio_tensor)
                    audio = audio_tensor.numpy()

                if audio.shape[0] > 1:
                    audio = audio.mean(0)
                else:
                    audio = audio[0]

                audio = np.pad(audio, (0, max(0, self.cfg.sample_len - len(audio))))[: self.cfg.sample_len]
                waveform = torch.tensor(audio, dtype=torch.float32)
                label = self.class_to_idx[folder]
                return waveform, label

            except Exception as e:
                logging.error(f"Error loading {fp} (retry {retry+1}): {e}")
                if retry == self.max_retries - 1:
                    waveform = torch.zeros(self.cfg.sample_len, dtype=torch.float32)
                    label = 0
                    return waveform, label
                time.sleep(0.1 * (retry + 1))


# ------------------------------------------------------------
# Preprocessor (1초 전용)
# ------------------------------------------------------------
class FBankPreprocessor1Sec(nn.Module):
    """1초 오디오용 FBank 전처리"""
    def __init__(self, cfg: AudioMAE1SecConfig):
        super().__init__()
        self.cfg = cfg
        self.resampler = torchaudio.transforms.Resample(cfg.sr, cfg.target_sr)
        
        self.register_buffer("mean", torch.tensor(float(dataset_mean), dtype=torch.float32))
        self.register_buffer("std",  torch.tensor(float(dataset_std),  dtype=torch.float32))
        
        duration_sec = float(cfg.sample_len) / float(cfg.sr)
        self.target_len_16k = int(round(duration_sec * cfg.target_sr))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        waveform: (B, 48000) @ 48kHz
        returns: (B, 128, 128) - [time, freq]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        wav = waveform.cpu().float()
        wav16 = self.resampler(wav)
        
        # pad/trim
        T = wav16.shape[-1]
        if T > self.target_len_16k:
            wav16 = wav16[..., :self.target_len_16k]
        elif T < self.target_len_16k:
            wav16 = F.pad(wav16, (0, self.target_len_16k - T))
        
        # fbank
        feats = []
        for b in range(wav16.shape[0]):
            fb = kaldi.fbank(
                wav16[b].unsqueeze(0),
                htk_compat=True,
                window_type="hanning",
                num_mel_bins=self.cfg.num_mel_bins,
                sample_frequency=self.cfg.target_sr,
                frame_length=25.0,
                frame_shift=10.0,
            )
            feats.append(fb)
        fb = torch.stack(feats, dim=0)  # (B, time, 128)
        
        # pad/trim to target_time_frames
        t = fb.shape[1]
        if t > self.cfg.target_time_frames:
            fb = fb[:, :self.cfg.target_time_frames, :]
        elif t < self.cfg.target_time_frames:
            fb = F.pad(fb, (0, 0, 0, self.cfg.target_time_frames - t))
        
        # normalize
        fb = (fb - self.mean) / (self.std * 2.0)
        
        return fb  # (B, 128, 128)


# ------------------------------------------------------------
# Vision Transformer Components
# ------------------------------------------------------------
class PatchEmbed(nn.Module):
    """Spectrogram to patch embedding"""
    def __init__(self, time_frames=128, freq_bins=128, patch_size=16, embed_dim=384):
        super().__init__()
        self.time_frames = time_frames
        self.freq_bins = freq_bins
        self.patch_size = patch_size
        
        self.num_patches_time = time_frames // patch_size
        self.num_patches_freq = freq_bins // patch_size
        self.num_patches = self.num_patches_time * self.num_patches_freq
        
        self.proj = nn.Conv2d(1, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        x: (B, T, F) = (B, 128, 128)
        returns: (B, num_patches, embed_dim)
        """
        x = x.unsqueeze(1)  # (B, 1, T, F)
        x = self.proj(x)     # (B, embed_dim, T//P, F//P)
        x = x.flatten(2)     # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ------------------------------------------------------------
# AudioMAE 1-sec Model
# ------------------------------------------------------------
class AudioMAE1Sec(nn.Module):
    """1초 오디오용 AudioMAE"""
    def __init__(self, cfg: AudioMAE1SecConfig):
        super().__init__()
        self.cfg = cfg
        
        # Patch embedding
        self.patch_embed = PatchEmbed(
            time_frames=cfg.target_time_frames,
            freq_bins=cfg.num_mel_bins,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # CLS token & position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.embed_dim))
        
        # Encoder
        self.blocks = nn.ModuleList([
            Block(cfg.embed_dim, cfg.num_heads, cfg.mlp_ratio, qkv_bias=True)
            for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.embed_dim)
        
        # Decoder (MAE)
        self.decoder_embed = nn.Linear(cfg.embed_dim, cfg.decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, cfg.decoder_embed_dim))
        
        self.decoder_blocks = nn.ModuleList([
            Block(cfg.decoder_embed_dim, cfg.decoder_num_heads, cfg.mlp_ratio, qkv_bias=True)
            for _ in range(cfg.decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(cfg.decoder_embed_dim)
        self.decoder_pred = nn.Linear(cfg.decoder_embed_dim, cfg.patch_size ** 2, bias=True)
        
        # Fine-tuning head
        if cfg.finetune and cfg.num_classes > 0:
            self.head = nn.Linear(cfg.embed_dim, cfg.num_classes)
        else:
            self.head = None
        
        self.initialize_weights()

    def initialize_weights(self):
        # pos embed
        torch.nn.init.normal_(self.pos_embed, std=0.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=0.02)
        
        # cls & mask token
        torch.nn.init.normal_(self.cls_token, std=0.02)
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # linear layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        """
        x: (B, N, D)
        returns: x_masked, mask, ids_restore
        """
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        mask = torch.ones([B, N], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        """Encoder with masking"""
        # Patch embed
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add pos embed (without cls)
        x = x + self.pos_embed[:, 1:, :]
        
        # Masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        """Decoder"""
        # Embed tokens
        x = self.decoder_embed(x)
        
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        
        # Add pos embed
        x = x + self.decoder_pos_embed
        
        # Decoder blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # Predictor
        x = self.decoder_pred(x)
        
        # Remove cls token
        x = x[:, 1:, :]
        
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: (B, T, F)
        pred: (B, num_patches, patch_size^2)
        mask: (B, num_patches), 0=keep, 1=remove
        """
        # Patchify target
        B, T, F = imgs.shape
        p = self.cfg.patch_size
        
        imgs = imgs.unsqueeze(1)  # (B, 1, T, F)
        x = imgs.unfold(2, p, p).unfold(3, p, p)  # (B, 1, num_patches_t, num_patches_f, p, p)
        x = x.contiguous().view(B, -1, p * p)  # (B, num_patches, p^2)
        
        loss = (pred - x) ** 2
        loss = loss.mean(dim=-1)  # (B, num_patches)
        
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def forward(self, x, mask_ratio=None):
        """
        x: (B, 128, 128)
        """
        if mask_ratio is None:
            mask_ratio = self.cfg.mask_ratio
        
        if self.cfg.finetune:
            # Fine-tuning mode: no masking, just encoder
            latent, _, _ = self.forward_encoder(x, mask_ratio=0.0)
            cls_token = latent[:, 0]
            return self.head(cls_token) if self.head is not None else cls_token
        else:
            # Pretraining mode: MAE
            latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)
            loss = self.forward_loss(x, pred, mask)
            return loss, pred, mask


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
def cosine_scheduler(base_value, final_value, epochs, warmup_epochs=0):
    """Cosine learning rate schedule"""
    warmup_schedule = np.linspace(0, base_value, warmup_epochs) if warmup_epochs > 0 else np.array([])
    iters = np.arange(epochs - warmup_epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    schedule = np.concatenate((warmup_schedule, schedule))
    return schedule


def run_epoch_pretrain(model, preproc, loader, optimizer, cfg: AudioMAE1SecConfig, 
                       accelerator: Accelerator, epoch: int, scheduler=None):
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(loader, leave=False, desc=f"Pretrain Epoch {epoch+1}")
    for waveform, _ in pbar:
        waveform = waveform.cpu()
        with torch.no_grad():
            feats = preproc(waveform)  # (B, 128, 128)
        feats = feats.to(accelerator.device, non_blocking=True)
        
        loss, pred, mask = model(feats)
        
        optimizer.zero_grad(set_to_none=True)
        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), cfg.gradient_clip)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / max(1, n_batches)


def run_epoch_finetune(model, preproc, loader, optimizer, cfg: AudioMAE1SecConfig, 
                       accelerator: Accelerator, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    
    pbar = tqdm(loader, leave=False, desc="train" if train else "eval")
    for waveform, label in pbar:
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
        
        acc = (logits.argmax(dim=1) == label).float().mean().item()
        total_loss += loss.item()
        total_acc += acc
        n_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{acc:.4f}"})
    
    return {
        "loss": total_loss / max(1, n_batches),
        "acc": total_acc / max(1, n_batches),
    }


# ------------------------------------------------------------
# Programmatic Entry
# ------------------------------------------------------------
def run_pretrain(
    cfg: Optional[AudioMAE1SecConfig] = None,
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
    
    warnings.filterwarnings("ignore")
    
    # Fixed to pretrain mode only
    mode = "pretrain"

    if accelerator.is_main_process:
        print("\n=== AudioMAE 1-sec Pretraining ===")

    if cfg is None:
        cfg = AudioMAE1SecConfig(finetune=False)

    if datasets is None:
        datasets = ["DINOS"]

    for dataset_name in datasets:
        if dataset_name != "DINOS":
            if accelerator.is_main_process:
                print(f"[WARN] AudioMAE scratch pretraining supports DINOS only. Skipping {dataset_name}.")
            continue

        cfg.dataset = "DINOS"
        if cfg.train_data_path is None:
            cfg.train_data_path = "./Datasets/DINOS_Pretrain"

        set_deterministic_mode(cfg.seed)
        logger = setup_logging(cfg.log_dir)

        if accelerator.is_main_process:
            logger.info(f"Mode: Pretrain (MAE)")
            logger.info(f"Time frames: {cfg.target_time_frames}, Patch size: {cfg.patch_size}")
            logger.info(f"Num patches: {(cfg.target_time_frames // cfg.patch_size) ** 2}")

        # Dataset
        dataset = AudioDataset1Sec(cfg.train_data_path, cfg)

        _nw = min(cfg.num_workers, os.cpu_count() or 1)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=_nw,
            pin_memory=True,
            persistent_workers=_nw > 0,
            worker_init_fn=worker_init_fn,
            drop_last=True,
        )

        # Model
        preproc = FBankPreprocessor1Sec(cfg)
        model = AudioMAE1Sec(cfg)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.wd,
        )

        # Initialize cosine annealing scheduler with minimum learning rate
        eta_min = getattr(cfg, "lr_min", getattr(cfg, "min_lr", 1e-6))
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=eta_min)

        model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

        Path(cfg.checkpoint_dir).mkdir(exist_ok=True, parents=True)

        logger.info("Starting training...")
        for epoch in range(cfg.epochs):
            avg_loss = run_epoch_pretrain(model, preproc, loader, optimizer, cfg, accelerator, epoch, scheduler)

            if accelerator.is_main_process:
                current_lr = optimizer.param_groups[0]['lr']
                logger.info(f"Epoch {epoch+1}/{cfg.epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

                if (epoch + 1) % cfg.save_interval == 0:
                    ckpt_path = Path(cfg.checkpoint_dir) / f"audiomae_Scratch_pretrain_epoch_{epoch+1:04d}.pth"
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": accelerator.unwrap_model(model).state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "config": asdict(cfg),
                    }, ckpt_path)
                    logger.info(f"Saved checkpoint: {ckpt_path}")

            scheduler.step()

        if accelerator.is_main_process:
            logger.info("Training completed!")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    run_pretrain()


if __name__ == "__main__":
    main()
