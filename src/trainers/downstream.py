"""
Unified downstream training module for all models.
"""
from __future__ import annotations

import os
import sys
import random
import time
import logging
import warnings
import glob
import traceback
import hashlib
from pathlib import Path
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, StepLR
import torchaudio
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

from ..unified_config import (
    UnifiedConfig,
    ANOMALY_NORMAL_CLASSES,
    AUDIOMAE_MEAN, AUDIOMAE_STD, AUDIOMAE_TARGET_SR, AUDIOMAE_N_FRAMES, AUDIOMAE_N_MELS,
    PRETRAINED_MODEL_PATHS,
    get_embed_dim,
)
from ..utils import suppress_warnings, set_deterministic_mode

suppress_warnings()


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(log_dir: str, run_name: str) -> logging.Logger:
    """Setup logging to file and console."""
    out = Path(log_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(out / "train.log"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# =============================================================================
# Downstream Classifier
# =============================================================================

class DownstreamClassifier(nn.Module):
    """MLP classifier head for downstream tasks."""

    def __init__(self, embed_dim: int, num_classes: int, hidden_dim: int = 256, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimpleAE(nn.Module):
    """Simple autoencoder for anomaly detection."""

    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, max(256, latent_dim * 4)),
            nn.ReLU(),
            nn.Linear(max(256, latent_dim * 4), latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, max(256, latent_dim * 4)),
            nn.ReLU(),
            nn.Linear(max(256, latent_dim * 4), input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


# =============================================================================
# Feature Extractors
# =============================================================================

def load_feature_extractor(
    model_type: str,
    backbone: Optional[str],
    config: UnifiedConfig,
    device: str,
) -> Tuple[nn.Module, int]:
    """
    Load the feature extractor model.

    Returns:
        Tuple of (model, embed_dim)
    """
    if model_type == "AudioMAE":
        return _load_audiomae_extractor(backbone, config, device)
    elif model_type == "EAT":
        return _load_eat_extractor(backbone, config, device)
    elif model_type == "VGGish":
        return _load_vggish_extractor(device)
    elif model_type == "CLAP":
        return _load_clap_extractor(config, device)
    elif model_type == "openSMILE":
        return _load_opensmile_extractor(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _safe_torch_load(path: Path | str, map_location: str = "cpu") -> Any:
    path = Path(path)
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading .safetensors checkpoints requires the `safetensors` package."
            ) from exc
        return load_file(str(path))

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Older PyTorch versions do not support weights_only.
        return torch.load(path, map_location=map_location)


def _find_latest_checkpoint(search_dirs: List[Path], patterns: List[str]) -> Optional[Path]:
    candidates: List[Path] = []
    for d in search_dirs:
        if not d.exists():
            continue
        for pattern in patterns:
            candidates.extend([p for p in d.glob(pattern) if p.is_file()])

    if not candidates:
        return None

    # Prefer explicit "best" checkpoints when present.
    for p in candidates:
        if re.search(r"best", p.name, re.IGNORECASE):
            return p

    # Fall back to most recently modified.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _resolve_audiomae_finetuned_checkpoint(config: UnifiedConfig) -> Optional[Path]:
    # Use user-selected model if available
    if config.backbone_model_path:
        user_path = Path(config.backbone_model_path)
        if user_path.exists():
            return user_path
        logging.warning(f"User-selected model not found: {config.backbone_model_path}")

    # Otherwise, search for checkpoints automatically
    search_dirs: List[Path] = []
    for raw in [getattr(config, "checkpoint_dir", None), getattr(config, "save_dir", None), "./AudioMAE_Models_FT"]:
        if not raw:
            continue
        path = Path(raw)
        if path not in search_dirs:
            search_dirs.append(path)

    patterns = [
        "AudioMAE_DINOS_best.pth",
        "audiomae_randompos_*_epoch_*.pth",
        "audiomae_randompos_*_best*.pth",
    ]
    return _find_latest_checkpoint(search_dirs, patterns)


def _load_audiomae_extractor(
    backbone: Optional[str],
    config: UnifiedConfig,
    device: str,
) -> Tuple[nn.Module, int]:
    """Load AudioMAE feature extractor."""
    import timm

    tag = "audiomae"
    if backbone == "pretrained" or backbone is None:
        model_id = PRETRAINED_MODEL_PATHS["AudioMAE"]
        model_name = model_id if model_id.startswith("hf_hub:") else f"hf_hub:{model_id}"
        model = timm.create_model(model_name, pretrained=True)
        embed_dim = 768
        tag = model_id
    elif backbone == "finetuned":
        model_id = PRETRAINED_MODEL_PATHS["AudioMAE"]
        model_name = model_id if model_id.startswith("hf_hub:") else f"hf_hub:{model_id}"
        model = timm.create_model(model_name, pretrained=False)
        checkpoint_path = _resolve_audiomae_finetuned_checkpoint(config)
        if checkpoint_path and checkpoint_path.exists():
            ckpt = _safe_torch_load(checkpoint_path)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            tag = str(checkpoint_path)
            logging.info(f"Loaded AudioMAE finetuned checkpoint: {checkpoint_path}")
        else:
            logging.warning("No AudioMAE finetuned checkpoint found. Using random init weights.")
            tag = "finetuned"
        embed_dim = 768
    elif backbone == "scratch":
        # Load AudioMAE 1-sec scratch model
        from ..pipelines.AudioMAE_scratch import AudioMAE1SecConfig, AudioMAE1Sec
        scratch_cfg = AudioMAE1SecConfig()
        scratch_cfg.sr = config.sr
        scratch_cfg.sample_len = config.sample_len
        scratch_cfg.target_sr = config.sr
        scratch_cfg.num_mel_bins = config.n_mels
        scratch_cfg.embed_dim = config.embed_dim
        scratch_cfg.depth = config.depth
        scratch_cfg.num_heads = config.nhead
        scratch_cfg.decoder_embed_dim = config.decoder_dim
        if isinstance(config.patch_size, (tuple, list)):
            scratch_cfg.patch_size = config.patch_size[0]
        else:
            scratch_cfg.patch_size = int(config.patch_size)

        model = AudioMAE1Sec(scratch_cfg)

        # Use user-selected model if available
        if config.backbone_model_path:
            checkpoint_path = Path(config.backbone_model_path)
            if checkpoint_path.exists():
                ckpt = _safe_torch_load(checkpoint_path)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict, strict=False)
                tag = str(checkpoint_path)
                logging.info(f"Loaded user-selected AudioMAE scratch model: {checkpoint_path}")
            else:
                logging.warning(f"User-selected model not found: {config.backbone_model_path}")
                tag = "scratch"
        else:
            # Otherwise, search for checkpoints automatically
            ckpt_dirs = [Path("./AudioMAE_Models_Scratch"), Path(config.save_dir)]
            ckpt_files = []
            for d in ckpt_dirs:
                if d.exists():
                    ckpt_files.extend(sorted(d.glob("audiomae_Scratch_pretrain_epoch_*.pth")))

            if ckpt_files:
                checkpoint_path = ckpt_files[-1]
                ckpt = _safe_torch_load(checkpoint_path)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict, strict=False)
                tag = str(checkpoint_path)
            else:
                tag = "scratch"

        embed_dim = scratch_cfg.embed_dim
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    cache_root = Path(config.save_dir) / "features" / "audiomae"
    model._audiomae_cache = AudioMAEEmbeddingCache(cache_dir=str(cache_root))
    if backbone == "scratch":
        model._audiomae_cache_params = {
            "sr": config.sr,
            "frames": getattr(scratch_cfg, "target_time_frames", 128),
            "mels": config.n_mels,
            "mean": config.dataset_mean,
            "std": config.dataset_std,
            "tag": tag,
        }
    else:
        model._audiomae_cache_params = {
            "sr": AUDIOMAE_TARGET_SR,
            "frames": AUDIOMAE_N_FRAMES,
            "mels": AUDIOMAE_N_MELS,
            "mean": AUDIOMAE_MEAN,
            "std": AUDIOMAE_STD,
            "tag": tag,
        }

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, embed_dim


def _load_eat_extractor(
    backbone: Optional[str],
    config: UnifiedConfig,
    device: str,
) -> Tuple[nn.Module, int]:
    """Load EAT feature extractor."""
    from ..utils import EATMAE

    logger = logging.getLogger(__name__)
    if backbone == "pretrained" or backbone is None:
        # Load from HuggingFace
        from huggingface_hub import hf_hub_download
        model_id = PRETRAINED_MODEL_PATHS["EAT"]
        repo_id = model_id.replace("https://huggingface.co/", "")
        # Create model with pretrained config
        pretrain_config = UnifiedConfig(
            embed_dim=768, depth=12, nhead=12, decoder_dim=512
        )
        model = EATMAE(pretrain_config)
        # Download and load weights
        try:
            checkpoint_path = None
            last_download_error: Optional[Exception] = None
            for filename in ("model.safetensors", "pytorch_model.bin"):
                try:
                    checkpoint_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                    )
                    logger.info(f"Resolved EAT checkpoint file: {filename}")
                    break
                except Exception as exc:
                    last_download_error = exc

            if checkpoint_path is None:
                raise RuntimeError(
                    f"No supported checkpoint file found in {repo_id}. "
                    "Tried: model.safetensors, pytorch_model.bin"
                ) from last_download_error

            state_dict = _safe_torch_load(checkpoint_path)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded EAT pretrained checkpoint: {checkpoint_path}")
        except Exception as e:
            logging.warning(f"Could not load pretrained EAT: {e}")
        embed_dim = 768
    elif backbone == "finetuned":
        pretrain_config = UnifiedConfig(embed_dim=768, depth=12, nhead=12, decoder_dim=512)
        model = EATMAE(pretrain_config)

        # Use user-selected model if available
        if config.backbone_model_path:
            checkpoint_path = Path(config.backbone_model_path)
            if checkpoint_path.exists():
                ckpt = _safe_torch_load(checkpoint_path)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded user-selected EAT finetuned model: {checkpoint_path}")
            else:
                logger.warning(f"User-selected model not found: {config.backbone_model_path}")
        else:
            # Otherwise, search for checkpoints automatically
            checkpoint_path = Path("./EAT_Models_FT/EAT_DINOS_finetuned_final.pth")
            fallback_path = Path("./EAT_Models_FT/EAT_DINOS_best.pth")
            if checkpoint_path.exists():
                ckpt = _safe_torch_load(checkpoint_path)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded EAT finetuned checkpoint: {checkpoint_path}")
            elif fallback_path.exists():
                ckpt = _safe_torch_load(fallback_path)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded EAT finetuned fallback checkpoint: {fallback_path}")
            else:
                logger.warning("No EAT finetuned checkpoint found. Using random init weights.")
        embed_dim = 768
    elif backbone == "scratch":
        model = EATMAE(config)

        # Use user-selected model if available
        if config.backbone_model_path:
            checkpoint_path = Path(config.backbone_model_path)
            if checkpoint_path.exists():
                ckpt = _safe_torch_load(checkpoint_path)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded user-selected EAT scratch model: {checkpoint_path}")
            else:
                logger.warning(f"User-selected model not found: {config.backbone_model_path}")
        else:
            # Otherwise, search for checkpoints automatically
            checkpoint_path = Path("./EAT_Models_Scratch/EAT_DINOS_epoch_0020.pth")
            if checkpoint_path.exists():
                ckpt = _safe_torch_load(checkpoint_path)
                state_dict = ckpt.get("model_state_dict", ckpt)
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded EAT scratch checkpoint: {checkpoint_path}")
            else:
                logger.warning("No EAT scratch checkpoint found. Using random init weights.")
        embed_dim = config.embed_dim
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, embed_dim


def _fix_vggish_device(model: nn.Module, device: str) -> nn.Module:
    """Fix VGGish internal state after .to(device).

    The torchvggish VGGish model caches ``self.device`` at init time and uses
    it in ``forward()`` to move inputs (``x = x.to(self.device)``).  A later
    ``.to(new_device)`` moves parameters but does NOT update that attribute,
    causing a device mismatch.  We also disable pre/post-processing since we
    handle that externally in ``_extract_vggish_features``.
    """
    # Disable internal pre/post-processing (we do it externally)
    if hasattr(model, "preprocess"):
        model.preprocess = False
    if hasattr(model, "postprocess"):
        model.postprocess = False
    if hasattr(model, "pproc"):
        model.pproc = None
    # Sync the cached device attribute with the actual device
    if hasattr(model, "device"):
        model.device = torch.device(device)
    return model


def _load_vggish_extractor(device: str) -> Tuple[nn.Module, int]:
    """Load VGGish feature extractor."""
    try:
        import torchvggish

        model = None
        if hasattr(torchvggish, "vggish"):
            fn = torchvggish.vggish
            try:
                model = fn(pretrained=True, postprocess=False)
            except TypeError:
                try:
                    model = fn(pretrained=True)
                except TypeError:
                    model = fn()
        elif hasattr(torchvggish, "VGGish"):
            cls = torchvggish.VGGish
            try:
                model = cls(pretrained=True, postprocess=False)
            except TypeError:
                try:
                    model = cls(pretrained=True)
                except TypeError:
                    model = cls()

        if model is None:
            raise AttributeError("No vggish constructor found")

        model.to(device)
        _fix_vggish_device(model, device)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        return model, 128

    except Exception as e:
        # Last resort: torch.hub fallback
        try:
            model = torch.hub.load(
                "harritaylor/torchvggish", "vggish", pretrained=True, preprocess=False
            )
            model.to(device)
            _fix_vggish_device(model, device)
            model.eval()
            for p in model.parameters():
                p.requires_grad = False
            return model, 128
        except Exception as e_hub:
            raise RuntimeError(
                "Failed to load VGGish via PyPI torchvggish and torch.hub.\n"
                f"[PyPI torchvggish error] {e}\n"
                f"[torch.hub error] {e_hub}"
            )


def _load_clap_extractor(config: UnifiedConfig, device: str) -> Tuple[nn.Module, int]:
    """Load CLAP feature extractor."""
    from transformers import ClapModel, ClapProcessor

    model_name = "laion/clap-htsat-unfused"
    model = ClapModel.from_pretrained(model_name)
    # Cache processor to avoid repeated downloads in feature extraction.
    model._clap_processor = ClapProcessor.from_pretrained(model_name)
    model._clap_model_name = model_name
    cache_root = Path(config.save_dir) / "features" / "clap"
    model._clap_cache = ClapEmbeddingCache(cache_dir=str(cache_root))
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model, 512


def _stable_id_for_path(path: str) -> str:
    return hashlib.md5(path.encode("utf-8")).hexdigest()


class ClapEmbeddingCache:
    """Simple on-disk cache for CLAP embeddings."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, key: str) -> Optional[np.ndarray]:
        path = self.cache_dir / f"{key}.npy"
        if path.exists():
            return np.load(path)
        return None

    def save(self, key: str, arr: np.ndarray) -> None:
        path = self.cache_dir / f"{key}.npy"
        np.save(path, arr.astype(np.float32))


class AudioMAEEmbeddingCache:
    """Simple on-disk cache for AudioMAE CLS embeddings."""

    def __init__(self, cache_dir: str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, key: str) -> Optional[np.ndarray]:
        path = self.cache_dir / f"{key}.npy"
        if path.exists():
            return np.load(path)
        return None

    def save(self, key: str, arr: np.ndarray) -> None:
        path = self.cache_dir / f"{key}.npy"
        np.save(path, arr.astype(np.float32))


class OpenSMILEExtractor:
    """openSMILE extractor with optional on-disk caching."""

    def __init__(
        self,
        cache_dir: str,
        feature_set: str,
        target_sr: int,
    ):
        import opensmile

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.target_sr = target_sr
        self.feature_set = feature_set

        feature_set_key = feature_set.lower()
        if feature_set_key in ["egemapsv02", "egemaps", "egemaps_v02"]:
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.eGeMAPSv02,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        elif feature_set_key in ["compare_2016", "compare2016", "compare"]:
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet.ComParE_2016,
                feature_level=opensmile.FeatureLevel.Functionals,
            )
        else:
            raise ValueError(f"Unknown openSMILE feature_set: {feature_set}")

    def extract(
        self,
        wav_1d: torch.Tensor,
        sr: int,
        cache_key: Optional[str] = None,
    ) -> np.ndarray:
        if cache_key is not None:
            npy_path = self.cache_dir / f"{cache_key}.npy"
            if npy_path.exists():
                return np.load(npy_path)

        x = wav_1d.detach().cpu().numpy().astype(np.float32)
        df = self.smile.process_signal(x, sr)
        feat = df.iloc[0].to_numpy(dtype=np.float32)

        if cache_key is not None:
            np.save(npy_path, feat)

        return feat


def _load_opensmile_extractor(config: UnifiedConfig) -> Tuple[Any, int]:
    """Load openSMILE feature extractor."""
    feature_set = getattr(config, "openSMILE_feature_set", "egemapsv02")
    cache_dir = Path(config.save_dir) / "features" / "openSMILE"
    extractor = OpenSMILEExtractor(
        cache_dir=str(cache_dir),
        feature_set=feature_set,
        target_sr=config.sr,
    )
    return extractor, 88
    import opensmile

    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    return smile, 88


# =============================================================================
# Datasets
# =============================================================================

class DownstreamDataset(Dataset):
    """Generic downstream dataset for audio classification."""

    def __init__(
        self,
        root_dir: str,
        downstream: str,
        model_type: str,
        config: UnifiedConfig,
        is_train: bool = True,
        max_retries: int = 3,
    ):
        super().__init__()
        self.config = config
        self.model_type = model_type
        self.is_train = is_train
        self.max_retries = max_retries
        self.error_count = defaultdict(int)

        self.file_list: List[Tuple[str, int, str]] = []
        self.class_to_idx: Dict[str, int] = {}

        downstream_path = Path(root_dir) / downstream
        if not downstream_path.exists():
            raise FileNotFoundError(f"Downstream directory not found: {downstream_path}")

        class_names = [d.name for d in sorted(downstream_path.iterdir()) if d.is_dir()]
        self.class_to_idx = {name: idx for idx, name in enumerate(sorted(class_names))}

        for class_name in class_names:
            class_path = downstream_path / class_name
            audio_files = []
            for ext in ["*.wav", "*.flac", "*.mp3", "*.ogg"]:
                audio_files.extend(glob.glob(str(class_path / ext)))
            for fp in audio_files:
                self.file_list.append((fp, self.class_to_idx[class_name], class_name))

        if config.seed is not None:
            random.seed(config.seed)
            random.shuffle(self.file_list)

        # Calculate target lengths based on model
        if model_type == "AudioMAE":
            if config.backbone == "scratch":
                self.target_sr = config.sr
                self.fixed_len = config.sample_len
            else:
                self.target_sr = AUDIOMAE_TARGET_SR
                self.fixed_len = int(config.sample_len / config.sr * self.target_sr)
        elif model_type == "VGGish":
            self.target_sr = 16_000
            duration_sec = float(config.sample_len) / float(config.sr)
            self.fixed_len = int(round(duration_sec * self.target_sr))
        elif model_type == "CLAP":
            self.target_sr = 48_000
            self.fixed_len = int(10 * self.target_sr)
        else:
            self.target_sr = config.sr
            self.fixed_len = config.sample_len

        logging.info(f"Dataset: {len(self.file_list)} samples, {len(class_names)} classes")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        for retry in range(self.max_retries):
            fp = None
            try:
                fp, label, cname = self.file_list[idx]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wav, sr = torchaudio.load(fp)

                # mono
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                wav = wav.squeeze(0)

                # resample
                if sr != self.target_sr:
                    wav = torchaudio.functional.resample(wav, sr, self.target_sr)

                # crop/pad
                T = wav.shape[0]
                if T > self.fixed_len:
                    if self.is_train:
                        start = random.randint(0, T - self.fixed_len)
                    else:
                        start = 0
                    wav = wav[start:start + self.fixed_len]
                elif T < self.fixed_len:
                    wav = F.pad(wav, (0, self.fixed_len - T))

                return wav, int(label), cname, fp

            except Exception as e:
                if fp is not None:
                    self.error_count[fp] += 1
                logging.error(f"Error loading {fp} (attempt {retry+1}/{self.max_retries}): {e}")
                if retry == self.max_retries - 1:
                    return self.__getitem__((idx + 1) % len(self))
                time.sleep(0.1 * (retry + 1))


class AnomalyDataset(Dataset):
    """Dataset for anomaly detection (normal samples only)."""

    def __init__(
        self,
        root_dir: str,
        downstream: str,
        model_type: str,
        config: UnifiedConfig,
        is_train: bool = True,
    ):
        super().__init__()
        self.config = config
        self.model_type = model_type
        self.is_train = is_train

        # Get base downstream name (e.g., "Yornew" from "Yornew_anomaly")
        base_downstream = downstream.replace("_anomaly", "")
        normal_classes = ANOMALY_NORMAL_CLASSES.get(base_downstream, [])

        if not normal_classes:
            raise ValueError(f"No normal classes defined for {base_downstream}")

        self.file_list: List[str] = []
        downstream_path = Path(root_dir) / base_downstream

        for class_name in normal_classes:
            class_path = downstream_path / class_name
            if class_path.exists():
                for ext in ["*.wav", "*.flac", "*.mp3", "*.ogg"]:
                    self.file_list.extend(glob.glob(str(class_path / ext)))

        if config.seed is not None:
            random.seed(config.seed)
            random.shuffle(self.file_list)

        # Target sample rate and length
        if model_type == "AudioMAE":
            if config.backbone == "scratch":
                self.target_sr = config.sr
                self.fixed_len = config.sample_len
            else:
                self.target_sr = AUDIOMAE_TARGET_SR
                self.fixed_len = int(config.sample_len / config.sr * self.target_sr)
        elif model_type == "VGGish":
            self.target_sr = 16_000
            duration_sec = float(config.sample_len) / float(config.sr)
            self.fixed_len = int(round(duration_sec * self.target_sr))
        elif model_type == "CLAP":
            self.target_sr = 48_000
            self.fixed_len = int(10 * self.target_sr)
        else:
            self.target_sr = config.sr
            self.fixed_len = config.sample_len

        logging.info(f"Anomaly dataset: {len(self.file_list)} normal samples")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fp = self.file_list[idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wav, sr = torchaudio.load(fp)

            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            wav = wav.squeeze(0)

            if sr != self.target_sr:
                wav = torchaudio.functional.resample(wav, sr, self.target_sr)

            T = wav.shape[0]
            if T > self.fixed_len:
                if self.is_train:
                    start = random.randint(0, T - self.fixed_len)
                else:
                    start = 0
                wav = wav[start:start + self.fixed_len]
            elif T < self.fixed_len:
                wav = F.pad(wav, (0, self.fixed_len - T))

            return wav, fp

        except Exception as e:
            logging.error(f"Error loading {fp}: {e}")
            return self.__getitem__((idx + 1) % len(self))


# =============================================================================
# Preprocessing Functions
# =============================================================================

def preprocess_batch(
    batch_wav: torch.Tensor,
    model_type: str,
    config: UnifiedConfig,
    device: str,
) -> torch.Tensor:
    """Preprocess waveforms for feature extraction."""
    if model_type == "AudioMAE":
        if config.backbone == "scratch":
            return _preprocess_audiomae_scratch(batch_wav, config, device)
        return _preprocess_audiomae(batch_wav, device)
    elif model_type == "EAT":
        return _preprocess_eat(batch_wav, config, device)
    elif model_type == "VGGish":
        # VGGish expects CPU waveform.
        return batch_wav.cpu()
    elif model_type == "CLAP":
        # CLAP processor expects CPU numpy/list.
        return batch_wav.cpu()
    elif model_type == "openSMILE":
        # openSMILE runs on CPU; avoid unnecessary GPU transfers.
        return batch_wav.cpu()
    else:
        return batch_wav.to(device)


def _preprocess_audiomae(batch_wav: torch.Tensor, device: str) -> torch.Tensor:
    """Preprocess for AudioMAE (mel-spectrogram)."""
    from torchaudio.compliance import kaldi

    B = batch_wav.shape[0]
    fbanks = []

    for i in range(B):
        x = batch_wav[i]
        x = x - x.mean()

        melspec = kaldi.fbank(
            x.unsqueeze(0),
            htk_compat=True,
            window_type="hanning",
            num_mel_bins=AUDIOMAE_N_MELS,
        )

        if melspec.shape[0] < AUDIOMAE_N_FRAMES:
            melspec = F.pad(melspec, (0, 0, 0, AUDIOMAE_N_FRAMES - melspec.shape[0]))
        else:
            melspec = melspec[:AUDIOMAE_N_FRAMES]

        melspec = (melspec - AUDIOMAE_MEAN) / (AUDIOMAE_STD * 2.0)
        fbanks.append(melspec)

    x = torch.stack(fbanks, dim=0).unsqueeze(1)  # (B, 1, 1024, 128)
    return x.to(device)


def _preprocess_audiomae_scratch(
    batch_wav: torch.Tensor,
    config: UnifiedConfig,
    device: str,
) -> torch.Tensor:
    """Preprocess for AudioMAE scratch (1-sec) model."""
    from ..pipelines.AudioMAE_scratch import AudioMAE1SecConfig, FBankPreprocessor1Sec

    if not hasattr(config, "_audiomae_scratch_preproc"):
        scratch_cfg = AudioMAE1SecConfig()
        scratch_cfg.sr = config.sr
        scratch_cfg.sample_len = config.sample_len
        scratch_cfg.target_sr = config.sr
        scratch_cfg.num_mel_bins = config.n_mels
        scratch_cfg.embed_dim = config.embed_dim
        scratch_cfg.depth = config.depth
        scratch_cfg.num_heads = config.nhead
        scratch_cfg.decoder_embed_dim = config.decoder_dim
        if isinstance(config.patch_size, (tuple, list)):
            scratch_cfg.patch_size = config.patch_size[0]
        else:
            scratch_cfg.patch_size = int(config.patch_size)
        config._audiomae_scratch_preproc = FBankPreprocessor1Sec(scratch_cfg)

    fb = config._audiomae_scratch_preproc(batch_wav.cpu())
    return fb.to(device)
def _preprocess_eat(batch_wav: torch.Tensor, config: UnifiedConfig, device: str) -> torch.Tensor:
    """Preprocess for EAT (mel-spectrogram)."""
    from ..utils import SpecConverter

    spec_converter = SpecConverter(config).to(device)
    wav = batch_wav.to(device)
    spec = spec_converter(wav)  # (B, 1, F, T)
    return spec


# =============================================================================
# Feature Extraction
# =============================================================================

@torch.no_grad()
def extract_features(
    model: nn.Module,
    batch: torch.Tensor,
    model_type: str,
    device: str,
    file_paths: Optional[List[str]] = None,
    clap_truncation: Optional[str] = None,
    clap_padding: Optional[str] = None,
) -> torch.Tensor:
    """Extract features from batch."""
    if model_type == "AudioMAE":
        cache = getattr(model, "_audiomae_cache", None)
        tag = getattr(model, "_audiomae_tag", "audiomae")
        if cache is None or file_paths is None:
            return _extract_audiomae_features(model, batch)

        params = getattr(model, "_audiomae_cache_params", None)
        if params is None:
            params = {
                "sr": AUDIOMAE_TARGET_SR,
                "frames": AUDIOMAE_N_FRAMES,
                "mels": AUDIOMAE_N_MELS,
                "mean": AUDIOMAE_MEAN,
                "std": AUDIOMAE_STD,
                "tag": tag,
            }

        keys = [
            _stable_id_for_path(
                f"{fp}|sr={params['sr']}|frames={params['frames']}|"
                f"mels={params['mels']}|mean={params['mean']}|std={params['std']}|tag={params['tag']}"
            )
            for fp in file_paths
        ]

        emb_list: List[Optional[torch.Tensor]] = []
        missing_idx: List[int] = []
        for i, k in enumerate(keys):
            cached = cache.load(k)
            if cached is None:
                emb_list.append(None)
                missing_idx.append(i)
            else:
                emb_list.append(torch.from_numpy(cached).float())

        if missing_idx:
            x_missing = batch[missing_idx]
            emb_missing = _extract_audiomae_features(model, x_missing).detach().cpu()
            for j, i in enumerate(missing_idx):
                emb_list[i] = emb_missing[j]
                cache.save(keys[i], emb_missing[j].numpy())

        return torch.stack([e for e in emb_list], dim=0).to(device)
    elif model_type == "EAT":
        return _extract_eat_features(model, batch)
    elif model_type == "VGGish":
        return _extract_vggish_features(model, batch, device)
    elif model_type == "CLAP":
        return _extract_clap_features(
            model,
            batch,
            device,
            file_paths=file_paths,
            truncation=clap_truncation,
            padding=clap_padding,
        )
    elif model_type == "openSMILE":
        return _extract_opensmile_features(model, batch, device, file_paths)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _extract_audiomae_features(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Extract AudioMAE CLS token features."""
    if hasattr(model, "forward_features"):
        feats = model.forward_features(batch)
        if isinstance(feats, dict):
            feats = feats.get("x", feats.get("last_hidden_state"))
        if feats.dim() == 3:
            return feats[:, 0]  # CLS token
        return feats

    if hasattr(model, "forward_encoder"):
        if batch.dim() == 4:
            batch = batch.squeeze(1)
        feats, _, _ = model.forward_encoder(batch, mask_ratio=0.0)
        return feats[:, 0]

    raise RuntimeError("AudioMAE model does not support feature extraction.")


def _extract_eat_features(model: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Extract EAT features."""
    feats = model.forward_features(batch)
    if feats.dim() == 3:
        return feats.mean(dim=1)  # Mean pool
    return feats


def _extract_vggish_features(model: nn.Module, batch: torch.Tensor, device: str) -> torch.Tensor:
    """Extract VGGish features."""
    from torchvggish import vggish_input

    # Use model's actual device to avoid mismatch when device str differs
    model_device = next(model.parameters()).device

    if batch.dim() != 2:
        raise ValueError(f"Expected waveforms shape (B,T), got {tuple(batch.shape)}")

    B = batch.shape[0]
    ex_list: List[torch.Tensor] = []
    counts: List[int] = []

    for i in range(B):
        wav = batch[i].detach().cpu().numpy()
        examples = vggish_input.waveform_to_examples(wav, 16000)
        if isinstance(examples, np.ndarray):
            ex = torch.from_numpy(examples)
        elif torch.is_tensor(examples):
            ex = examples.detach().cpu()
        else:
            ex = torch.as_tensor(examples)
        ex = ex.float()
        ex_list.append(ex)
        counts.append(ex.shape[0])

    if sum(counts) == 0:
        return torch.zeros((B, 128), dtype=torch.float32, device=model_device)

    ex_all = torch.cat(ex_list, dim=0).to(model_device, non_blocking=True)
    emb_all = model(ex_all)

    out: List[torch.Tensor] = []
    offset = 0
    for n in counts:
        out.append(emb_all[offset:offset + n].mean(dim=0, keepdim=True))
        offset += n

    return torch.cat(out, dim=0)


def _extract_clap_features(
    model: nn.Module,
    batch: torch.Tensor,
    device: str,
    file_paths: Optional[List[str]] = None,
    truncation: Optional[str] = None,
    padding: Optional[str] = None,
) -> torch.Tensor:
    """Extract CLAP audio features."""
    processor = getattr(model, "_clap_processor", None)
    if processor is None:
        from transformers import ClapProcessor
        processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        model._clap_processor = processor
    cache = getattr(model, "_clap_cache", None)
    model_name = getattr(model, "_clap_model_name", "laion/clap-htsat-unfused")
    truncation = truncation or "fusion"
    padding = padding or "max_length"

    # Build cache keys
    keys = None
    if file_paths is not None:
        keys = [
            _stable_id_for_path(
                f"{fp}|sr=48000|len={batch.shape[-1]}|model={model_name}|trunc={truncation}"
            )
            for fp in file_paths
        ]

    emb_list: List[Optional[torch.Tensor]] = []
    missing_idx: List[int] = []
    for i in range(batch.shape[0]):
        if cache is not None and keys is not None:
            cached = cache.load(keys[i])
            if cached is not None:
                emb_list.append(torch.from_numpy(cached).float())
                continue
        emb_list.append(None)
        missing_idx.append(i)

    if missing_idx:
        audio_list = []
        for i in missing_idx:
            w = batch[i]
            w_np = w.detach().cpu().numpy()
            w_np = np.squeeze(w_np)
            if w_np.ndim != 1:
                w_np = w_np.reshape(-1)
            audio_list.append(w_np)

        try:
            inputs = processor(
                audio=audio_list,
                sampling_rate=48000,
                return_tensors="pt",
                padding=padding,
                truncation=truncation,
            )
        except TypeError:
            inputs = processor(
                audios=audio_list,
                sampling_rate=48000,
                return_tensors="pt",
                padding=padding,
                truncation=truncation,
            )

        if "input_features" in inputs:
            feats = inputs["input_features"]
            if feats.dim() == 4 and feats.shape[1] != 1:
                inputs["input_features"] = feats.mean(dim=1, keepdim=True)

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.get_audio_features(**inputs)
            if isinstance(outputs, torch.Tensor):
                emb = outputs
            elif isinstance(outputs, (tuple, list)) and len(outputs) > 0:
                emb = outputs[0]
            else:
                emb = getattr(outputs, "pooler_output", None)
                if emb is None:
                    emb = getattr(outputs, "last_hidden_state", None)
                    if emb is not None and emb.dim() == 3:
                        emb = emb[:, 0]
            if emb is None:
                raise RuntimeError("CLAP audio features not found in model output.")

            # L2 normalize (CLIP-style)
            emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-12)
            emb_cpu = emb.detach().cpu()

        for j, i in enumerate(missing_idx):
            emb_list[i] = emb_cpu[j]
            if cache is not None and keys is not None:
                cache.save(keys[i], emb_cpu[j].numpy())

    return torch.stack([e for e in emb_list], dim=0).to(device)


def _extract_opensmile_features(
    extractor: OpenSMILEExtractor,
    batch: torch.Tensor,
    device: str,
    file_paths: Optional[List[str]] = None,
) -> torch.Tensor:
    """Extract openSMILE features."""
    embeddings = []
    for i in range(batch.shape[0]):
        wav = batch[i].cpu()
        cache_key = None
        if file_paths is not None and i < len(file_paths):
            cache_key = _stable_id_for_path(
                f"{file_paths[i]}|sr={extractor.target_sr}|len={wav.shape[-1]}"
            )
        feat_np = extractor.extract(wav, extractor.target_sr, cache_key=cache_key)
        embeddings.append(torch.from_numpy(feat_np.copy()).float())

    features = torch.stack(embeddings)
    if device != "cpu":
        features = features.to(device)
    return features


# =============================================================================
# Scheduler Factory
# =============================================================================

def get_scheduler(optimizer, config: UnifiedConfig):
    """Create learning rate scheduler."""
    if config.scheduler_type == "cosine":
        return CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.lr_min
        )
    elif config.scheduler_type == "plateau":
        return ReduceLROnPlateau(
            optimizer, mode="min", factor=config.lr_decay_factor,
            patience=config.lr_decay_patience, min_lr=config.lr_min
        )
    elif config.scheduler_type == "step":
        return StepLR(optimizer, step_size=30, gamma=config.lr_decay_factor)
    return None


# =============================================================================
# Training Functions
# =============================================================================

def train_classification(
    downstream: str,
    config: UnifiedConfig,
    feature_extractor: nn.Module,
    embed_dim: int,
    accelerator: Accelerator,
) -> Dict[str, Any]:
    """Train classification downstream task."""
    logger = logging.getLogger(__name__)
    device = accelerator.device

    # Create dataset and dataloader
    dataset = DownstreamDataset(
        root_dir=config.train_data_path,
        downstream=downstream,
        model_type=config.model_type,
        config=config,
        is_train=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    num_classes = len(dataset.class_to_idx)

    # Create classifier
    classifier = DownstreamClassifier(
        embed_dim=embed_dim,
        num_classes=num_classes,
        hidden_dim=config.classifier_hidden_dim,
        dropout=config.classifier_dropout,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=config.lr,
        weight_decay=config.wd,
    )
    scheduler = get_scheduler(optimizer, config)
    criterion = nn.CrossEntropyLoss()

    # Prepare with accelerator
    classifier, optimizer, dataloader = accelerator.prepare(
        classifier, optimizer, dataloader
    )
    if hasattr(feature_extractor, "to"):
        feature_extractor = feature_extractor.to(device)

    # Training loop
    best_loss = float("inf")
    best_acc = 0.0
    patience_counter = 0

    save_dir = Path(config.save_dir) / downstream
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        clap_truncation = getattr(config, "clap_truncation_train", "fusion")
        clap_padding = getattr(config, "clap_padding", "max_length")

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_idx, (wav, labels, _, file_paths) in enumerate(pbar):
            # Preprocess and extract features
            preprocessed = preprocess_batch(wav, config.model_type, config, device)

            with torch.no_grad():
                features = extract_features(
                    feature_extractor,
                    preprocessed,
                    config.model_type,
                    device,
                    file_paths=file_paths,
                    clap_truncation=clap_truncation,
                    clap_padding=clap_padding,
                )

            # Forward pass
            logits = classifier(features)
            labels = labels.to(device)
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            accelerator.backward(loss)
            if config.gradient_clip > 0:
                accelerator.clip_grad_norm_(classifier.parameters(), config.gradient_clip)
            optimizer.step()

            # Metrics
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "acc": f"{100. * correct / total:.2f}%",
            })

        # Epoch metrics
        avg_loss = total_loss / len(dataloader)
        accuracy = 100. * correct / total

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        logger.info(f"Epoch {epoch+1}: loss={avg_loss:.4f}, acc={accuracy:.2f}%")

        # Save best model
        if accelerator.is_main_process:
            if avg_loss < best_loss:
                best_loss = avg_loss
                best_acc = accuracy
                patience_counter = 0
                torch.save(
                    accelerator.unwrap_model(classifier).state_dict(),
                    save_dir / f"{config.model_type}_{downstream}_best.pth",
                )
            else:
                patience_counter += 1

        # Early stopping
        if patience_counter >= config.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model
    if accelerator.is_main_process:
        torch.save(
            accelerator.unwrap_model(classifier).state_dict(),
            save_dir / f"{config.model_type}_{downstream}_final.pth",
        )

    return {
        "downstream": downstream,
        "best_loss": best_loss,
        "best_acc": best_acc,
        "epochs_trained": epoch + 1,
    }


def train_anomaly(
    downstream: str,
    config: UnifiedConfig,
    feature_extractor: nn.Module,
    embed_dim: int,
    accelerator: Accelerator,
) -> Dict[str, Any]:
    """Train anomaly detection downstream task."""
    logger = logging.getLogger(__name__)
    device = accelerator.device

    # Create dataset (normal samples only)
    dataset = AnomalyDataset(
        root_dir=config.train_data_path,
        downstream=downstream,
        model_type=config.model_type,
        config=config,
        is_train=True,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Create autoencoder
    ae_model = SimpleAE(input_dim=embed_dim, latent_dim=64)

    # Optimizer
    optimizer = torch.optim.AdamW(
        ae_model.parameters(),
        lr=config.lr,
        weight_decay=config.wd,
    )
    scheduler = get_scheduler(optimizer, config)

    # Prepare with accelerator
    ae_model, optimizer, dataloader = accelerator.prepare(
        ae_model, optimizer, dataloader
    )
    if hasattr(feature_extractor, "to"):
        feature_extractor = feature_extractor.to(device)

    # Training loop
    best_loss = float("inf")
    patience_counter = 0

    save_dir = Path(config.save_dir) / downstream
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        ae_model.train()
        total_loss = 0.0

        clap_truncation = getattr(config, "clap_truncation_train", "fusion")
        clap_padding = getattr(config, "clap_padding", "max_length")

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        for batch_idx, (wav, file_paths) in enumerate(pbar):
            # Preprocess and extract features
            preprocessed = preprocess_batch(wav, config.model_type, config, device)

            with torch.no_grad():
                features = extract_features(
                    feature_extractor,
                    preprocessed,
                    config.model_type,
                    device,
                    file_paths=file_paths,
                    clap_truncation=clap_truncation,
                    clap_padding=clap_padding,
                )

            # Forward pass
            x_hat, z = ae_model(features)
            loss = F.mse_loss(x_hat, features)

            # Backward pass
            optimizer.zero_grad()
            accelerator.backward(loss)
            if config.gradient_clip > 0:
                accelerator.clip_grad_norm_(ae_model.parameters(), config.gradient_clip)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Epoch metrics
        avg_loss = total_loss / len(dataloader)

        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        logger.info(f"Epoch {epoch+1}: loss={avg_loss:.6f}")

        # Save best model
        if accelerator.is_main_process:
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                torch.save(
                    accelerator.unwrap_model(ae_model).state_dict(),
                    save_dir / f"{config.model_type}_{downstream}_best.pth",
                )
            else:
                patience_counter += 1

        # Early stopping
        if patience_counter >= config.patience:
            logger.info(f"Early stopping at epoch {epoch+1}")
            break

    # Save final model
    if accelerator.is_main_process:
        torch.save(
            accelerator.unwrap_model(ae_model).state_dict(),
            save_dir / f"{config.model_type}_{downstream}_final.pth",
        )

    return {
        "downstream": downstream,
        "best_loss": best_loss,
        "epochs_trained": epoch + 1,
    }


# =============================================================================
# Main Entry Point
# =============================================================================

def run_downstream_training(
    tasks: List[str],
    config: UnifiedConfig,
) -> List[Dict[str, Any]]:
    """
    Run downstream training for specified tasks.

    Args:
        tasks: List of downstream task names
        config: Unified configuration

    Returns:
        List of result dictionaries for each task
    """
    # Setup logging
    run_name = f"{config.model_type}_{config.mode}"
    logger = setup_logging(config.log_dir, run_name)
    logger.info(f"Starting downstream training: {tasks}")

    # Set seed
    set_deterministic_mode(config.seed)

    # Setup accelerator
    if config.use_multi_gpu:
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    else:
        accelerator = Accelerator()

    device = accelerator.device
    logger.info(f"Using device: {device}")

    # Load feature extractor
    logger.info(f"Loading {config.model_type} feature extractor...")
    feature_extractor, embed_dim = load_feature_extractor(
        model_type=config.model_type,
        backbone=config.backbone,
        config=config,
        device=str(device),
    )
    logger.info(f"Feature extractor loaded. Embedding dim: {embed_dim}")

    # Train each task
    results = []
    for task in tasks:
        logger.info(f"\n{'='*50}")
        logger.info(f"Training: {task}")
        logger.info(f"{'='*50}")

        try:
            if task.endswith("_anomaly"):
                result = train_anomaly(
                    downstream=task,
                    config=config,
                    feature_extractor=feature_extractor,
                    embed_dim=embed_dim,
                    accelerator=accelerator,
                )
            else:
                result = train_classification(
                    downstream=task,
                    config=config,
                    feature_extractor=feature_extractor,
                    embed_dim=embed_dim,
                    accelerator=accelerator,
                )
            results.append(result)
            logger.info(f"Completed: {task} - {result}")

        except Exception as e:
            logger.error(f"Failed: {task} - {e}")
            traceback.print_exc()
            results.append({"downstream": task, "error": str(e)})

    return results
