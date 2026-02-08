from __future__ import annotations
import os, sys, glob, time, random, logging, warnings, csv, math, json, hashlib
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from collections import defaultdict
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import replace
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm
import argparse
from accelerate import Accelerator
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("Warning: huggingface_hub not installed. Install with: pip install huggingface-hub")

from src.utils import (
    suppress_warnings,
    set_deterministic_mode,
    SpecConverter,
    EATMAE,
    EATMAETeacher,
    BlockMaskingGenerator,
    get_audio_info_safe
)
from src.unified_config import UnifiedConfig, create_config

suppress_warnings()

# ============================================================================
# SSL Dataset (for fine-tuning with self-supervised learning)
# ============================================================================

class RobustAudioDataset(Dataset):
    """SSL audio dataset for fine-tuning - loads from pretrain directories"""
    def __init__(self, wav_root: str | os.PathLike | List[str | os.PathLike], cfg: UnifiedConfig,
                 clips_per_folder: int | None = None, max_retries: int = 3, use_cache: bool = True):
        super().__init__()
        self.cfg = cfg
        self.max_retries = max_retries
        self.error_count = defaultdict(int)

        if cfg.seed is not None:
            random.seed(cfg.seed)

        # Support multiple root directories
        if isinstance(wav_root, (list, tuple)):
            wav_roots = [Path(root) for root in wav_root]
        else:
            wav_roots = [Path(wav_root)]

        # Generate cache file path based on dataset directories
        cache_key = "_".join(sorted([str(root) for root in wav_roots]))
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()[:8]
        cache_dir = Path("./cache")
        cache_dir.mkdir(exist_ok=True)
        cache_file = cache_dir / f"dataset_cache_{cache_hash}.json"

        clip_map: Dict[str, List[tuple[str, int, str]]] = defaultdict(list)

        # Try to load from cache
        if use_cache and cache_file.exists():
            try:
                logging.info(f"Loading dataset from cache: {cache_file}")
                with open(cache_file, 'r') as f:
                    cache_data = json.load(f)

                # Verify cache is valid by checking a few random files
                cached_files = cache_data.get('files', [])
                if cached_files:
                    sample_size = min(10, len(cached_files))
                    samples = random.sample(cached_files, sample_size)
                    all_exist = all(os.path.exists(fp) for fp, _, _ in samples)

                    if all_exist:
                        # Cache is valid, use it
                        for fp, n_clips, folder in cached_files:
                            for k in range(n_clips):
                                clip_map[folder].append((fp, k * cfg.sample_len, folder))

                        logging.info(f"Cache loaded successfully: {len(cached_files)} files, {len(clip_map)} folders")
                    else:
                        logging.warning("Cache validation failed, rescanning...")
                        clip_map = defaultdict(list)
            except Exception as e:
                logging.warning(f"Failed to load cache: {e}, rescanning...")
                clip_map = defaultdict(list)

        # If cache not loaded, scan files
        if not clip_map:
            logging.info("Starting audio file scan...")
            logging.info(f"Scanning directories: {[str(root) for root in wav_roots]}")

            # Expand file patterns (support more audio formats)
            audio_patterns = ["**/*.wav", "**/*.WAV", "**/*.mp3", "**/*.flac"]
            audio_files = []

            for wav_root in wav_roots:
                for pattern in audio_patterns:
                    audio_files.extend(glob.glob(str(wav_root / pattern), recursive=True))

            logging.info(f"Found {len(audio_files)} audio files")

            # Store cache data
            cache_files = []

            # Show scan progress
            valid_files = 0
            for fp in tqdm(audio_files, desc="Analyzing file info"):
                try:
                    audio_info = get_audio_info_safe(fp)
                    if audio_info is None:
                        continue

                    frames, file_sr = audio_info

                    if frames < cfg.sample_len:  # Check minimum length
                        continue

                    valid_files += 1
                    n_clips = math.ceil(frames / cfg.sample_len)

                    # Find which root directory this file belongs to
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

                    # Add to cache data
                    cache_files.append((fp, n_clips, folder))

                    for k in range(n_clips):
                        clip_map[folder].append((fp, k * cfg.sample_len, folder))

                except Exception as e:
                    logging.debug(f"Failed to process file {fp}: {e}")
                    continue

            logging.info(f"Valid files: {valid_files}, folders: {len(clip_map)}")

            # Save cache
            if use_cache:
                try:
                    cache_data = {
                        'roots': [str(root) for root in wav_roots],
                        'sample_len': cfg.sample_len,
                        'files': cache_files,
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(cache_file, 'w') as f:
                        json.dump(cache_data, f)
                    logging.info(f"Cache saved to {cache_file}")
                except Exception as e:
                    logging.warning(f"Failed to save cache: {e}")

        # Create index list
        self.index_list = []
        for fld, clips in clip_map.items():
            chosen = clips if clips_per_folder is None else \
                     random.sample(clips, min(len(clips), clips_per_folder))
            self.index_list.extend(chosen)

        self.class_to_idx = {f: i for i, f in enumerate(sorted(clip_map))}

        logging.info(f"Total clips: {len(self.index_list)}")

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        for retry in range(self.max_retries):
            try:
                fp, start, folder = self.index_list[idx]

                # Check file existence
                if not os.path.exists(fp):
                    raise FileNotFoundError(f"File not found: {fp}")

                # Suppress warnings while loading audio
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    # Load audio using soundfile (avoids torchcodec issues)
                    import soundfile as sf
                    try:
                        # Read specific frames
                        audio, _sr = sf.read(
                            fp,
                            start=start,
                            frames=self.cfg.sample_len,
                            dtype='float32',
                            always_2d=True
                        )
                        audio = audio.T  # Transpose to [channels, samples]
                    except Exception as e:
                        # fallback: load entire file then slice
                        audio, _sr = sf.read(fp, dtype='float32', always_2d=True)
                        audio = audio.T  # Transpose to [channels, samples]
                        end_frame = start + self.cfg.sample_len
                        audio = audio[:, start:end_frame]

                # Handle sample rate mismatch
                if _sr != self.cfg.sr:
                    # Convert to torch tensor for resampling
                    audio_tensor = torch.from_numpy(audio)
                    resampler = torchaudio.transforms.Resample(_sr, self.cfg.sr)
                    audio_tensor = resampler(audio_tensor)
                    audio = audio_tensor.numpy()

                # Convert to mono if stereo
                if audio.shape[0] > 1:
                    audio = audio.mean(0)  # Average across channel dimension
                else:
                    audio = audio[0]  # Remove dimension for single channel

                # Pad if necessary
                audio = np.pad(audio, (0, max(0, self.cfg.sample_len - len(audio))))[:self.cfg.sample_len]

                # Convert to tensor
                waveform = torch.tensor(audio, dtype=torch.float32)

                return waveform, self.class_to_idx[folder]

            except Exception as e:
                self.error_count[fp] += 1
                logging.error(f"Error loading {fp} (attempt {retry+1}/{self.max_retries}): {e}")

                if retry == self.max_retries - 1:
                    logging.error(f"Max retries exceeded for {fp}, using next sample")
                    return self.__getitem__((idx + 1) % len(self))

                time.sleep(0.1 * (retry + 1))

# ============================================================================
# Training Utilities
# ============================================================================

def setup_logging(log_dir: str):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'finetuning.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def worker_init_fn(worker_id):
    """Initialize worker seed for DataLoader"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def convert_as2_state_dict(hf_state_dict):
    """Convert Hugging Face EAT state dict to match our EATMAE structure"""
    converted = {}
    
    # Map patch embedding: model.local_encoder.proj -> patch_embed.proj
    if 'model.local_encoder.proj.weight' in hf_state_dict:
        converted['patch_embed.proj.weight'] = hf_state_dict['model.local_encoder.proj.weight']
    if 'model.local_encoder.proj.bias' in hf_state_dict:
        converted['patch_embed.proj.bias'] = hf_state_dict['model.local_encoder.proj.bias']
    
    # Map CLS token: model.extra_tokens -> cls_token
    if 'model.extra_tokens' in hf_state_dict:
        # extra_tokens might contain cls_token
        converted['cls_token'] = hf_state_dict['model.extra_tokens'][:, :1, :]
    
    # Map transformer blocks: model.blocks.X -> encoder.layers.X
    for i in range(12):  # Assuming 12 layers
        # Attention: qkv -> in_proj, proj -> out_proj
        if f'model.blocks.{i}.attn.qkv.weight' in hf_state_dict:
            converted[f'encoder.layers.{i}.self_attn.in_proj_weight'] = hf_state_dict[f'model.blocks.{i}.attn.qkv.weight']
        if f'model.blocks.{i}.attn.qkv.bias' in hf_state_dict:
            converted[f'encoder.layers.{i}.self_attn.in_proj_bias'] = hf_state_dict[f'model.blocks.{i}.attn.qkv.bias']
        if f'model.blocks.{i}.attn.proj.weight' in hf_state_dict:
            converted[f'encoder.layers.{i}.self_attn.out_proj.weight'] = hf_state_dict[f'model.blocks.{i}.attn.proj.weight']
        if f'model.blocks.{i}.attn.proj.bias' in hf_state_dict:
            converted[f'encoder.layers.{i}.self_attn.out_proj.bias'] = hf_state_dict[f'model.blocks.{i}.attn.proj.bias']
        
        # MLP: fc1 -> linear1, fc2 -> linear2
        if f'model.blocks.{i}.mlp.fc1.weight' in hf_state_dict:
            converted[f'encoder.layers.{i}.linear1.weight'] = hf_state_dict[f'model.blocks.{i}.mlp.fc1.weight']
        if f'model.blocks.{i}.mlp.fc1.bias' in hf_state_dict:
            converted[f'encoder.layers.{i}.linear1.bias'] = hf_state_dict[f'model.blocks.{i}.mlp.fc1.bias']
        if f'model.blocks.{i}.mlp.fc2.weight' in hf_state_dict:
            converted[f'encoder.layers.{i}.linear2.weight'] = hf_state_dict[f'model.blocks.{i}.mlp.fc2.weight']
        if f'model.blocks.{i}.mlp.fc2.bias' in hf_state_dict:
            converted[f'encoder.layers.{i}.linear2.bias'] = hf_state_dict[f'model.blocks.{i}.mlp.fc2.bias']
        
        # LayerNorm
        if f'model.blocks.{i}.norm1.weight' in hf_state_dict:
            converted[f'encoder.layers.{i}.norm1.weight'] = hf_state_dict[f'model.blocks.{i}.norm1.weight']
        if f'model.blocks.{i}.norm1.bias' in hf_state_dict:
            converted[f'encoder.layers.{i}.norm1.bias'] = hf_state_dict[f'model.blocks.{i}.norm1.bias']
        if f'model.blocks.{i}.norm2.weight' in hf_state_dict:
            converted[f'encoder.layers.{i}.norm2.weight'] = hf_state_dict[f'model.blocks.{i}.norm2.weight']
        if f'model.blocks.{i}.norm2.bias' in hf_state_dict:
            converted[f'encoder.layers.{i}.norm2.bias'] = hf_state_dict[f'model.blocks.{i}.norm2.bias']
    
    return converted

def load_pretrained_model(model: EATMAE, pretrained_path: str, device: str, unfreeze: bool = True):
    """Load pretrained model and optionally unfreeze for fine-tuning"""
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
    
    print(f"Loading pretrained model: {pretrained_path}")
    
    try:
        checkpoint = torch.load(pretrained_path, map_location=device, weights_only=False)
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Check if this is an AS2 (Hugging Face) model that needs conversion
        needs_conversion = any(key.startswith('model.blocks.') for key in state_dict.keys())
        
        if needs_conversion:
            print("Detected Hugging Face EAT model format. Converting state dict...")
            state_dict = convert_as2_state_dict(state_dict)
            print(f"Converted {len(state_dict)} parameters")
        
        # Load with strict=False to allow missing decoder weights
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        # Filter out expected missing keys (decoder-related)
        decoder_keys = [k for k in missing_keys if 'decoder' in k or 'mask_token' in k]
        critical_missing = [k for k in missing_keys if k not in decoder_keys]
        
        if critical_missing:
            print(f"Warning: Missing critical keys: {critical_missing[:5]}...")
        
        print("Pretrained model loaded successfully")
        if decoder_keys:
            print(f"Note: {len(decoder_keys)} decoder-related keys not loaded (expected for downstream tasks)")
        
        # Set requires_grad based on unfreeze parameter
        if unfreeze:
            print("Unfreezing encoder for fine-tuning")
            model.train()
            for param in model.parameters():
                param.requires_grad = True
        else:
            print("Freezing encoder (feature extraction only)")
            model.eval()
            for param in model.parameters():
                param.requires_grad = False
            
        return True
        
    except Exception as e:
        print(f"Model loading failed: {e}")
        return False

def download_as2_model(save_path: str) -> bool:
    """Download AS2 pretrained model from Hugging Face"""
    if not HF_AVAILABLE:
        print("Error: huggingface_hub is not installed.")
        print("Please install it with: pip install huggingface-hub")
        return False
    
    try:
        print("Downloading AS2 pretrained model from Hugging Face...")
        print("Repository: worstchan/EAT-base_epoch30_pretrain")
        
        # Download the safetensors model file
        downloaded_path = hf_hub_download(
            repo_id="worstchan/EAT-base_epoch30_pretrain",
            filename="model.safetensors",
            cache_dir=None,
            force_download=False
        )
        
        print(f"Model downloaded: {downloaded_path}")
        print("Converting safetensors to PyTorch format...")
        
        # Load safetensors and convert to PyTorch format
        try:
            from safetensors.torch import load_file
            state_dict = load_file(downloaded_path)
        except ImportError:
            print("Error: safetensors not installed. Installing...")
            import subprocess
            subprocess.check_call(["pip", "install", "safetensors"])
            from safetensors.torch import load_file
            state_dict = load_file(downloaded_path)
        
        # Save as PyTorch checkpoint
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({'model_state_dict': state_dict}, save_path)
        
        print(f"Model converted and saved successfully: {save_path}")
        return True
        
    except Exception as e:
        print(f"Failed to download model: {e}")
        print(f"Please download manually from: https://huggingface.co/worstchan/EAT-base_epoch30_pretrain")
        return False

def update_teacher(student, teacher, m=0.999):
    """Update teacher model with EMA - only update encoder parameters"""
    with torch.no_grad():
        # Teacher is wrapped in EATMAETeacher, so we access teacher.model
        # Only update parameters that exist in both models
        teacher_state = teacher.model.state_dict()
        student_state = student.state_dict()

        for name in teacher_state.keys():
            if name in student_state:
                teacher_state[name] = m * teacher_state[name] + (1 - m) * student_state[name]

        teacher.model.load_state_dict(teacher_state)

def create_optimized_dataloader(dataset, cfg, train=True):
    """Create optimized DataLoader"""
    num_workers = min(cfg.num_workers, os.cpu_count() or 1)

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=(cfg.device == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=worker_init_fn
    )

    return dataloader

# ============================================================================
# SSL Fine-tuning Functions
# ============================================================================

def run_epoch_with_amp(student, teacher, spec_converter, loader, opt, cfg, accelerator, train=True):
    """Run one epoch with SSL (MAE) training"""
    student.train() if train else student.eval()
    teacher.eval()

    total_loss = 0.
    total_frame_loss = 0.
    total_utter_loss = 0.
    pbar = tqdm(loader, leave=False, desc="train" if train else "eval")

    # Create block masking generator - unwrap model to access grid_size
    student_module = accelerator.unwrap_model(student)
    mask_gen = BlockMaskingGenerator(student_module.grid_size, cfg.mask_ratio, cfg.block_size)

    for batch_idx, (waveform, label) in enumerate(pbar):
        waveform = waveform.to(cfg.device)

        # Convert waveform to spectrogram on GPU
        with torch.no_grad():
            spec = spec_converter(waveform)

        if train:
            opt.zero_grad()

        # Teacher forward pass (no masking)
        with torch.no_grad():
            t_patch, t_utter = teacher(spec)

        # Multi-clone approach for student
        K = cfg.num_clones
        spec_student = spec.repeat_interleave(K, 0)
        ids_restore, len_keep = mask_gen(spec_student.shape[0], accelerator.device)

        # Student forward pass (with masking)
        s_recon, s_cls = student(spec_student, ids_restore=ids_restore)

        # Calculate masked region
        ids_masked = torch.argsort(ids_restore, 1)[:, len_keep:]
        mask_bool = torch.zeros(spec_student.shape[0], s_recon.shape[1], dtype=torch.bool, device=accelerator.device)
        mask_bool.scatter_(1, ids_masked, True)

        # Calculate losses (only on masked regions for frame loss)
        frame_loss = F.mse_loss(s_recon[mask_bool], t_patch.repeat_interleave(K, 0)[mask_bool])
        utter_loss = F.mse_loss(s_cls, t_utter.repeat_interleave(K, 0))

        loss = frame_loss + cfg.utter_loss_weight * utter_loss

        if train:
            # Accelerator handles mixed precision automatically
            accelerator.backward(loss)

            # Gradient clipping
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(student.parameters(), cfg.gradient_clip)
            else:
                grad_norm = torch.tensor(0.0)

            opt.step()

            # Update teacher with EMA (unwrap model for multi-GPU)
            student_unwrapped = accelerator.unwrap_model(student)
            teacher_unwrapped = accelerator.unwrap_model(teacher)
            update_teacher(student_unwrapped, teacher_unwrapped, m=cfg.teacher_momentum)

        # Update metrics
        total_loss += loss.item()
        total_frame_loss += frame_loss.item()
        total_utter_loss += utter_loss.item()

        # Update progress bar
        pbar.set_postfix({
            'loss': loss.item(),
            'loc': frame_loss.item(),
            'cls': utter_loss.item(),
            'grad': grad_norm.item() if train else 0
        })

    # Calculate epoch averages
    n_batches = len(loader)
    avg_loss = total_loss / n_batches
    avg_frame_loss = total_frame_loss / n_batches
    avg_utter_loss = total_utter_loss / n_batches

    return {
        'loss': avg_loss,
        'frame_loss': avg_frame_loss,
        'utter_loss': avg_utter_loss
    }

# ============================================================================
# Programmatic Entry
# ============================================================================

def run_finetune(
    cfg: Optional[UnifiedConfig] = None,
    datasets: Optional[List[str]] = None,
    device: Optional[str] = None,
    use_multi_gpu: bool = False,
    use_hf_model: bool = False,
    pretrained_path: Optional[str] = None,
    resume: Optional[str] = None,
    epochs: Optional[int] = None,
):
    if device and device.startswith("cuda:") and not use_multi_gpu:
        try:
            torch.cuda.set_device(int(device.split(":")[1]))
        except Exception:
            pass
    """Programmatic SSL fine-tuning entrypoint."""
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=1,
        log_with=None,
        cpu=(device == "cpu"),
    )

    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend")
    warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")
    warnings.filterwarnings("ignore", message=".*Flash Attention.*")
    warnings.filterwarnings("ignore", message=".*Memory Efficient attention.*")

    if datasets is None:
        if accelerator.is_main_process:
            print("\n=== Select dataset for SSL fine-tuning ===")
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

        mapping = {1: 'DINOS', 2: 'DCASE', 3: 'Hybrid', 4: 'All'}
        if choice not in mapping:
            if accelerator.is_main_process:
                print("Invalid selection after broadcast.")
            return

        datasets_to_run = ['DINOS', 'DCASE', 'Hybrid'] if mapping[choice] == 'All' else [mapping[choice]]
    else:
        datasets_to_run = datasets

    for dataset_name in datasets_to_run:
        if cfg is None:
            cfg_run = create_config(model_type="EAT", mode="finetune", dataset=dataset_name, device=device)
        else:
            cfg_run = replace(cfg, dataset=dataset_name)

        if device is not None and hasattr(cfg_run, "device"):
            cfg_run.device = device

        if epochs is not None:
            cfg_run.epochs = epochs

        if cfg_run.train_data_path is None:
            cfg_run.train_data_path = f"./Datasets/{cfg_run.dataset}_Pretrain"

        # Determine pretrained model path
        if use_hf_model:
            pretrained_path_use = "./EAT_Models/HF_AS2_pretrained.pth"
            if not os.path.exists(pretrained_path_use) and accelerator.is_main_process:
                print("\n" + "="*60)
                print("Downloading Hugging Face pretrained model...")
                print("Repository: worstchan/EAT-base_epoch30_pretrain")
                print("="*60 + "\n")
                if not download_as2_model(pretrained_path_use):
                    print("Failed to download model. Exiting.")
                    return
            if accelerator.num_processes > 1:
                import torch.distributed as dist
                dist.barrier()
        elif pretrained_path:
            pretrained_path_use = pretrained_path
        else:
            pretrained_path_use = f"./EAT_Models/EAT_{cfg_run.dataset}_epoch_0020.pth"

        set_deterministic_mode(cfg_run.seed or 42)
        logger = setup_logging(cfg_run.log_dir)

        if accelerator.is_main_process:
            logger.info(f"Starting SSL fine-tuning for dataset {cfg_run.dataset}")
            logger.info(f"Configuration: {cfg_run}")
            logger.info(f"Using {accelerator.num_processes} GPU(s) for fine-tuning")
            logger.info("Using EAT architecture with SSL (MAE)")

        Path(cfg_run.checkpoint_dir).mkdir(exist_ok=True, parents=True)

        logger.info("Loading training dataset...")
        if cfg_run.dataset == 'Hybrid':
            train_data_paths = ["./Datasets/DCASE_Pretrain", "./Datasets/DINOS_Pretrain"]
            logger.info(f"Using Hybrid dataset: {train_data_paths}")
            train_dataset = RobustAudioDataset(train_data_paths, cfg_run)
        else:
            train_dataset = RobustAudioDataset(cfg_run.train_data_path, cfg_run)

        logger.info(f"Training dataset size: {len(train_dataset)}")
        train_loader = create_optimized_dataloader(train_dataset, cfg_run, train=True)

        if accelerator.is_main_process:
            logger.info("Initializing models with EAT architecture...")
        spec_converter = SpecConverter(cfg_run).to(accelerator.device)
        student = EATMAE(cfg_run, is_teacher=False).to(accelerator.device)
        teacher = EATMAETeacher(cfg_run).to(accelerator.device)

        # Load pretrained model
        if os.path.exists(pretrained_path_use):
            if accelerator.is_main_process:
                logger.info(f"Loading pretrained model from: {pretrained_path_use}")
            checkpoint = torch.load(pretrained_path_use, map_location=accelerator.device, weights_only=False)

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            if use_hf_model:
                needs_conversion = any(key.startswith('model.blocks.') for key in state_dict.keys())
                if needs_conversion and accelerator.is_main_process:
                    logger.info("Detected Hugging Face EAT model format. Converting state dict...")
                    state_dict = convert_as2_state_dict(state_dict)
                    logger.info(f"Converted {len(state_dict)} parameters")

            student.load_state_dict(state_dict, strict=False)
        else:
            if accelerator.is_main_process:
                logger.warning(f"Pretrained model not found at {pretrained_path_use}, starting from scratch")

        student_dict = student.state_dict()
        teacher_dict = {k: v for k, v in student_dict.items() if k in teacher.model.state_dict()}
        teacher.model.load_state_dict(teacher_dict, strict=False)

        optimizer = torch.optim.AdamW(student.parameters(), lr=cfg_run.lr, weight_decay=cfg_run.wd)
        eta_min = getattr(cfg_run, "lr_min", 1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg_run.epochs, eta_min=eta_min)

        student, teacher, optimizer, train_loader, scheduler = accelerator.prepare(
            student, teacher, optimizer, train_loader, scheduler
        )

        if accelerator.is_main_process:
            logger.info(f"Starting SSL fine-tuning for {cfg_run.epochs} epochs...")

        best_loss = float('inf')
        for epoch in range(cfg_run.epochs):
            train_metrics = run_epoch_with_amp(
                student, teacher, spec_converter, train_loader,
                optimizer, cfg_run, accelerator, train=True
            )

            if accelerator.is_main_process:
                logger.info(
                    f"Epoch {epoch+1}/{cfg_run.epochs} - "
                    f"Loss: {train_metrics['loss']:.4f}, "
                    f"Frame: {train_metrics['frame_loss']:.4f}, "
                    f"Utter: {train_metrics['utter_loss']:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

                if train_metrics['loss'] < best_loss:
                    best_loss = train_metrics['loss']
                    save_path = Path(cfg_run.checkpoint_dir) / f"EAT_{cfg_run.dataset}_finetuned_best.pth"
                    model_to_save = accelerator.unwrap_model(student)

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_metrics['loss'],
                        'frame_loss': train_metrics['frame_loss'],
                        'utter_loss': train_metrics['utter_loss'],
                        'timestamp': datetime.now().isoformat(),
                    }, save_path)
                    logger.info(f"Best model saved: {save_path}")

            scheduler.step()

        if accelerator.is_main_process:
            save_path = Path(cfg_run.checkpoint_dir) / f"EAT_{cfg_run.dataset}_finetuned_final.pth"
            model_to_save = accelerator.unwrap_model(student)

            torch.save({
                'epoch': cfg_run.epochs,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_metrics['loss'],
                'timestamp': datetime.now().isoformat(),
            }, save_path)
            logger.info(f"Final model saved: {save_path}")
            logger.info(f"=== SSL fine-tuning complete: {cfg_run.dataset} ===")
            logger.info(f"Best loss: {best_loss:.4f}")

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main SSL fine-tuning function with multi-GPU support via Accelerate"""
    parser = argparse.ArgumentParser(description='EAT Model SSL Fine-tuning')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=10, help='Number of fine-tuning epochs')
    parser.add_argument('--use_hf_model', action='store_true',
                       help='Use Hugging Face pretrained model (worstchan/EAT-base_epoch30_pretrain)')
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pretrained model (if not using HF model)')
    args = parser.parse_args()

    run_finetune(
        datasets=None,
        use_hf_model=args.use_hf_model,
        pretrained_path=args.pretrained_path,
        resume=args.resume,
        epochs=args.epochs,
    )
    return
    # Initialize accelerator for multi-GPU training
    # This automatically detects and uses all available GPUs
    accelerator = Accelerator(
        mixed_precision='fp16',  # FP16 for faster training (use 'bf16' if supported)
        gradient_accumulation_steps=1,
        log_with=None,  # Disable default logging
    )

    # Print GPU information (only on main process)
    if accelerator.is_main_process:
        print("\n" + "="*70)
        print("Multi-GPU Training with Accelerate")
        print("="*70)
        print(f"Number of processes: {accelerator.num_processes}")
        print(f"Current process index: {accelerator.process_index}")
        print(f"Device: {accelerator.device}")
        print(f"Mixed precision: {accelerator.mixed_precision}")
        print(f"Distributed type: {accelerator.distributed_type}")
        if torch.cuda.is_available():
            print(f"CUDA devices available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print("="*70 + "\n")

    # Parse arguments
    parser = argparse.ArgumentParser(description='EAT Model SSL Fine-tuning')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--epochs', type=int, default=10, help='Number of fine-tuning epochs')
    parser.add_argument('--use_hf_model', action='store_true',
                       help='Use Hugging Face pretrained model (worstchan/EAT-base_epoch30_pretrain)')
    parser.add_argument('--pretrained_path', type=str, default=None,
                       help='Path to pretrained model (if not using HF model)')
    args = parser.parse_args()

    run_finetune(
        datasets=None,
        use_hf_model=args.use_hf_model,
        pretrained_path=args.pretrained_path,
        resume=args.resume,
        epochs=args.epochs,
    )
    return

    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend")
    warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")
    warnings.filterwarnings("ignore", message=".*Flash Attention.*")
    warnings.filterwarnings("ignore", message=".*Memory Efficient attention.*")

    # Only show prompts on main process
    if accelerator.is_main_process:
        print("\n=== Select dataset for SSL fine-tuning ===")
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
        choice = 0  # Placeholder for non-main processes

    # Broadcast choice to all processes using tensor
    if accelerator.num_processes > 1:
        choice_tensor = torch.tensor([choice], dtype=torch.long, device=accelerator.device)
        torch.distributed.broadcast(choice_tensor, src=0)
        choice = choice_tensor.item()

    mapping = {1: 'DINOS', 2: 'DCASE', 3: 'Hybrid', 4: 'All'}
    if choice not in mapping:
        if accelerator.is_main_process:
            print("Invalid selection after broadcast.")
        return

    if mapping[choice] == 'All':
        datasets_to_run = ['DINOS', 'DCASE', 'Hybrid']
    else:
        datasets_to_run = [mapping[choice]]

    for dataset_name in datasets_to_run:
        cfg = create_config(model_type="EAT", mode="finetune", dataset=dataset_name, epochs=args.epochs)

        # Ensure train_data_path matches selected dataset
        if cfg.train_data_path is None:
            cfg.train_data_path = f"./Datasets/{cfg.dataset}_Pretrain"

        # Determine pretrained model path
        if args.use_hf_model:
            # Use Hugging Face pretrained model
            pretrained_path = "./EAT_Models/HF_AS2_pretrained.pth"
            if not os.path.exists(pretrained_path) and accelerator.is_main_process:
                print("\n" + "="*60)
                print("Downloading Hugging Face pretrained model...")
                print("Repository: worstchan/EAT-base_epoch30_pretrain")
                print("="*60 + "\n")
                if not download_as2_model(pretrained_path):
                    print("Failed to download model. Exiting.")
                    return
            # Wait for main process to finish downloading
            if accelerator.num_processes > 1:
                import torch.distributed as dist
                dist.barrier()
        elif args.pretrained_path:
            # Use user-specified pretrained model
            pretrained_path = args.pretrained_path
        else:
            # Use local pretrained model (default)
            pretrained_path = f"./EAT_Models/EAT_{cfg.dataset}_epoch_0020.pth"

        # Setup
        set_deterministic_mode(cfg.seed or 42)
        logger = setup_logging(cfg.log_dir)

        # Only log and save config on main process
        if accelerator.is_main_process:
            logger.info(f"Starting SSL fine-tuning for dataset {cfg.dataset}")
            logger.info(f"Configuration: {cfg}")
            logger.info(f"Using {accelerator.num_processes} GPU(s) for fine-tuning")
            logger.info("Using EAT architecture with SSL (MAE)")

        # Create directories
        Path(cfg.checkpoint_dir).mkdir(exist_ok=True, parents=True)

        # Load datasets
        logger.info("Loading training dataset...")

        # Handle Hybrid dataset: use both DCASE_Pretrain and DINOS_Pretrain
        if cfg.dataset == 'Hybrid':
            train_data_paths = [
                "./Datasets/DCASE_Pretrain",
                "./Datasets/DINOS_Pretrain"
            ]
            logger.info(f"Using Hybrid dataset: {train_data_paths}")
            train_dataset = RobustAudioDataset(train_data_paths, cfg)
        else:
            train_dataset = RobustAudioDataset(cfg.train_data_path, cfg)

        logger.info(f"Training dataset size: {len(train_dataset)}")

        # Create dataloader
        train_loader = create_optimized_dataloader(train_dataset, cfg, train=True)

        # Initialize models
        if accelerator.is_main_process:
            logger.info("Initializing models with EAT architecture...")
        spec_converter = SpecConverter(cfg).to(accelerator.device)
        student = EATMAE(cfg, is_teacher=False).to(accelerator.device)
        teacher = EATMAETeacher(cfg).to(accelerator.device)

        # Load pretrained model
        if os.path.exists(pretrained_path):
            if accelerator.is_main_process:
                logger.info(f"Loading pretrained model from: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=accelerator.device, weights_only=False)

            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # Check if this is a Hugging Face model that needs conversion
            if args.use_hf_model:
                needs_conversion = any(key.startswith('model.blocks.') for key in state_dict.keys())
                if needs_conversion and accelerator.is_main_process:
                    logger.info("Detected Hugging Face EAT model format. Converting state dict...")
                    state_dict = convert_as2_state_dict(state_dict)
                    logger.info(f"Converted {len(state_dict)} parameters")

            # Load state dict (strict=False to allow for minor mismatches)
            missing_keys, unexpected_keys = student.load_state_dict(state_dict, strict=False)
            if accelerator.is_main_process:
                if missing_keys:
                    logger.warning(f"Missing keys: {missing_keys[:5]}...")
                if unexpected_keys:
                    logger.warning(f"Unexpected keys: {unexpected_keys[:5]}...")
                logger.info("Pretrained model loaded successfully")
        else:
            if accelerator.is_main_process:
                logger.warning(f"Pretrained model not found at {pretrained_path}, starting from scratch")

        # Copy student parameters to teacher
        student_dict = student.state_dict()
        teacher_dict = {k: v for k, v in student_dict.items() if k in teacher.model.state_dict()}
        teacher.model.load_state_dict(teacher_dict, strict=False)

        # Setup optimizer
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.wd
        )

        # Setup scheduler
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)

        # Prepare for distributed training
        student, teacher, optimizer, train_loader, scheduler = accelerator.prepare(
            student, teacher, optimizer, train_loader, scheduler
        )

        # Training loop
        if accelerator.is_main_process:
            logger.info(f"Starting SSL fine-tuning for {cfg.epochs} epochs...")

        best_loss = float('inf')

        for epoch in range(cfg.epochs):
            # Train
            train_metrics = run_epoch_with_amp(
                student, teacher, spec_converter, train_loader,
                optimizer, cfg, accelerator, train=True
            )

            # Log metrics
            if accelerator.is_main_process:
                logger.info(
                    f"Epoch {epoch+1}/{cfg.epochs} - "
                    f"Loss: {train_metrics['loss']:.4f}, "
                    f"Frame: {train_metrics['frame_loss']:.4f}, "
                    f"Utter: {train_metrics['utter_loss']:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}"
                )

                # Save checkpoint
                if train_metrics['loss'] < best_loss:
                    best_loss = train_metrics['loss']
                    save_path = Path(cfg.checkpoint_dir) / f"EAT_{cfg.dataset}_finetuned_best.pth"

                    # Unwrap model for saving
                    model_to_save = accelerator.unwrap_model(student)

                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model_to_save.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': train_metrics['loss'],
                        'frame_loss': train_metrics['frame_loss'],
                        'utter_loss': train_metrics['utter_loss'],
                        'timestamp': datetime.now().isoformat(),
                    }, save_path)
                    logger.info(f"Best model saved: {save_path}")

            # Update learning rate
            scheduler.step()

        # Save final model
        if accelerator.is_main_process:
            save_path = Path(cfg.checkpoint_dir) / f"EAT_{cfg.dataset}_finetuned_final.pth"
            model_to_save = accelerator.unwrap_model(student)

            torch.save({
                'epoch': cfg.epochs,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_metrics['loss'],
                'timestamp': datetime.now().isoformat(),
            }, save_path)
            logger.info(f"Final model saved: {save_path}")
            logger.info(f"=== SSL fine-tuning complete: {cfg.dataset} ===")
            logger.info(f"Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    main()

'''
===================================================================================
EAT SSL Fine-tuning with Multi-GPU Support (via Accelerate)
===================================================================================

SINGLE GPU:
-----------
python EAT_finetuning.py --use_hf_model --epochs 10

MULTI-GPU (Recommended - Auto-detect all GPUs):
------------------------------------------------
accelerate launch EAT_finetuning.py --use_hf_model --epochs 10

MULTI-GPU (Specify GPUs):
--------------------------
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch EAT_finetuning.py --use_hf_model --epochs 10

MULTI-GPU with Custom Config:
------------------------------
# First, configure accelerate (run once)
accelerate config

# Then launch with your config
accelerate launch EAT_finetuning.py --use_hf_model --epochs 10

OTHER OPTIONS:
--------------
# Use local pretrained model
accelerate launch EAT_finetuning.py --epochs 20

# Use custom pretrained model
accelerate launch EAT_finetuning.py --pretrained_path ./path/to/model.pth --epochs 10

# Resume from checkpoint
accelerate launch EAT_finetuning.py --resume ./EAT_Models/checkpoint.pth

# Combine options
accelerate launch EAT_finetuning.py --use_hf_model --epochs 20

NOTES:
------
- The code automatically uses all available GPUs when launched with 'accelerate launch'
- Mixed precision (FP16) is enabled by default for faster training
- Data is automatically distributed across GPUs
- Gradients are synchronized automatically
- Only the main process saves checkpoints to avoid conflicts

PERFORMANCE TIPS:
-----------------
- For 4 GPUs, effective batch size = batch_size * 4
- Adjust batch_size in config if you get OOM errors
- Use CUDA_VISIBLE_DEVICES to control which GPUs to use
- Monitor GPU memory usage with: watch -n 1 nvidia-smi
'''
