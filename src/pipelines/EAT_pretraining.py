from __future__ import annotations
import os, math, glob, random, sys, json, yaml, hashlib, traceback
from dataclasses import asdict, replace
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
from collections import defaultdict
import logging
from datetime import datetime
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchaudio
from tqdm import tqdm
import csv
import warnings
from accelerate import Accelerator

from src.utils import (
    suppress_warnings,
    set_deterministic_mode,
    SpecConverter,
    EATMAE,
    EATMAETeacher,
    get_audio_info_safe,
)
from src.unified_config import UnifiedConfig, create_config

suppress_warnings()

# ============================================================================
# Utilities
# ============================================================================

def setup_logging(log_dir: str):
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def worker_init_fn(worker_id):
    """Initialize worker seed for DataLoader"""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def save_experiment_config(cfg: UnifiedConfig, save_dir: str):
    """Save experiment configuration"""
    config_dict = asdict(cfg)
    
    # Generate experiment ID
    config_str = json.dumps(config_dict, sort_keys=True)
    experiment_id = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    # Save config file
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    config_path = save_dir / f"config_{experiment_id}.json"
    
    with open(config_path, 'w') as f:
        json.dump({
            'experiment_id': experiment_id,
            'timestamp': datetime.now().isoformat(),
            'config': config_dict,
            'python_version': sys.version,
            'pytorch_version': torch.__version__,
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'cudnn_version': torch.backends.cudnn.version() if torch.cuda.is_available() else None
        }, f, indent=2)
    
    return experiment_id

# ============================================================================
# Dataset - Improved version
# ============================================================================

class RobustAudioDataset(Dataset):
    """Improved audio dataset - resolves deprecated API issues with caching support"""
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
# Training Components
# ============================================================================

class MetricLogger:
    """Metric logger for TensorBoard"""
    def __init__(self, log_dir: str):
        self.writer = SummaryWriter(log_dir)
        self.step = 0
        
    def log_scalar(self, tag: str, value: float, step: int = None):
        step = step or self.step
        self.writer.add_scalar(tag, value, step)
        
    def log_histogram(self, tag: str, values: torch.Tensor, step: int = None):
        step = step or self.step
        self.writer.add_histogram(tag, values, step)
        
    def log_model_weights(self, model: nn.Module, step: int = None):
        step = step or self.step
        for name, param in model.named_parameters():
            if param.grad is not None:
                self.writer.add_histogram(f'gradients/{name}', param.grad, step)
            self.writer.add_histogram(f'weights/{name}', param, step)
            
    def close(self):
        self.writer.close()

class CheckpointManager:
    """Checkpoint manager for model saving/loading - save all epochs"""
    def __init__(self, save_dir: str, cfg: UnifiedConfig, accelerator=None):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_files = []
        self.cfg = cfg
        self.accelerator = accelerator
        
    def save_checkpoint(self, model, optimizer, epoch, metrics, is_best=False):
        """Save checkpoint - save all epochs"""
        # Only save on main process
        if self.accelerator and not self.accelerator.is_main_process:
            return None
            
        # Unwrap model if using accelerator
        model_to_save = self.accelerator.unwrap_model(model) if self.accelerator else model
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'torch_version': torch.__version__,
        }
        
        # Generate filename
        filename = f"EAT_{self.cfg.dataset}_epoch_{epoch+1:04d}.pth"
        filepath = self.save_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint, filepath)
        self.checkpoint_files.append(filepath)
        logging.info(f"Saved checkpoint for epoch {epoch+1} with loss {metrics['loss']:.4f}")
        
        # Save best model separately
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            logging.info(f"Saved best model with loss {metrics['loss']:.4f}")
        
        return filepath
    
    def load_checkpoint(self, model, optimizer=None, checkpoint_path=None):
        """Load checkpoint"""
        if checkpoint_path is None:
            checkpoints = sorted(self.save_dir.glob("checkpoint_*.pth"))
            if not checkpoints:
                logging.warning("No checkpoint found")
                return None
            checkpoint_path = checkpoints[-1]
        
        logging.info(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint

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
# Training Loop
# ============================================================================

def run_epoch_with_amp(student, teacher, spec_converter, loader, opt, cfg, logger, accelerator, train=True):
    """Run one epoch with mixed precision training (eat.py style)"""
    student.train() if train else student.eval()
    teacher.eval()
    
    total_loss = 0.
    total_frame_loss = 0.
    total_utter_loss = 0.
    pbar = tqdm(loader, leave=False, desc="train" if train else "eval")
    
    # Create block masking generator - unwrap model to access grid_size
    from src.utils import BlockMaskingGenerator
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
        
        # Log to TensorBoard
        if train and logger and batch_idx % cfg.log_interval == 0:
            global_step = batch_idx + len(loader) * logger.step
            logger.log_scalar('batch/loss', loss.item(), global_step)
            logger.log_scalar('batch/frame_loss', frame_loss.item(), global_step)
            logger.log_scalar('batch/utter_loss', utter_loss.item(), global_step)
            if train:
                logger.log_scalar('batch/grad_norm', grad_norm.item(), global_step)
                logger.log_scalar('batch/learning_rate', opt.param_groups[0]['lr'], global_step)
    
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

def run_pretrain(
    cfg: Optional[UnifiedConfig] = None,
    datasets: Optional[List[str]] = None,
    device: Optional[str] = None,
    use_multi_gpu: bool = False,
    resume: Optional[str] = None,
):
    if device and device.startswith("cuda:") and not use_multi_gpu:
        try:
            torch.cuda.set_device(int(device.split(":")[1]))
        except Exception:
            pass
    # Initialize accelerator for multi-GPU training
    accelerator = Accelerator(
        mixed_precision='fp16',  # or 'bf16' if supported
        gradient_accumulation_steps=1,
        cpu=(device == "cpu"),
    )

    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend")
    warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")
    warnings.filterwarnings("ignore", message=".*Flash Attention.*")
    warnings.filterwarnings("ignore", message=".*Memory Efficient attention.*")

    if datasets is None:
        if accelerator.is_main_process:
            print("\n=== Select pretraining dataset ===")
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
            cfg_run = create_config(model_type="EAT", mode="scratch", dataset=dataset_name, device=device)
        else:
            cfg_run = replace(cfg, dataset=dataset_name)

        if device is not None and hasattr(cfg_run, "device"):
            cfg_run.device = device

        if cfg_run.train_data_path is None:
            cfg_run.train_data_path = f"./Datasets/{cfg_run.dataset}_Pretrain"

        set_deterministic_mode(cfg_run.seed or 42)
        logger = setup_logging(cfg_run.log_dir)

        if accelerator.is_main_process:
            experiment_id = save_experiment_config(cfg_run, cfg_run.log_dir)
            logger.info(f"Starting experiment {experiment_id} for dataset {cfg_run.dataset}")
            logger.info(f"Configuration: {cfg_run}")
            logger.info(f"Using {accelerator.num_processes} GPU(s) for training")
            logger.info("Using EAT architecture from eat.py")
            logger.info("Checkpoints for all epochs will be saved.")
        else:
            experiment_id = None

        if accelerator.num_processes > 1:
            import torch.distributed as dist
            exp_id_list = [experiment_id]
            dist.broadcast_object_list(exp_id_list, src=0)
            experiment_id = exp_id_list[0]

        Path(cfg_run.checkpoint_dir).mkdir(exist_ok=True, parents=True)

        tb_logger = MetricLogger(f"{cfg_run.log_dir}/runs/{experiment_id}") if cfg_run.tensorboard else None

        logger.info("Loading training dataset...")

        if cfg_run.dataset == 'Hybrid':
            train_data_paths = [
                "./Datasets/DCASE_Pretrain",
                "./Datasets/DINOS_Pretrain"
            ]
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

        student_dict = student.state_dict()
        teacher_dict = {k: v for k, v in student_dict.items() if k in teacher.model.state_dict()}
        teacher.model.load_state_dict(teacher_dict, strict=False)

        if accelerator.is_main_process:
            total_params = sum(p.numel() for p in student.parameters())
            trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")

        optimizer = torch.optim.AdamW(student.parameters(), lr=cfg_run.lr, weight_decay=cfg_run.wd)

        eta_min = getattr(cfg_run, "lr_min", 1e-6)
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg_run.epochs, eta_min=eta_min)

        student, teacher, optimizer, train_loader = accelerator.prepare(
            student, teacher, optimizer, train_loader
        )

        checkpoint_manager = CheckpointManager(cfg_run.checkpoint_dir, cfg_run, accelerator)

        start_epoch = 0
        if resume:
            checkpoint = checkpoint_manager.load_checkpoint(student, optimizer, resume)
            if checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                if accelerator.is_main_process:
                    logger.info(f"Resumed from epoch {start_epoch}")

        if accelerator.is_main_process:
            logger.info("Starting training...")

        csv_path = f"{cfg_run.log_dir}/training_log_{experiment_id}.csv"
        if accelerator.is_main_process:
            csvfile = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['epoch', 'train_loss', 'train_frame_loss',
                               'train_utter_loss', 'learning_rate', 'epoch_time'])

        for epoch in range(start_epoch, cfg_run.epochs):
            epoch_start = time.time()

            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch + 1}/{cfg_run.epochs}")

            train_metrics = run_epoch_with_amp(
                student, teacher, spec_converter, train_loader,
                optimizer, cfg_run, tb_logger, accelerator, train=True
            )

            current_lr = optimizer.param_groups[0]['lr']
            epoch_time = time.time() - epoch_start
            train_loss = accelerator.gather(torch.tensor(train_metrics['loss']).to(accelerator.device)).mean().item()

            if accelerator.is_main_process:
                logger.info(
                    f"Epoch {epoch + 1} - "
                    f"Train Loss: {train_loss:.4f}, "
                    f"LR: {current_lr:.6f}, "
                    f"Time: {epoch_time:.2f}s"
                )

                csv_writer.writerow([
                    epoch + 1,
                    train_loss,
                    train_metrics['frame_loss'],
                    train_metrics['utter_loss'],
                    current_lr,
                    epoch_time
                ])
                csvfile.flush()

            if tb_logger and accelerator.is_main_process:
                tb_logger.step = epoch
                tb_logger.log_scalar('epoch/train_loss', train_loss, epoch)
                tb_logger.log_scalar('epoch/learning_rate', current_lr, epoch)

                if epoch % 10 == 0:
                    tb_logger.log_model_weights(accelerator.unwrap_model(student), epoch)

            accelerator.wait_for_everyone()

            checkpoint_manager.save_checkpoint(
                student, optimizer, epoch,
                {'loss': train_loss, **train_metrics},
                is_best=False
            )

            scheduler.step()

        if accelerator.is_main_process:
            csvfile.close()
            logger.info("Training completed!")
            logger.info(f"Checkpoints for all {cfg_run.epochs} epochs have been saved.")

        if tb_logger and accelerator.is_main_process:
            tb_logger.close()

# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    run_pretrain(datasets=None, resume=args.resume)
    return
    # Initialize accelerator for multi-GPU training
    accelerator = Accelerator(
        mixed_precision='fp16',  # or 'bf16' if supported
        gradient_accumulation_steps=1,
    )
    
    # Parse arguments and load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()

    run_pretrain(datasets=None, resume=args.resume)
    return
    
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend")
    warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")
    warnings.filterwarnings("ignore", message=".*Flash Attention.*")
    warnings.filterwarnings("ignore", message=".*Memory Efficient attention.*")
    
    # Only show prompts on main process
    if accelerator.is_main_process:
        print("\n=== Select pretraining dataset ===")
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
    
    # Broadcast choice to all processes using tensor (more reliable than object_list)
    if accelerator.num_processes > 1:
        choice_tensor = torch.tensor([choice], dtype=torch.long, device=accelerator.device)
        # Broadcast from rank 0 to all other ranks
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
        cfg = create_config(model_type="EAT", mode="scratch", dataset=dataset_name, device=device)
        
        # Ensure train_data_path matches selected dataset
        if cfg.train_data_path is None:
            cfg.train_data_path = f"./Datasets/{cfg.dataset}_Pretrain"
       
        # Setup
        set_deterministic_mode(cfg.seed or 42)
        logger = setup_logging(cfg.log_dir)
        
        # Only log and save config on main process
        if accelerator.is_main_process:
            experiment_id = save_experiment_config(cfg, cfg.log_dir)
            logger.info(f"Starting experiment {experiment_id} for dataset {cfg.dataset}")
            logger.info(f"Configuration: {cfg}")
            logger.info(f"Using {accelerator.num_processes} GPU(s) for training")
            logger.info("Using EAT architecture from eat.py")
            logger.info("Checkpoints for all epochs will be saved.")
        else:
            experiment_id = None
        
        # Broadcast experiment_id to all processes
        if accelerator.num_processes > 1:
            import torch.distributed as dist
            exp_id_list = [experiment_id]
            dist.broadcast_object_list(exp_id_list, src=0)
            experiment_id = exp_id_list[0]
        
        # Create directories
        Path(cfg.checkpoint_dir).mkdir(exist_ok=True, parents=True)
        
        # Initialize TensorBoard logger
        tb_logger = MetricLogger(f"{cfg.log_dir}/runs/{experiment_id}") if cfg.tensorboard else None
        
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
        
        # Copy only shared parameters (encoder parts) from student to teacher
        student_dict = student.state_dict()
        teacher_dict = {k: v for k, v in student_dict.items() if k in teacher.model.state_dict()}
        teacher.model.load_state_dict(teacher_dict, strict=False)
        
        # Log model architecture
        if accelerator.is_main_process:
            total_params = sum(p.numel() for p in student.parameters())
            trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(student.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
        
        # Initialize cosine annealing scheduler with minimum learning rate
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=1e-6)
        
        # Prepare models, optimizer, and dataloader with accelerator
        student, teacher, optimizer, train_loader = accelerator.prepare(
            student, teacher, optimizer, train_loader
        )
        
        # Initialize checkpoint manager
        checkpoint_manager = CheckpointManager(cfg.checkpoint_dir, cfg, accelerator)
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if args.resume:
            checkpoint = checkpoint_manager.load_checkpoint(student, optimizer, args.resume)
            if checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                if accelerator.is_main_process:
                    logger.info(f"Resumed from epoch {start_epoch}")
        
        # Training loop
        if accelerator.is_main_process:
            logger.info("Starting training...")
        
        # CSV logging (only on main process)
        csv_path = f"{cfg.log_dir}/training_log_{experiment_id}.csv"
        if accelerator.is_main_process:
            csvfile = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['epoch', 'train_loss', 'train_frame_loss', 
                               'train_utter_loss', 'learning_rate', 'epoch_time'])
        
        for epoch in range(start_epoch, cfg.epochs):
            epoch_start = time.time()
            
            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch + 1}/{cfg.epochs}")
            
            # Training
            train_metrics = run_epoch_with_amp(
                student, teacher, spec_converter, train_loader, 
                optimizer, cfg, tb_logger, accelerator, train=True
            )
                
            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start
            
            # Gather metrics from all processes
            train_loss = accelerator.gather(torch.tensor(train_metrics['loss']).to(accelerator.device)).mean().item()
            
            # Log epoch metrics (only on main process)
            if accelerator.is_main_process:
                logger.info(f"Epoch {epoch + 1} - "
                           f"Train Loss: {train_loss:.4f}, "
                           f"LR: {current_lr:.6f}, "
                           f"Time: {epoch_time:.2f}s")
                
                # Log to CSV
                csv_writer.writerow([
                    epoch + 1,
                    train_loss,
                    train_metrics['frame_loss'],
                    train_metrics['utter_loss'],
                    current_lr,
                    epoch_time
                ])
                csvfile.flush()
                
            # Log to TensorBoard (only on main process)
            if tb_logger and accelerator.is_main_process:
                tb_logger.step = epoch
                tb_logger.log_scalar('epoch/train_loss', train_loss, epoch)
                tb_logger.log_scalar('epoch/learning_rate', current_lr, epoch)
                
                # Log model weights periodically
                if epoch % 10 == 0:
                    tb_logger.log_model_weights(accelerator.unwrap_model(student), epoch)
            
            # Wait for all processes to finish epoch
            accelerator.wait_for_everyone()
            
            # Save checkpoint - save all epochs (only on main process)
            checkpoint_manager.save_checkpoint(
                student, optimizer, epoch,
                {'loss': train_loss, **train_metrics},
                is_best=False
            )
            
            # Update learning rate scheduler
            scheduler.step()
        
        # Close CSV file on main process
        if accelerator.is_main_process:
            csvfile.close()
            logger.info("Training completed!")
            logger.info(f"Checkpoints for all {cfg.epochs} epochs have been saved.")
        
        # Close TensorBoard logger
        if tb_logger and accelerator.is_main_process:
            tb_logger.close()

if __name__ == "__main__":
    main()
