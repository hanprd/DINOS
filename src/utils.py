from __future__ import annotations

import logging
import os
import random
import warnings
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torchaudio


def suppress_warnings() -> None:
    """Suppress noisy torchaudio warnings."""
    warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio._backend")
    warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*")
    warnings.filterwarnings("ignore", message=".*torchaudio.load_with_torchcodec.*")


def set_deterministic_mode(seed: int = 42) -> None:
    """Set deterministic mode for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cuda.enable_mem_efficient_sdp(False)

    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        pass


def build_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=True):
    """Build 2D sinusoidal position embedding (from eat.py)."""
    grid_h, grid_w = grid_size
    grid_h = np.arange(grid_h, dtype=np.float32)
    grid_w = np.arange(grid_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_h.size, grid_w.size])
    
    # Split embed_dim between height and width
    omega = np.arange(embed_dim // 4, dtype=np.float32) / (embed_dim / 4.)
    omega = 1. / 10000**omega
    
    out_h = np.einsum('m,d->md', grid[0].reshape(-1), omega)
    out_w = np.einsum('m,d->md', grid[1].reshape(-1), omega)
    
    # Concatenate sin/cos for both dimensions: total = (embed_dim/4)*2*2 = embed_dim
    pos_embed = np.concatenate([np.sin(out_h), np.cos(out_h), np.sin(out_w), np.cos(out_w)], axis=1)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


class BlockMaskingGenerator:
    """Block masking generator (from eat.py)."""
    def __init__(self, input_size, mask_ratio, block_size):
        self.height, self.width = input_size
        self.num_patches = self.height * self.width
        self.mask_ratio = mask_ratio
        # Cap block size to grid dimensions for safety
        self.block_size = (min(block_size[0], self.height), min(block_size[1], self.width))

    def __call__(self, B, device):
        mask = torch.ones(B, self.height, self.width, device=device)
        num_keep = int(self.num_patches * (1 - self.mask_ratio))
        
        for b in range(B):
            current_kept = 0
            while current_kept < num_keep:
                h = torch.randint(0, self.height - self.block_size[0] + 1, (1,)).item()
                w = torch.randint(0, self.width - self.block_size[1] + 1, (1,)).item()
                if mask[b, h:h+self.block_size[0], w:w+self.block_size[1]].sum() > 0:
                    mask[b, h:h+self.block_size[0], w:w+self.block_size[1]] = 0
                current_kept = (mask[b] == 0).sum().item()
        
        mask_flat = mask.flatten(1)
        noise = torch.rand(B, self.num_patches, device=device)
        ids_shuffle = torch.argsort(mask_flat + noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        len_keep = int(self.num_patches * (1 - self.mask_ratio))
        return ids_restore, len_keep


class LayerNorm2d(nn.Module):
    """2D Layer Normalization (from eat.py)."""
    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    
    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class CNNEncoder(nn.Module):
    """CNN encoder for patch embedding (from eat.py)."""
    def __init__(self, in_chans=1, embed_dim=768, patch_size=(16, 16), use_comp: bool = False, comp_channels: int = 32):
        super().__init__()
        self.use_comp = use_comp
        self.comp = None
        if use_comp:
            self.comp = nn.Conv2d(in_chans, comp_channels, kernel_size=3, stride=1, padding=1)
            self.proj = nn.Conv2d(comp_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x):
        if self.comp is not None:
            x = self.comp(x)
        return self.proj(x)



class CNNDecoder(nn.Module):
    """CNN decoder for reconstruction (from eat.py)."""
    def __init__(self, in_dim, out_dim, depth=6, kernel_size=3):
        super().__init__()
        layers = []
        for i in range(depth - 1):
            layers.extend([
                nn.ConvTranspose2d(in_dim, in_dim, kernel_size, stride=1, padding=kernel_size//2),
                LayerNorm2d(in_dim), 
                nn.GELU()
            ])
        layers.append(nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=1, padding=kernel_size//2))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x, H, W):
        B, N, D = x.shape
        x = self.net(x.transpose(1, 2).reshape(B, D, H, W))
        return x.flatten(2).transpose(1, 2)


class SpecConverter(nn.Module):
    """GPU-based spectrogram converter (from eat.py)."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sr,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            power=2.0
        )
        self.db = torchaudio.transforms.AmplitudeToDB("power", top_db=cfg.top_db)

    def forward(self, waveform):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        
        # Convert to spectrogram
        spec = self.mel(waveform)
        spec = self.db(spec)
        
        # Force 128x128
        target_frames = 128
        if spec.shape[-1] > target_frames:
            spec = spec[:, :, :, :target_frames]
        elif spec.shape[-1] < target_frames:
            spec = torch.nn.functional.pad(spec, (0, target_frames - spec.shape[-1]))
            
        # Normalize
        spec = (spec - self.cfg.dataset_mean) / (self.cfg.dataset_std + 1e-6)
        
        return spec


def get_audio_info_safe(filepath: str) -> Optional[Tuple[int, int]]:
    """Safely get audio file information using metadata only (fast)."""
    try:
        if not os.path.exists(filepath):
            return None

        if os.path.getsize(filepath) < 1024:
            return None

        # Use torchaudio.info() which only reads metadata, not audio data (much faster)
        info = torchaudio.info(filepath)
        frames = info.num_frames
        sample_rate = info.sample_rate
        return frames, sample_rate

    except Exception as e:
        logging.debug(f"Failed to get info for {filepath}: {e}")
        return None


class EATMAE(nn.Module):
    """EAT model (from eat.py)."""
    def __init__(self, cfg, is_teacher=False):
        super().__init__()
        self.cfg = cfg
        self.is_teacher = is_teacher
        self.grid_size = (128 // cfg.patch_size[0], 128 // cfg.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        
        self.patch_embed = CNNEncoder(
            1,
            cfg.embed_dim,
            cfg.patch_size,
            use_comp=getattr(cfg, "use_comp_cnn", False),
            comp_channels=getattr(cfg, "comp_channels", 32),
        )
        self.register_buffer('pos_embed', torch.from_numpy(
            build_2d_sincos_pos_embed(cfg.embed_dim, self.grid_size)).float().unsqueeze(0), persistent=False)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            cfg.embed_dim, cfg.nhead, cfg.embed_dim*4, cfg.dropout, 'gelu', 
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.depth)
        
        if not self.is_teacher:
            self.decoder_embed = nn.Linear(cfg.embed_dim, cfg.decoder_dim)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_dim))
            self.register_buffer('dec_pos_embed', torch.from_numpy(
                build_2d_sincos_pos_embed(cfg.decoder_dim, self.grid_size)).float().unsqueeze(0), persistent=False)
            self.decoder = CNNDecoder(cfg.decoder_dim, cfg.embed_dim, 6)
        
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.cls_token, std=0.02)
        if not self.is_teacher:
            torch.nn.init.normal_(self.mask_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, LayerNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.xavier_uniform_(m.weight)

    def forward_features(self, spec):
        """Forward pass for feature extraction only."""
        x = self.patch_embed(spec).flatten(2).transpose(1, 2) + self.pos_embed[:, 1:, :]
        x_input = torch.cat((self.cls_token.expand(x.shape[0], -1, -1) + self.pos_embed[:, :1, :], x), dim=1)
        out = self.encoder(x_input)
        return out[:, 0, :]

    def forward(self, x, ids_restore=None):
        x = self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed[:, 1:, :]
        
        if not self.is_teacher and ids_restore is not None:
            ids_shuffle = torch.argsort(ids_restore, dim=1)
            len_keep = int(self.num_patches * (1 - self.cfg.mask_ratio))
            x_vis = torch.gather(x, 1, ids_shuffle[:, :len_keep].unsqueeze(-1).repeat(1, 1, x.shape[-1]))
        else:
            x_vis = x
            
        x_input = torch.cat((self.cls_token.expand(x_vis.shape[0], -1, -1) + self.pos_embed[:, :1, :], x_vis), dim=1)
        
        if self.is_teacher:
            out = x_input
            outputs = [out := mod(out) for mod in self.encoder.layers]
            avg_out = torch.stack(outputs).mean(0)
            return avg_out[:, 1:, :], avg_out[:, 0, :]
        else:
            out = self.encoder(x_input)
            x_vis_dec = self.decoder_embed(out[:, 1:, :])
            mask_tokens = self.mask_token.repeat(x_vis_dec.shape[0], ids_restore.shape[1] - x_vis_dec.shape[1], 1)
            x_full = torch.gather(
                torch.cat([x_vis_dec, mask_tokens], 1), 1, 
                ids_restore.unsqueeze(-1).repeat(1, 1, x_vis_dec.shape[-1])
            ) + self.dec_pos_embed[:, 1:, :]
            return self.decoder(x_full, self.grid_size[0], self.grid_size[1]), out[:, 0, :]


class EATMAETeacher(nn.Module):
    """EAT Teacher model (from eat.py)."""
    def __init__(self, cfg):
        super().__init__()
        # Create a student model but mark as teacher
        self.model = EATMAE(cfg, is_teacher=True)
        
        # Freeze all parameters
        for p in self.parameters():
            p.requires_grad = False
    
    def forward(self, spec):
        return self.model(spec)