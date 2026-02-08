"""
Unified pretraining entrypoints.
"""
from __future__ import annotations

from typing import List, Optional

from ..unified_config import UnifiedConfig


def _sync_attr(dst, src, dst_name: str, src_name: str) -> None:
    if hasattr(dst, dst_name) and hasattr(src, src_name):
        setattr(dst, dst_name, getattr(src, src_name))


def _build_audiomae_scratch_config(config: UnifiedConfig):
    from ..pipelines.AudioMAE_scratch import AudioMAE1SecConfig

    cfg = AudioMAE1SecConfig(finetune=False)
    _sync_attr(cfg, config, "dataset", "dataset")
    _sync_attr(cfg, config, "train_data_path", "train_data_path")
    _sync_attr(cfg, config, "log_dir", "log_dir")
    _sync_attr(cfg, config, "checkpoint_dir", "checkpoint_dir")
    _sync_attr(cfg, config, "sr", "sr")
    _sync_attr(cfg, config, "sample_len", "sample_len")
    _sync_attr(cfg, config, "target_sr", "sr")
    _sync_attr(cfg, config, "num_mel_bins", "n_mels")
    _sync_attr(cfg, config, "embed_dim", "embed_dim")
    _sync_attr(cfg, config, "depth", "depth")
    _sync_attr(cfg, config, "num_heads", "nhead")
    _sync_attr(cfg, config, "mask_ratio", "mask_ratio")
    _sync_attr(cfg, config, "decoder_embed_dim", "decoder_dim")
    _sync_attr(cfg, config, "batch_size", "batch_size")
    _sync_attr(cfg, config, "epochs", "epochs")
    _sync_attr(cfg, config, "lr", "lr")
    _sync_attr(cfg, config, "lr_min", "lr_min")
    _sync_attr(cfg, config, "wd", "wd")
    _sync_attr(cfg, config, "gradient_clip", "gradient_clip")
    _sync_attr(cfg, config, "num_workers", "num_workers")
    _sync_attr(cfg, config, "seed", "seed")
    _sync_attr(cfg, config, "log_interval", "log_interval")
    _sync_attr(cfg, config, "save_interval", "save_interval")

    if hasattr(cfg, "patch_size") and isinstance(config.patch_size, tuple):
        cfg.patch_size = config.patch_size[0]

    return cfg


def run_pretraining(
    config: UnifiedConfig,
    datasets: Optional[List[str]] = None,
) -> None:
    """
    Dispatch pretraining based on model_type.
    """
    if config.model_type == "AudioMAE":
        from ..pipelines.AudioMAE_scratch import run_pretrain

        cfg = _build_audiomae_scratch_config(config)
        run_pretrain(
            cfg=cfg,
            datasets=datasets or ["DINOS"],
            device=config.device,
            use_multi_gpu=config.use_multi_gpu,
        )
        return

    if config.model_type == "EAT":
        from ..pipelines.EAT_pretraining import run_pretrain

        run_pretrain(
            cfg=config,
            datasets=datasets,
            device=config.device,
            use_multi_gpu=config.use_multi_gpu,
        )
        return

    raise ValueError(f"Pretraining not supported for model_type={config.model_type}")
