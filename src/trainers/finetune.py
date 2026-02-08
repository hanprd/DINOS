"""
Unified finetuning entrypoints.
"""
from __future__ import annotations

from typing import List, Optional

from ..unified_config import UnifiedConfig


def _sync_attr(dst, src, dst_name: str, src_name: str) -> None:
    if hasattr(dst, dst_name) and hasattr(src, src_name):
        setattr(dst, dst_name, getattr(src, src_name))


def _build_audiomae_finetune_config(config: UnifiedConfig):
    from ..pipelines.AudioMAE_finetuning import AudioMAEFTConfig

    cfg = AudioMAEFTConfig()
    _sync_attr(cfg, config, "dataset", "dataset")
    _sync_attr(cfg, config, "train_data_path", "train_data_path")
    _sync_attr(cfg, config, "log_dir", "log_dir")
    _sync_attr(cfg, config, "checkpoint_dir", "checkpoint_dir")
    _sync_attr(cfg, config, "sr", "sr")
    _sync_attr(cfg, config, "sample_len", "sample_len")
    _sync_attr(cfg, config, "num_mel_bins", "n_mels")
    _sync_attr(cfg, config, "batch_size", "batch_size")
    _sync_attr(cfg, config, "epochs", "epochs")
    _sync_attr(cfg, config, "lr", "lr")
    _sync_attr(cfg, config, "lr_min", "lr_min")
    _sync_attr(cfg, config, "wd", "wd")
    _sync_attr(cfg, config, "gradient_clip", "gradient_clip")
    _sync_attr(cfg, config, "num_workers", "num_workers")
    _sync_attr(cfg, config, "seed", "seed")
    _sync_attr(cfg, config, "tensorboard", "tensorboard")
    _sync_attr(cfg, config, "log_interval", "log_interval")

    if getattr(config, "pretrained_model_path", ""):
        cfg.model_name = config.pretrained_model_path

    return cfg


def run_finetuning(
    config: UnifiedConfig,
    datasets: Optional[List[str]] = None,
) -> None:
    """
    Dispatch finetuning based on model_type.
    """
    if config.model_type == "AudioMAE":
        from ..pipelines.AudioMAE_finetuning import run_finetune

        cfg = _build_audiomae_finetune_config(config)
        run_finetune(
            cfg=cfg,
            datasets=datasets,
            device=config.device,
            use_multi_gpu=config.use_multi_gpu,
        )
        return

    if config.model_type == "EAT":
        from ..pipelines.EAT_finetuning import run_finetune

        run_finetune(
            cfg=config,
            datasets=datasets,
            device=config.device,
            use_multi_gpu=config.use_multi_gpu,
            use_hf_model=True,
        )
        return

    if config.model_type == "IMPACT":
        from ..pipelines.IMPACT_finetuning import run_finetune

        run_finetune(
            cfg=config,
            datasets=datasets,
            device=config.device,
            use_multi_gpu=config.use_multi_gpu,
        )
        return

    raise ValueError(f"Finetuning not supported for model_type={config.model_type}")
