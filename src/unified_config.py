"""
Unified configuration system for training/evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, Dict, Any, List
import torch


# =============================================================================
# Menu/Task Constants
# =============================================================================

MODELS = ["openSMILE", "VGGish", "CLAP", "AudioMAE", "EAT"]

CLASSIFICATION_TASKS = ["RenishawL", "VF2", "Yornew", "ColdSpray"]
ANOMALY_TASKS = ["Yornew_anomaly", "ColdSpray_anomaly"]
ALL_DOWNSTREAM_TASKS = CLASSIFICATION_TASKS + ANOMALY_TASKS


# =============================================================================
# Base Configuration
# =============================================================================

@dataclass
class UnifiedConfig:
    """Unified configuration for all models and modes."""

    # =========================================================================
    # Runtime settings
    # =========================================================================
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    use_multi_gpu: bool = False
    seed: int = 42

    # =========================================================================
    # Model settings
    # =========================================================================
    model_type: str = "AudioMAE"  # openSMILE, VGGish, CLAP, AudioMAE, EAT
    mode: str = "downstream"  # scratch, finetune, downstream, pretrain
    backbone: Optional[str] = None  # pretrained, finetuned, scratch (for downstream)

    # =========================================================================
    # Audio settings
    # =========================================================================
    sr: int = 48_000
    sample_len: int = 48_000
    n_fft: int = 2048
    win_length: int = 2048
    hop_length: int = 376
    n_mels: int = 128
    top_db: int = 80

    # Normalization Stats (AudioSet-2M)
    dataset_mean: float = -4.268
    dataset_std: float = 4.569

    # =========================================================================
    # Model architecture (fallback defaults - overridden by MODEL_PRESETS)
    # =========================================================================
    embed_dim: int = 384
    depth: int = 8
    nhead: int = 12
    decoder_dim: int = 256
    base_channels: int = 32
    patch_size: Tuple[int, int] = (16, 16)
    block_size: Tuple[int, int] = (5, 5)
    num_clones: int = 16
    mask_ratio: float = 0.6
    dropout: float = 0.0
    utter_loss_weight: float = 1.0
    use_comp_cnn: bool = False
    comp_channels: int = 32

    # =========================================================================
    # Classifier settings (downstream)
    # =========================================================================
    classifier_hidden_dim: int = 256
    classifier_dropout: float = 0.0

    # =========================================================================
    # Training settings
    # =========================================================================
    batch_size: int = 32
    num_workers: int = 32
    lr: float = 5e-4
    lr_min: float = 1e-6
    wd: float = 0.05
    epochs: int = 200
    gradient_clip: float = 1.0
    use_amp: bool = True

    # =========================================================================
    # Scheduler settings
    # =========================================================================
    scheduler_type: str = "cosine"  # cosine, plateau, step
    lr_decay_factor: float = 0.5
    lr_decay_patience: int = 5
    patience: int = 10  # Early stopping patience

    # =========================================================================
    # Logging settings
    # =========================================================================
    log_interval: int = 10
    save_interval: int = 10
    save_best_only: bool = False
    tensorboard: bool = True
    experiment_name: str = ""

    # =========================================================================
    # Teacher-student settings (for pretraining/finetuning)
    # =========================================================================
    teacher_momentum: float = 0.996
    teacher_momentum_max: float = 1.0
    frame_loss_type: str = "mse"  # mse, huber, l1
    utter_loss_type: str = "mse"

    # =========================================================================
    # Paths (will be set based on model_type and mode)
    # =========================================================================
    dataset: str = "DINOS"
    train_data_path: Optional[str] = None
    test_data_path: Optional[str] = None
    pretrained_model_path: str = ""
    backbone_model_path: Optional[str] = None  # User-selected model for downstream
    save_dir: str = ""
    log_dir: str = ""
    results_dir: str = ""
    checkpoint_dir: str = ""
    clips_per_folder: Optional[int] = None
    use_weighted_sampling: bool = False

    def __post_init__(self):
        """Set default paths based on model_type and mode."""
        self._setup_paths()
        self._validate()

    def _setup_paths(self):
        """Setup default paths based on model and mode."""
        model = self.model_type
        mode = self.mode

        # Base directories
        if model in ["AudioMAE", "EAT"]:
            if mode == "scratch":
                model_dir = f"{model}_Models_Scratch"
                results_base = f"{model}_Results_Scratch"
            elif mode == "finetune":
                model_dir = f"{model}_Models_FT"
                results_base = f"{model}_Results_FT"
            else:
                model_dir = f"{model}_Models"
                results_base = f"{model}_Results"
        else:
            model_dir = f"{model}_Models"
            results_base = f"{model}_Results"

        # Set paths
        if not self.save_dir:
            self.save_dir = f"./{model_dir}"
        if not self.log_dir:
            self.log_dir = f"./{model_dir}/logs"
        if not self.results_dir:
            self.results_dir = f"./{results_base}/{self.dataset}"
        if not self.checkpoint_dir:
            self.checkpoint_dir = self.save_dir
        if not self.experiment_name:
            self.experiment_name = f"{self.model_type}_{self.mode}"

        # Dataset paths
        if self.train_data_path is None:
            if mode in ["scratch", "finetune", "pretrain"]:
                self.train_data_path = f"./Datasets/{self.dataset}_Pretrain"
            else:
                self.train_data_path = f"./Datasets/{self.dataset}_Downstreams_train"

        if self.test_data_path is None:
            self.test_data_path = f"./Datasets/{self.dataset}_Downstreams_test"

    def _validate(self):
        """Validate configuration."""
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.epochs > 0, "epochs must be positive"
        assert self.lr > 0, "lr must be positive"
        assert 0 < self.mask_ratio < 1, "mask_ratio must be between 0 and 1"
        assert self.embed_dim % self.nhead == 0, "embed_dim must be divisible by nhead"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)

    def update(self, **kwargs):
        """Update config with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self._setup_paths()
        self._validate()

    @classmethod
    def from_menu(cls, menu_result: dict) -> "UnifiedConfig":
        """Create config from menu selection result."""
        config = create_config(
            model_type=menu_result["model_type"],
            mode=menu_result["mode"],
            device=menu_result["device"],
            use_multi_gpu=menu_result["use_multi_gpu"],
            backbone=menu_result.get("backbone"),
        )

        # Set checkpoint path for fine-tuning
        if menu_result.get("checkpoint_path"):
            config.pretrained_model_path = menu_result["checkpoint_path"]

        # Set user-selected model path for downstream
        if menu_result.get("backbone_model_path"):
            config.backbone_model_path = menu_result["backbone_model_path"]

        # Update with user-specified hyperparameters
        user_config = menu_result.get("config", {})
        config.update(
            batch_size=user_config.get("batch_size", config.batch_size),
            num_workers=user_config.get("num_workers", config.num_workers),
            lr=user_config.get("lr", config.lr),
            lr_min=user_config.get("lr_min", config.lr_min),
            epochs=user_config.get("epochs", config.epochs),
            wd=user_config.get("wd", config.wd),
            gradient_clip=user_config.get("gradient_clip", config.gradient_clip),
            seed=user_config.get("seed", config.seed),
        )

        return config


# =============================================================================
# Model-specific architectural and training presets
# =============================================================================
MODEL_PRESETS = {
    # AudioMAE configurations
    ("AudioMAE", "pretrain"): {
        "embed_dim": 768,
        "depth": 12,
        "nhead": 12,
        "decoder_dim": 512,
        "lr": 1e-4,
        "epochs": 20,
        "batch_size": 32,
        "gradient_clip": 3.0,
        "mask_ratio": 0.6,
        "utter_loss_weight": 1.0,
    },
    ("AudioMAE", "finetune"): {
        "embed_dim": 768,
        "depth": 12,
        "nhead": 12,
        "decoder_dim": 512,
        "lr": 1e-5,
        "epochs": 10,
        "batch_size": 32,
        "gradient_clip": 3.0,
        "mask_ratio": 0.6,
        "utter_loss_weight": 1.0,
    },
    ("AudioMAE", "scratch"): {
        "embed_dim": 768,
        "depth": 12,
        "nhead": 12,
        "decoder_dim": 512,
        "lr": 1e-4,
        "epochs": 20,
        "batch_size": 32,
        "gradient_clip": 3.0,
        "mask_ratio": 0.6,
        "utter_loss_weight": 1.0,
    },
    ("AudioMAE", "downstream"): {
        "lr": 5e-4,
        "epochs": 200,
        "batch_size": 32,
        "gradient_clip": 1.0,
        "classifier_hidden_dim": 256,
        "classifier_dropout": 0.0,
    },
    # EAT configurations
    ("EAT", "pretrain"): {
        "embed_dim": 768,
        "depth": 12,
        "nhead": 12,
        "decoder_dim": 512,
        "lr": 1e-4,
        "epochs": 20,
        "batch_size": 32,
        "gradient_clip": 3.0,
        "mask_ratio": 0.8,
        "utter_loss_weight": 1.0,
    },
    ("EAT", "finetune"): {
        "embed_dim": 768,
        "depth": 12,
        "nhead": 12,
        "decoder_dim": 512,
        "lr": 1e-5,
        "epochs": 10,
        "batch_size": 32,
        "gradient_clip": 3.0,
        "mask_ratio": 0.8,
        "utter_loss_weight": 1.0,
    },
    ("EAT", "scratch"): {
        "embed_dim": 768,
        "depth": 12,
        "nhead": 12,
        "decoder_dim": 512,
        "lr": 1e-4,
        "epochs": 20,
        "batch_size": 32,
        "gradient_clip": 3.0,
        "block_size": (5, 5),
        "num_clones": 16,
        "mask_ratio": 0.8,
        "utter_loss_weight": 1.0,
    },
    ("EAT", "downstream"): {
        "lr": 5e-4,
        "epochs": 200,
        "batch_size": 32,
        "gradient_clip": 1.0,
        "classifier_hidden_dim": 256,
        "classifier_dropout": 0.0,
    },
}


# =============================================================================
# Model-specific configurations
# =============================================================================

# AudioMAE constants
AUDIOMAE_MEAN = -4.2677393
AUDIOMAE_STD = 4.5689974
AUDIOMAE_TARGET_SR = 16_000
AUDIOMAE_N_MELS = 128
AUDIOMAE_N_FRAMES = 1024

# Pretrained model paths
PRETRAINED_MODEL_PATHS = {
    "AudioMAE": "gaunernst/vit_base_patch16_1024_128.audiomae_as2m_ft_as20k",
    "EAT": "worstchan/EAT-base_epoch30_pretrain",
}

# Model embedding dimensions
MODEL_EMBED_DIMS = {
    "openSMILE": 88,  # eGeMAPSv02 feature set
    "VGGish": 128,
    "CLAP": 512,
    "AudioMAE": 768,  # Pretrained model
    "EAT": 768,  # Pretrained model
}


# =============================================================================
# Anomaly Detection Configuration
# =============================================================================

ANOMALY_NORMAL_CLASSES = {
    "ColdSpray": ["1-normal_stethoscope", "5-normal_mic"],
    "Yornew": [
        "007-1.0_6.35_25.4_4000_chatter",
        "009-1.0_6.35_25.4_12000_chatter",
        "011-3.0_6.35_76.2_12000_chatter",
        "012-3.0_6.35_76.2_8000_chatter",
    ],
}


# =============================================================================
# Factory Functions
# =============================================================================

def create_config(
    model_type: str,
    mode: str,
    device: Optional[str] = None,
    use_multi_gpu: bool = False,
    backbone: Optional[str] = None,
    **kwargs,
) -> UnifiedConfig:
    """
    Create a configuration with appropriate presets.

    Args:
        model_type: Model name (openSMILE, VGGish, CLAP, AudioMAE, EAT)
        mode: Training mode (scratch, finetune, downstream, pretrain)
        device: Device string
        use_multi_gpu: Whether to use multi-GPU
        **kwargs: Additional configuration overrides

    Returns:
        UnifiedConfig with appropriate defaults
    """
    # Start with base config
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = UnifiedConfig(
        model_type=model_type,
        mode=mode,
        device=device,
        use_multi_gpu=use_multi_gpu,
        backbone=backbone,
    )

    # Apply model-specific presets
    model_presets = MODEL_PRESETS.get((model_type, mode), {})
    if model_presets:
        config.update(**model_presets)

    # For downstream, overlay architecture params from the corresponding
    # training preset so that the model is built with the correct dimensions.
    if mode == "downstream" and backbone in ("scratch", "pretrained", "finetuned"):
        _ARCH_KEYS = {
            "embed_dim", "depth", "nhead", "decoder_dim", "patch_size",
            "block_size", "num_clones", "mask_ratio", "use_comp_cnn",
            "comp_channels", "base_channels",
        }
        source_mode = "scratch" if backbone == "scratch" else "pretrain"
        source_presets = MODEL_PRESETS.get((model_type, source_mode), {})
        arch_overrides = {k: v for k, v in source_presets.items() if k in _ARCH_KEYS}
        if arch_overrides:
            config.update(**arch_overrides)

    # Apply user overrides
    if kwargs:
        config.update(**kwargs)

    return config


def get_backbone_path(model_type: str, backbone: str) -> str:
    """
    Get the path to the backbone model.

    Args:
        model_type: Model name
        backbone: Backbone type (pretrained, finetuned, scratch)

    Returns:
        Path or HuggingFace model ID
    """
    if backbone == "pretrained":
        return PRETRAINED_MODEL_PATHS.get(model_type, "")
    elif backbone == "finetuned":
        return f"./{model_type}_Models_FT/{model_type}_DINOS_best.pth"
    elif backbone == "scratch":
        return f"./{model_type}_Models_Scratch/{model_type}_DINOS_best.pth"
    return ""


def get_embed_dim(model_type: str, mode: Optional[str] = None) -> int:
    """
    Get the embedding dimension for a model.

    Args:
        model_type: Model name
        mode: Training mode (scratch, finetune, pretrain, downstream)

    Returns:
        Embedding dimension
    """
    # Try to get from MODEL_PRESETS first if mode is provided
    if mode:
        presets = MODEL_PRESETS.get((model_type, mode), {})
        if "embed_dim" in presets:
            return presets["embed_dim"]

    # Fallback to MODEL_EMBED_DIMS
    return MODEL_EMBED_DIMS.get(model_type, 768)