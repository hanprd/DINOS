"""
Interactive menu system for unified training/evaluation.
"""
from __future__ import annotations

import torch
from typing import Tuple, List, Optional
import os
from pathlib import Path

from .unified_config import (
    MODELS,
    CLASSIFICATION_TASKS,
    ANOMALY_TASKS,
    ALL_DOWNSTREAM_TASKS,
    ANOMALY_NORMAL_CLASSES,
    PRETRAINED_MODEL_PATHS,
)


# =============================================================================
# Device Selection
# =============================================================================

def select_device() -> Tuple[str, bool]:
    """
    Interactive device selection.

    Returns:
        Tuple of (device_string, use_multi_gpu)
        - ("cpu", False) for CPU
        - ("cuda:X", False) for single GPU
        - ("cuda", True) for multi-GPU with Accelerate
    """
    print("\n" + "=" * 50)
    print("Device Selection")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("CUDA not available. Using CPU.")
        return "cpu", False

    num_gpus = torch.cuda.device_count()
    print(f"CUDA available. {num_gpus} GPU(s) detected.")

    print("\n1. CPU")
    print("2. Single GPU")
    if num_gpus > 1:
        print("3. Multi GPU (use all GPUs)")

    while True:
        try:
            choice = input("\nSelect device [2]: ").strip()
            if choice == "":
                choice = "2"
            choice = int(choice)

            if choice == 1:
                return "cpu", False
            elif choice == 2:
                if num_gpus > 1:
                    return _select_single_gpu(num_gpus)
                return "cuda:0", False
            elif choice == 3 and num_gpus > 1:
                return "cuda", True
            else:
                print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


def _select_single_gpu(num_gpus: int) -> Tuple[str, bool]:
    """Select a specific GPU when multiple are available."""
    print("\nAvailable GPUs:")
    for i in range(num_gpus):
        print(f"  {i}: {torch.cuda.get_device_name(i)}")

    while True:
        try:
            gpu_input = input(f"\nSelect GPU [0]: ").strip()
            if gpu_input == "":
                gpu_id = 0
            else:
                gpu_id = int(gpu_input)

            if 0 <= gpu_id < num_gpus:
                return f"cuda:{gpu_id}", False
            print(f"Invalid GPU ID. Please select 0-{num_gpus - 1}.")
        except ValueError:
            print("Please enter a valid number.")


# =============================================================================
# Model Selection
# =============================================================================

def select_model() -> str:
    """
    Interactive model selection.

    Returns:
        Model name string: 'openSMILE', 'VGGish', 'CLAP', 'AudioMAE', 'EAT'
    """
    print("\n" + "=" * 50)
    print("Model Selection")
    print("=" * 50)

    for i, model in enumerate(MODELS, 1):
        print(f"{i}. {model}")

    while True:
        try:
            choice = input("\nSelect model: ").strip()
            choice = int(choice)

            if 1 <= choice <= len(MODELS):
                selected = MODELS[choice - 1]
                print(f"Selected: {selected}")
                return selected
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


# =============================================================================
# Mode Selection
# =============================================================================

def select_mode(model_type: str) -> str:
    """
    Interactive mode selection based on model type.

    Args:
        model_type: Selected model name

    Returns:
        Mode string: 'scratch', 'finetune', 'downstream', or 'pretrain'
    """
    print("\n" + "=" * 50)
    print("Mode Selection")
    print("=" * 50)

    # openSMILE, VGGish, CLAP only support downstream
    if model_type in ['openSMILE', 'VGGish', 'CLAP']:
        print(f"{model_type} supports downstream tasks only.")
        return "downstream"

    # AudioMAE, EAT: scratch, finetune, downstream
    if model_type in ['AudioMAE', 'EAT']:
        print("1. From Scratch Training (train lightweight model on DINOS)")
        print("2. Fine-tuning (fine-tune pretrained model on DINOS)")
        print("3. Downstream Tasks")

        while True:
            try:
                choice = input("\nSelect mode: ").strip()
                choice = int(choice)

                if choice == 1:
                    return "scratch"
                elif choice == 2:
                    return "finetune"
                elif choice == 3:
                    return "downstream"
                print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

    return "downstream"



# =============================================================================
# Model File Selection
# =============================================================================

def select_model_file(model_dir: Path, patterns: List[str]) -> Optional[str]:
    """
    Select a specific model file from a directory.

    Args:
        model_dir: Directory containing model files
        patterns: List of glob patterns to search for

    Returns:
        Path to selected model file, or None
    """
    if not model_dir.exists():
        print(f"Directory not found: {model_dir}")
        return None

    # Find all matching files
    model_files = []
    for pattern in patterns:
        model_files.extend(list(model_dir.glob(pattern)))

    # Remove duplicates and sort alphabetically by name
    model_files = sorted(set(model_files), key=lambda p: p.name.lower())

    if not model_files:
        print(f"No model files found in {model_dir}")
        print(f"Searched patterns: {', '.join(patterns)}")
        return None

    print(f"\nAvailable models ({len(model_files)}):")
    for i, model_file in enumerate(model_files, 1):
        print(f"  {i}. {model_file.name}")

    while True:
        try:
            choice = input(f"\nSelect model [1]: ").strip()
            if choice == "":
                choice = "1"
            choice = int(choice)

            if 1 <= choice <= len(model_files):
                selected_path = str(model_files[choice - 1])
                print(f"Selected: {selected_path}")
                return selected_path
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


# =============================================================================
# Backbone Selection (for AudioMAE/EAT downstream)
# =============================================================================

def select_backbone(model_type: str, mode: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Select backbone model for downstream tasks.

    Args:
        model_type: Selected model name
        mode: Selected mode

    Returns:
        Tuple of (backbone_type, model_path)
        - backbone_type: 'pretrained', 'finetuned', 'scratch', or None
        - model_path: Path to specific model file, or None
    """
    # Only applicable for AudioMAE/EAT downstream
    if mode != "downstream" or model_type not in ['AudioMAE', 'EAT']:
        return None, None

    print("\n" + "=" * 50)
    print("Backbone Selection")
    print("=" * 50)

    pretrained_path = PRETRAINED_MODEL_PATHS.get(model_type, "N/A")

    print(f"1. Pretrained Model ({pretrained_path})")
    print(f"2. Fine-tuned Model (./{model_type}_Models_FT/)")
    print(f"3. From Scratch Model (./{model_type}_Models_Scratch/)")

    while True:
        try:
            choice = input("\nSelect backbone [1]: ").strip()
            if choice == "":
                choice = "1"
            choice = int(choice)

            if choice == 1:
                return "pretrained", None
            elif choice == 2:
                # Select specific finetuned model
                model_dir = Path(f"./{model_type}_Models_FT")
                patterns = [
                    f"{model_type}_DINOS_best.pth",
                    f"{model_type.lower()}_*_epoch_*.pth",
                    f"{model_type.lower()}_*_best*.pth",
                    "*.pth",
                ]
                model_path = select_model_file(model_dir, patterns)
                return "finetuned", model_path
            elif choice == 3:
                # Select specific scratch model
                model_dir = Path(f"./{model_type}_Models_Scratch")
                patterns = [
                    f"{model_type}_DINOS_best.pth",
                    f"{model_type.lower()}_Scratch_pretrain_epoch_*.pth",
                    "*.pth",
                ]
                model_path = select_model_file(model_dir, patterns)
                return "scratch", model_path
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")



# =============================================================================
# Action Selection (Train/Test)
# =============================================================================

def select_action(mode: str) -> str:
    """
    Select action: train, test, or both.

    Args:
        mode: Selected mode

    Returns:
        Action string: 'train', 'test', or 'both'
    """
    # Scratch, finetune, pretrain are train-only
    if mode in ['scratch', 'finetune', 'pretrain']:
        return "train"

    print("\n" + "=" * 50)
    print("Action Selection")
    print("=" * 50)

    print("1. Train")
    print("2. Test (evaluation)")
    print("3. Train + Test")

    while True:
        try:
            choice = input("\nSelect action [1]: ").strip()
            if choice == "":
                choice = "1"
            choice = int(choice)

            if choice == 1:
                return "train"
            elif choice == 2:
                return "test"
            elif choice == 3:
                return "both"
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


# =============================================================================
# Downstream Task Selection
# =============================================================================

def select_tasks(mode: str) -> List[str]:
    """
    Select downstream tasks to run.

    Args:
        mode: Selected mode

    Returns:
        List of task names
    """
    # Non-downstream modes train on DINOS only
    if mode in ['scratch', 'finetune', 'pretrain']:
        print("\nTraining on DINOS dataset.")
        return ["DINOS"]

    print("\n" + "=" * 50)
    print("Downstream Task Selection")
    print("=" * 50)

    print("Classification tasks:")
    for i, task in enumerate(CLASSIFICATION_TASKS, 1):
        print(f"  {i}. {task}")

    print("\nAnomaly detection tasks:")
    for i, task in enumerate(ANOMALY_TASKS, len(CLASSIFICATION_TASKS) + 1):
        display = task.replace("_anomaly", "") + " (anomaly)"
        print(f"  {i}. {display}")

    print(f"\n{len(ALL_DOWNSTREAM_TASKS) + 1}. Classification only (1-{len(CLASSIFICATION_TASKS)})")
    print(f"{len(ALL_DOWNSTREAM_TASKS) + 2}. Anomaly detection only ({len(CLASSIFICATION_TASKS) + 1}-{len(ALL_DOWNSTREAM_TASKS)})")
    print(f"{len(ALL_DOWNSTREAM_TASKS) + 3}. All tasks")

    while True:
        try:
            choice = input("\nSelect task(s) [9]: ").strip()
            if choice == "":
                choice = "9"
            choice = int(choice)

            if 1 <= choice <= len(ALL_DOWNSTREAM_TASKS):
                selected = [ALL_DOWNSTREAM_TASKS[choice - 1]]
                print(f"Selected: {selected}")
                return selected
            elif choice == len(ALL_DOWNSTREAM_TASKS) + 1:  # Classification only
                print(f"Selected: {CLASSIFICATION_TASKS}")
                return CLASSIFICATION_TASKS.copy()
            elif choice == len(ALL_DOWNSTREAM_TASKS) + 2:  # Anomaly only
                print(f"Selected: {ANOMALY_TASKS}")
                return ANOMALY_TASKS.copy()
            elif choice == len(ALL_DOWNSTREAM_TASKS) + 3:  # All
                print(f"Selected: {ALL_DOWNSTREAM_TASKS}")
                return ALL_DOWNSTREAM_TASKS.copy()

            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")


# =============================================================================
# Hyperparameter Configuration
# =============================================================================

def configure_hyperparams(model_type: str, mode: str, action: str) -> dict:
    """
    Configure hyperparameters with interactive prompts.

    Args:
        model_type: Selected model type
        mode: Selected mode
        action: Selected action

    Returns:
        Dictionary of hyperparameters
    """
    # Get defaults from MODEL_PRESETS
    from .unified_config import MODEL_PRESETS
    presets = MODEL_PRESETS.get((model_type, mode), {})

    # Fallback defaults if not in MODEL_PRESETS
    default_lr = presets.get("lr", 5e-4)
    default_epochs = presets.get("epochs", 200)

    config = {
        "batch_size": presets.get("batch_size", 32),
        "num_workers": 32,
        "lr": default_lr,
        "lr_min": 1e-6,
        "epochs": default_epochs,
        "wd": 0.05,
        "gradient_clip": presets.get("gradient_clip", 1.0),
        "seed": 42,
    }

    # Test-only doesn't need training hyperparameters
    if action == "test":
        print("\nTest mode - using default configuration.")
        return config

    print("\n" + "=" * 50)
    print("Hyperparameter Configuration")
    print("=" * 50)
    print("Press Enter to use default values.\n")

    # Batch size
    config["batch_size"] = _get_int_input(
        f"Batch size [{config['batch_size']}]: ",
        config["batch_size"]
    )

    # Num workers
    config["num_workers"] = _get_int_input(
        f"Num workers [{config['num_workers']}]: ",
        config["num_workers"]
    )

    # Learning rate
    config["lr"] = _get_float_input(
        f"Learning rate [{config['lr']:.0e}]: ",
        config["lr"]
    )

    # Min learning rate
    config["lr_min"] = _get_float_input(
        f"Cosine scheduler min LR [{config['lr_min']:.0e}]: ",
        config["lr_min"]
    )

    # Epochs
    config["epochs"] = _get_int_input(
        f"Epochs [{config['epochs']}]: ",
        config["epochs"]
    )

    print("\n" + "-" * 50)
    print("Configuration summary:")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Num workers: {config['num_workers']}")
    print(f"  Learning rate: {config['lr']:.0e}")
    print(f"  Min LR: {config['lr_min']:.0e}")
    print(f"  Epochs: {config['epochs']}")
    print("-" * 50)

    return config


def _get_int_input(prompt: str, default: int) -> int:
    """Get integer input with default value."""
    while True:
        try:
            value = input(prompt).strip()
            if value == "":
                return default
            return int(value)
        except ValueError:
            print("Please enter a valid integer.")


def _get_float_input(prompt: str, default: float) -> float:
    """Get float input with default value."""
    while True:
        try:
            value = input(prompt).strip()
            if value == "":
                return default
            return float(value)
        except ValueError:
            print("Please enter a valid number.")


# =============================================================================
# Summary and Confirmation
# =============================================================================

def print_summary(
    device: str,
    use_multi_gpu: bool,
    model_type: str,
    mode: str,
    backbone: Optional[str],
    action: str,
    tasks: List[str],
    config: dict,
    backbone_model_path: Optional[str] = None,
) -> bool:
    """
    Print configuration summary and ask for confirmation.

    Returns:
        True if user confirms, False otherwise
    """
    print("\n" + "=" * 50)
    print("Configuration Summary")
    print("=" * 50)

    # Device
    if use_multi_gpu:
        print(f"Device: Multi-GPU (Accelerate)")
    else:
        print(f"Device: {device}")

    # Model and mode
    print(f"Model: {model_type}")
    print(f"Mode: {mode}")

    # Backbone (if applicable)
    if backbone:
        print(f"Backbone: {backbone}")

    # Selected model path (if applicable)
    if backbone_model_path:
        # Show just the filename for cleaner output
        model_name = Path(backbone_model_path).name
        print(f"Selected Model: {model_name}")

    # Action and tasks
    print(f"Action: {action}")
    print(f"Tasks: {', '.join(tasks)}")

    # Hyperparameters (if training)
    if action in ['train', 'both']:
        print(f"\nHyperparameters:")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Learning rate: {config['lr']:.0e}")
        print(f"  Epochs: {config['epochs']}")

    print("=" * 50)

    while True:
        confirm = input("\nProceed with this configuration? [Y/n]: ").strip().lower()
        if confirm in ["", "y", "yes"]:
            return True
        elif confirm in ["n", "no"]:
            return False
        print("Please enter 'y' or 'n'.")


# =============================================================================
# Full Menu Flow
# =============================================================================

def run_menu() -> Optional[dict]:
    """
    Run the complete interactive menu flow.

    Returns:
        Dictionary with all configuration options, or None if cancelled
    """
    print("\n" + "=" * 50)
    print("  Unified Training System")
    print("=" * 50)

    # 1. Device selection
    device, use_multi_gpu = select_device()

    # 2. Model selection
    model_type = select_model()

    # 3. Mode selection
    mode = select_mode(model_type)

    # 4. Checkpoint selection
    checkpoint_path = None

    # 5. Backbone/Model selection (for downstream tasks)
    backbone = None
    backbone_model_path = None

    if mode == "downstream":
        if model_type in ['AudioMAE', 'EAT']:
            # AudioMAE/EAT: select backbone type and specific model
            backbone, backbone_model_path = select_backbone(model_type, mode)

    # 6. Action selection (train/test/both)
    action = select_action(mode)

    # 7. Task selection
    tasks = select_tasks(mode)

    # 8. Hyperparameter configuration
    config = configure_hyperparams(model_type, mode, action)

    # 9. Summary and confirmation
    if not print_summary(device, use_multi_gpu, model_type, mode, backbone, action, tasks, config, backbone_model_path):
        print("\nConfiguration cancelled.")
        return None

    return {
        "device": device,
        "use_multi_gpu": use_multi_gpu,
        "model_type": model_type,
        "mode": mode,
        "backbone": backbone,
        "backbone_model_path": backbone_model_path,
        "checkpoint_path": checkpoint_path,
        "action": action,
        "tasks": tasks,
        "config": config,
    }


if __name__ == "__main__":
    # Test menu
    result = run_menu()
    if result:
        print("\nFinal configuration:")
        for k, v in result.items():
            print(f"  {k}: {v}")
