from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from src.menu import run_menu
from src.unified_config import UnifiedConfig
from src.trainers import (
    run_downstream_training,
    run_downstream_evaluation,
    run_pretraining,
    run_finetuning,
)


# =============================================================================
# Utilities
# =============================================================================

def _is_accelerate_launch() -> bool:
    return any(k in os.environ for k in ["ACCELERATE_PROCESS_ID", "LOCAL_RANK", "RANK", "WORLD_SIZE"])


def _save_menu_config(menu_result: dict) -> Path:
    cache_dir = Path("./cache")
    cache_dir.mkdir(exist_ok=True, parents=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = cache_dir / f"menu_config_{ts}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(menu_result, f, indent=2)
    return out_path


def _warn_single_gpu(reason: str) -> None:
    print(
        f"[WARN] {reason} does not support true multi-GPU. "
        "Proceeding with single-GPU/CPU execution."
    )


def _apply_downstream_paths(config: UnifiedConfig) -> None:
    if config.mode != "downstream":
        return

    model = config.model_type

    if model in ["openSMILE", "VGGish", "CLAP"]:
        config.save_dir = f"./{model}_Models"
        config.log_dir = f"./{model}_Models/logs"
        config.results_dir = f"./{model}_Results"
        config.checkpoint_dir = config.save_dir
        return

    if model == "AudioMAE":
        if config.backbone == "finetuned":
            config.save_dir = "./AudioMAE_Models_FT"
            config.log_dir = "./AudioMAE_Models_FT/logs"
            config.results_dir = f"./AudioMAE_Results_FT/{config.dataset}"
        elif config.backbone == "scratch":
            config.save_dir = "./AudioMAE_Models_Scratch"
            config.log_dir = "./AudioMAE_Models_Scratch/logs"
            config.results_dir = f"./AudioMAE_Results_Scratch/{config.dataset}"
        else:
            config.save_dir = "./AudioMAE_Models"
            config.log_dir = "./AudioMAE_Models/logs"
            config.results_dir = f"./AudioMAE_Results/{config.dataset}"
        config.checkpoint_dir = config.save_dir
        return

    if model == "EAT":
        if config.backbone == "finetuned":
            config.save_dir = "./EAT_Models_FT"
            config.log_dir = "./EAT_Models_FT/logs"
            config.results_dir = f"./EAT_Results_FT/{config.dataset}"
        elif config.backbone == "scratch":
            config.save_dir = "./EAT_Models_Scratch"
            config.log_dir = "./EAT_Models_Scratch/logs"
            config.results_dir = f"./EAT_Results_Scratch/{config.dataset}"
        else:
            config.save_dir = "./EAT_Models"
            config.log_dir = "./EAT_Models/logs"
            config.results_dir = f"./EAT_Results/{config.dataset}"
        config.checkpoint_dir = config.save_dir
        return


def dispatch(menu_result: dict) -> None:
    mode = menu_result["mode"]
    action = menu_result["action"]
    tasks = menu_result["tasks"]

    config = UnifiedConfig.from_menu(menu_result)
    _apply_downstream_paths(config)

    if mode in ["scratch", "pretrain"]:
        run_pretraining(config, datasets=tasks)
        return

    if mode == "finetune":
        run_finetuning(config, datasets=tasks)
        return

    if mode == "downstream":
        if action in ["train", "both"]:
            run_downstream_training(tasks, config)
        if action in ["test", "both"]:
            run_downstream_evaluation(tasks, config)
        return

    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified interactive menu runner")
    parser.add_argument("--menu-config", type=str, default=None, help="Path to saved menu config JSON")
    args = parser.parse_args()

    if args.menu_config:
        with open(args.menu_config, "r", encoding="utf-8") as f:
            menu_result = json.load(f)
    else:
        menu_result = run_menu()
        if menu_result is None:
            return

    # Multi-GPU relaunch via Accelerate if needed (and not already launched)
    if menu_result.get("use_multi_gpu") and not _is_accelerate_launch():
        model_type = menu_result.get("model_type")
        mode = menu_result.get("mode")
        if mode == "downstream" or model_type in ["openSMILE", "VGGish", "CLAP"]:
            _warn_single_gpu(model_type if mode != "downstream" else f"{model_type} downstream")
            menu_result["use_multi_gpu"] = False
        else:
            config_path = _save_menu_config(menu_result)
            cmd = [
                "accelerate",
                "launch",
                "--config_file",
                "accelerate_config.yaml",
                "run_menu.py",
                "--menu-config",
                str(config_path),
            ]
            print(f"[INFO] Relaunching with Accelerate: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
            return

    dispatch(menu_result)


if __name__ == "__main__":
    main()
