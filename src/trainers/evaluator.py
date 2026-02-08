"""
Unified evaluation module for downstream tasks.
"""
from __future__ import annotations

import os
import logging
import warnings
import glob
import traceback
from pathlib import Path
from collections import defaultdict
from typing import List, Tuple, Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc,
)

from ..unified_config import (
    UnifiedConfig,
    ANOMALY_NORMAL_CLASSES,
    AUDIOMAE_TARGET_SR,
)
from ..utils import suppress_warnings, set_deterministic_mode
from .downstream import (
    DownstreamClassifier,
    SimpleAE,
    load_feature_extractor,
    preprocess_batch,
    extract_features,
    DownstreamDataset,
    setup_logging,
)

suppress_warnings()


# =============================================================================
# Evaluation Dataset
# =============================================================================

class EvalDataset(Dataset):
    """Dataset for evaluation (no augmentation)."""

    def __init__(
        self,
        root_dir: str,
        downstream: str,
        model_type: str,
        config: UnifiedConfig,
    ):
        super().__init__()
        self.config = config
        self.model_type = model_type

        self.file_list: List[Tuple[str, int, str]] = []
        self.class_to_idx: Dict[str, int] = {}

        downstream_path = Path(root_dir) / downstream
        if not downstream_path.exists():
            raise FileNotFoundError(f"Test directory not found: {downstream_path}")

        class_names = [d.name for d in sorted(downstream_path.iterdir()) if d.is_dir()]
        self.class_to_idx = {name: idx for idx, name in enumerate(sorted(class_names))}
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        for class_name in class_names:
            class_path = downstream_path / class_name
            audio_files = []
            for ext in ["*.wav", "*.flac", "*.mp3", "*.ogg"]:
                audio_files.extend(glob.glob(str(class_path / ext)))
            for fp in audio_files:
                self.file_list.append((fp, self.class_to_idx[class_name], class_name))

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

        logging.info(f"Test dataset: {len(self.file_list)} samples, {len(class_names)} classes")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fp, label, cname = self.file_list[idx]
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
                wav = wav[:self.fixed_len]  # No random crop for test
            elif T < self.fixed_len:
                wav = F.pad(wav, (0, self.fixed_len - T))

            return wav, int(label), cname, fp

        except Exception as e:
            logging.error(f"Error loading {fp}: {e}")
            return self.__getitem__((idx + 1) % len(self))


# =============================================================================
# Visualization
# =============================================================================

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
):
    """Plot and save confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True label',
        xlabel='Predicted label',
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_distribution(
    errors: np.ndarray,
    threshold: float,
    save_path: str,
):
    """Plot reconstruction error distribution for anomaly detection."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(errors, bins=50, density=True, alpha=0.7, color='blue', label='Error distribution')
    ax.axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.4f}')

    ax.set_xlabel('Reconstruction Error')
    ax.set_ylabel('Density')
    ax.set_title('Error Distribution')
    ax.legend()

    fig.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_classification(
    downstream: str,
    config: UnifiedConfig,
    feature_extractor: nn.Module,
    embed_dim: int,
    device: str,
) -> Dict[str, Any]:
    """Evaluate classification downstream task."""
    logger = logging.getLogger(__name__)

    # Create test dataset
    dataset = EvalDataset(
        root_dir=config.test_data_path,
        downstream=downstream,
        model_type=config.model_type,
        config=config,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    num_classes = len(dataset.class_to_idx)

    # Load classifier
    classifier = DownstreamClassifier(
        embed_dim=embed_dim,
        num_classes=num_classes,
        hidden_dim=config.classifier_hidden_dim,
        dropout=0.0,
    )

    checkpoint_path = Path(config.save_dir) / downstream / f"{config.model_type}_{downstream}_final.pth"
    if checkpoint_path.exists():
        classifier.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return {"downstream": downstream, "error": "Checkpoint not found"}

    classifier = classifier.to(device)
    classifier.eval()

    # Inference
    all_preds = []
    all_labels = []
    all_probs = []

    clap_truncation = getattr(config, "clap_truncation_eval", "max_length")
    clap_padding = getattr(config, "clap_padding", "max_length")

    with torch.no_grad():
        for wav, labels, _, file_paths in tqdm(dataloader, desc=f"Evaluating {downstream}"):
            preprocessed = preprocess_batch(wav, config.model_type, config, device)
            features = extract_features(
                feature_extractor,
                preprocessed,
                config.model_type,
                device,
                file_paths=file_paths,
                clap_truncation=clap_truncation,
                clap_padding=clap_padding,
            )
            logits = classifier(features)
            probs = F.softmax(logits, dim=1)

            _, predicted = logits.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate metrics
    metrics = {
        "downstream": downstream,
        "accuracy": accuracy_score(all_labels, all_preds) * 100,
        "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds) * 100,
        "precision": precision_score(all_labels, all_preds, average='macro') * 100,
        "recall": recall_score(all_labels, all_preds, average='macro') * 100,
        "f1": f1_score(all_labels, all_preds, average='macro') * 100,
        "mcc": matthews_corrcoef(all_labels, all_preds),
    }

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    class_names = [dataset.idx_to_class[i] for i in range(num_classes)]

    # Per-class metrics
    per_class_precision = precision_score(all_labels, all_preds, average=None) * 100
    per_class_recall = recall_score(all_labels, all_preds, average=None) * 100
    per_class_f1 = f1_score(all_labels, all_preds, average=None) * 100

    # Per-class sample counts and accuracy from confusion matrix
    per_class_correct = cm.diagonal()
    per_class_total = cm.sum(axis=1)

    # Save results
    results_dir = Path(config.results_dir) / downstream
    results_dir.mkdir(parents=True, exist_ok=True)

    # Plot confusion matrix
    plot_confusion_matrix(
        cm, class_names,
        str(results_dir / f"{config.model_type}_{downstream}_confusion_matrix.png"),
        title=f"{config.model_type} - {downstream}",
    )

    import csv

    # Save overall metrics to CSV
    with open(results_dir / f"{config.model_type}_{downstream}_metrics.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    # Save per-class metrics to CSV
    per_class_fieldnames = ["class", "precision", "recall", "f1", "correct", "total", "accuracy"]
    per_class_rows = []
    for i, cname in enumerate(class_names):
        per_class_rows.append({
            "class": cname,
            "precision": per_class_precision[i],
            "recall": per_class_recall[i],
            "f1": per_class_f1[i],
            "correct": int(per_class_correct[i]),
            "total": int(per_class_total[i]),
            "accuracy": per_class_correct[i] / per_class_total[i] * 100 if per_class_total[i] > 0 else 0.0,
        })
    with open(results_dir / f"{config.model_type}_{downstream}_per_class.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=per_class_fieldnames)
        writer.writeheader()
        writer.writerows(per_class_rows)

    # Save confusion matrix to CSV
    cm_path = results_dir / f"{config.model_type}_{downstream}_confusion_matrix.csv"
    with open(cm_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["true \\ predicted"] + class_names)
        for i, cname in enumerate(class_names):
            writer.writerow([cname] + [int(v) for v in cm[i]])

    logger.info(f"Results: Acc={metrics['accuracy']:.2f}%, F1={metrics['f1']:.2f}%")

    return metrics


def evaluate_anomaly(
    downstream: str,
    config: UnifiedConfig,
    feature_extractor: nn.Module,
    embed_dim: int,
    device: str,
) -> Dict[str, Any]:
    """Evaluate anomaly detection downstream task."""
    logger = logging.getLogger(__name__)

    # Get base downstream name
    base_downstream = downstream.replace("_anomaly", "")

    # Create test dataset (all samples)
    dataset = EvalDataset(
        root_dir=config.test_data_path,
        downstream=base_downstream,
        model_type=config.model_type,
        config=config,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Load autoencoder
    ae_model = SimpleAE(input_dim=embed_dim, latent_dim=64)

    checkpoint_path = Path(config.save_dir) / downstream / f"{config.model_type}_{downstream}_final.pth"
    if checkpoint_path.exists():
        ae_model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    else:
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return {"downstream": downstream, "error": "Checkpoint not found"}

    ae_model = ae_model.to(device)
    ae_model.eval()

    # Get normal class names
    normal_classes = ANOMALY_NORMAL_CLASSES.get(base_downstream, [])

    # Inference
    all_errors = []
    all_labels = []  # 0 = normal, 1 = anomaly
    all_class_names = []

    clap_truncation = getattr(config, "clap_truncation_eval", "max_length")
    clap_padding = getattr(config, "clap_padding", "max_length")

    with torch.no_grad():
        for wav, _, class_names, file_paths in tqdm(dataloader, desc=f"Evaluating {downstream}"):
            preprocessed = preprocess_batch(wav, config.model_type, config, device)
            features = extract_features(
                feature_extractor,
                preprocessed,
                config.model_type,
                device,
                file_paths=file_paths,
                clap_truncation=clap_truncation,
                clap_padding=clap_padding,
            )

            x_hat, _ = ae_model(features)
            errors = F.mse_loss(x_hat, features, reduction='none').mean(dim=1)

            for i, cname in enumerate(class_names):
                is_normal = cname in normal_classes
                all_errors.append(errors[i].cpu().item())
                all_labels.append(0 if is_normal else 1)
                all_class_names.append(cname)

    all_errors = np.array(all_errors)
    all_labels = np.array(all_labels)

    # Calculate metrics
    # Find optimal threshold using precision-recall curve
    precision, recall, thresholds = precision_recall_curve(all_labels, all_errors)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]

    # Predictions
    all_preds = (all_errors > best_threshold).astype(int)

    metrics = {
        "downstream": downstream,
        "accuracy": accuracy_score(all_labels, all_preds) * 100,
        "precision": precision_score(all_labels, all_preds) * 100,
        "recall": recall_score(all_labels, all_preds) * 100,
        "f1": f1_score(all_labels, all_preds) * 100,
        "auroc": roc_auc_score(all_labels, all_errors) * 100,
        "auprc": auc(recall, precision) * 100,
        "threshold": best_threshold,
    }

    # Save results
    results_dir = Path(config.results_dir) / downstream
    results_dir.mkdir(parents=True, exist_ok=True)

    # Plot error distribution
    plot_error_distribution(
        all_errors, best_threshold,
        str(results_dir / f"{config.model_type}_{downstream}_error_distribution.png"),
    )

    # Save metrics to CSV
    import csv
    with open(results_dir / f"{config.model_type}_{downstream}_metrics.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    # Save per-class metrics to CSV
    all_class_names_arr = np.array(all_class_names)
    unique_classes = sorted(set(all_class_names))
    per_class_fieldnames = [
        "class", "label", "total",
        "mean_error", "std_error", "min_error", "max_error",
        "predicted_anomaly", "predicted_normal",
    ]
    per_class_rows = []
    for cname in unique_classes:
        mask = all_class_names_arr == cname
        cls_errors = all_errors[mask]
        cls_preds = all_preds[mask]
        is_normal = cname in normal_classes
        per_class_rows.append({
            "class": cname,
            "label": "normal" if is_normal else "anomaly",
            "total": int(mask.sum()),
            "mean_error": float(cls_errors.mean()),
            "std_error": float(cls_errors.std()),
            "min_error": float(cls_errors.min()),
            "max_error": float(cls_errors.max()),
            "predicted_anomaly": int(cls_preds.sum()),
            "predicted_normal": int((cls_preds == 0).sum()),
        })
    with open(results_dir / f"{config.model_type}_{downstream}_per_class.csv", 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=per_class_fieldnames)
        writer.writeheader()
        writer.writerows(per_class_rows)

    # Save confusion matrix to CSV (normal/anomaly binary)
    cm = confusion_matrix(all_labels, all_preds)
    cm_labels = ["normal", "anomaly"]
    cm_path = results_dir / f"{config.model_type}_{downstream}_confusion_matrix.csv"
    with open(cm_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["true \\ predicted"] + cm_labels)
        for i, label in enumerate(cm_labels):
            writer.writerow([label] + [int(v) for v in cm[i]])

    logger.info(f"Results: AUROC={metrics['auroc']:.2f}%, F1={metrics['f1']:.2f}%")

    return metrics


# =============================================================================
# Main Entry Point
# =============================================================================

def run_downstream_evaluation(
    tasks: List[str],
    config: UnifiedConfig,
) -> List[Dict[str, Any]]:
    """
    Run downstream evaluation for specified tasks.

    Args:
        tasks: List of downstream task names
        config: Unified configuration

    Returns:
        List of result dictionaries for each task
    """
    # Setup logging
    run_name = f"{config.model_type}_eval"
    logger = setup_logging(config.log_dir, run_name)
    logger.info(f"Starting downstream evaluation: {tasks}")

    # Set seed
    set_deterministic_mode(config.seed)

    device = config.device
    logger.info(f"Using device: {device}")

    # Load feature extractor
    logger.info(f"Loading {config.model_type} feature extractor...")
    feature_extractor, embed_dim = load_feature_extractor(
        model_type=config.model_type,
        backbone=config.backbone,
        config=config,
        device=device,
    )
    logger.info(f"Feature extractor loaded. Embedding dim: {embed_dim}")

    # Evaluate each task
    results = []
    for task in tasks:
        logger.info(f"\n{'='*50}")
        logger.info(f"Evaluating: {task}")
        logger.info(f"{'='*50}")

        try:
            if task.endswith("_anomaly"):
                result = evaluate_anomaly(
                    downstream=task,
                    config=config,
                    feature_extractor=feature_extractor,
                    embed_dim=embed_dim,
                    device=device,
                )
            else:
                result = evaluate_classification(
                    downstream=task,
                    config=config,
                    feature_extractor=feature_extractor,
                    embed_dim=embed_dim,
                    device=device,
                )
            results.append(result)
            logger.info(f"Completed: {task}")

        except Exception as e:
            logger.error(f"Failed: {task} - {e}")
            traceback.print_exc()
            results.append({"downstream": task, "error": str(e)})

    # Print summary
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Summary")
    logger.info("=" * 50)
    for result in results:
        if "error" not in result:
            if "auroc" in result:
                logger.info(f"{result['downstream']}: AUROC={result['auroc']:.2f}%, F1={result['f1']:.2f}%")
            else:
                logger.info(f"{result['downstream']}: Acc={result['accuracy']:.2f}%, F1={result['f1']:.2f}%")

    # Save aggregated CSV summaries
    results_root = Path(config.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    cls_summary_rows = []
    cls_detail_rows = []
    anom_summary_rows = []
    anom_detail_rows = []

    for result in results:
        downstream = result.get("downstream", "")
        is_anomaly = downstream.endswith("_anomaly")
        error = result.get("error")

        if is_anomaly:
            anom_summary_rows.append({
                "model": config.model_type,
                "downstream": downstream,
                "auroc": result.get("auroc"),
            })
            anom_detail_rows.append({
                "model": config.model_type,
                "downstream": downstream,
                "auroc": result.get("auroc"),
                "error": error,
            })
        else:
            cls_summary_rows.append({
                "model": config.model_type,
                "downstream": downstream,
                "macro_f1": result.get("f1"),
            })
            cls_detail_rows.append({
                "model": config.model_type,
                "downstream": downstream,
                "accuracy": result.get("accuracy"),
                "precision": result.get("precision"),
                "recall": result.get("recall"),
                "f1": result.get("f1"),
                "error": error,
            })

    import csv
    if cls_summary_rows:
        with open(results_root / f"{config.model_type}_summary_classification.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "downstream", "macro_f1"])
            writer.writeheader()
            writer.writerows(cls_summary_rows)
    if cls_detail_rows:
        with open(results_root / f"{config.model_type}_detail_classification.csv", "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["model", "downstream", "accuracy", "precision", "recall", "f1", "error"],
            )
            writer.writeheader()
            writer.writerows(cls_detail_rows)
    if anom_summary_rows:
        with open(results_root / f"{config.model_type}_summary_anomaly.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "downstream", "auroc"])
            writer.writeheader()
            writer.writerows(anom_summary_rows)
    if anom_detail_rows:
        with open(results_root / f"{config.model_type}_detail_anomaly.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "downstream", "auroc", "error"])
            writer.writeheader()
            writer.writerows(anom_detail_rows)

    # Save combined per-downstream results CSV
    if results:
        all_fieldnames = [
            "model", "downstream", "task_type",
            "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc",
            "auroc", "auprc", "threshold",
            "error",
        ]
        combined_rows = []
        for result in results:
            downstream = result.get("downstream", "")
            is_anomaly = downstream.endswith("_anomaly")
            row = {
                "model": config.model_type,
                "downstream": downstream,
                "task_type": "anomaly" if is_anomaly else "classification",
            }
            for key in all_fieldnames:
                if key not in row:
                    row[key] = result.get(key, "")
            combined_rows.append(row)

        combined_path = results_root / f"{config.model_type}_downstream_results.csv"
        with open(combined_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_fieldnames)
            writer.writeheader()
            writer.writerows(combined_rows)
        logger.info(f"Combined downstream results saved to {combined_path}")

    return results
