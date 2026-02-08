# Unified Audio Representation Learning Framework

A unified training and evaluation framework for audio representation models, supporting pretraining, fine-tuning, and downstream evaluation across multiple architectures.

## Supported Models

| Model | Type | Pretrain | Fine-tune | Downstream |
|-------|------|:--------:|:---------:|:----------:|
| **AudioMAE** | Masked Autoencoder | O | O | O |
| **EAT** | Efficient Audio Transformer | O | O | O |
| **CLAP** | Contrastive Language-Audio Pretraining | - | - | O |
| **VGGish** | CNN-based Audio Embedding | - | - | O |
| **openSMILE** | Acoustic Feature Extraction (eGeMAPSv02) | - | - | O |

## Project Structure

```
.
├── run_menu.py                  # Main entry point (interactive CLI)
├── accelerate_config.yaml       # Multi-GPU configuration
├── data_downloader.sh           # Dataset downloader (Kaggle)
└── src/
    ├── unified_config.py        # Centralized configuration management
    ├── menu.py                  # Interactive menu system
    ├── utils.py                 # Audio processing & model utilities
    ├── trainers/
    │   ├── pretrain.py          # Pretraining dispatcher
    │   ├── finetune.py          # Fine-tuning dispatcher
    │   ├── downstream.py        # Downstream task training
    │   └── evaluator.py         # Evaluation & metrics
    └── pipelines/
        ├── EAT_pretraining.py
        ├── EAT_finetuning.py
        ├── AudioMAE_scratch.py
        └── AudioMAE_finetuning.py
```

## Installation

### 1. Environment Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

uv venv venv --python 3.11
source venv/bin/activate
uv pip install pip setuptools wheel
```

### 2. Install Dependencies

```bash
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
uv pip install numpy pandas matplotlib scikit-learn tqdm timm
uv pip install tensorboard pyyaml umap-learn soundfile librosa seaborn accelerate huggingface-hub safetensors
uv pip install kagglehub
uv pip install opensmile resampy laion_clap
uv pip install plotly dash dash-bootstrap-components
```

### 3. Download Pre-trained Weights

The following pre-trained weights are required for fine-tuning and downstream evaluation:

- **CLAP**: [630k-audioset-best.pt](https://huggingface.co/lukewys/laion_clap/blob/main/630k-audioset-best.pt)
- **AudioMAE**: [vit_base_patch16_1024_128.audiomae_as2m](https://huggingface.co/gaunernst/vit_base_patch16_1024_128.audiomae_as2m)
- **EAT**: [EAT-base_epoch30_pretrain](https://huggingface.co/worstchan/EAT-base_epoch30_pretrain)

### 4. Prepare Dataset

Requires the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) with API credentials configured (`~/.kaggle/kaggle.json`).

```bash
./data_downloader.sh
```

The dataset directory should follow this structure:

```
Datasets/
├── DINOS_Pretrain/           # Pretraining audio data
├── DINOS_Downstreams_train/  # Downstream training data
│   ├── RenishawL/
│   ├── VF2/
│   ├── Yornew/
│   ├── ColdSpray/
│   ├── Yornew_anomaly/
│   └── ColdSpray_anomaly/
└── DINOS_Downstreams_test/   # Downstream test data
    └── (same structure as above)
```

## Usage

### Interactive Menu

The main entry point provides a step-by-step interactive menu:

```bash
python run_menu.py
```

The menu guides you through:
1. **Device selection** - CPU / Single GPU / Multi-GPU
2. **Model selection** - AudioMAE, EAT, CLAP, VGGish, openSMILE
3. **Mode selection** - Pretrain / Fine-tune / Downstream
4. **Action selection** - Train / Test / Both (downstream mode)
5. **Task selection** - Choose downstream datasets
6. **Hyperparameter configuration** - Learning rate, epochs, batch size, etc.

### Resume from Saved Config

Menu selections are automatically saved to `cache/`. To rerun with a saved configuration:

```bash
python run_menu.py --menu-config cache/menu_config_YYYYMMDD_HHMMSS.json
```

### Multi-GPU Training

Multi-GPU training is supported via Hugging Face Accelerate for pretraining and fine-tuning. When multi-GPU is selected in the menu, the framework automatically relaunches with Accelerate:

```bash
accelerate launch --config_file accelerate_config.yaml run_menu.py
```

Edit `accelerate_config.yaml` to adjust the number of GPUs and other distributed settings.

## Training Modes

### Pretraining (from scratch)

Train AudioMAE or EAT from scratch on the pretraining dataset using masked reconstruction objectives.

- **AudioMAE**: Random patch masking (60% ratio)
- **EAT**: Block masking strategy (80% ratio, 4x4 blocks)

### Fine-tuning

Fine-tune pretrained AudioMAE or EAT models on the pretraining dataset to adapt representations.

### Downstream Evaluation

Evaluate any model's representations on classification and anomaly detection tasks:

- **Classification**: RenishawL, VF2, Yornew, ColdSpray
- **Anomaly Detection**: Yornew_anomaly, ColdSpray_anomaly

A linear classifier (MLP) is trained on frozen features for classification tasks. An autoencoder is used for anomaly detection, trained on normal samples and detecting anomalies via reconstruction error.

## Results

After evaluation, results are saved to `[Model]_Results/` directories:

- Per-class metrics (accuracy, precision, recall, F1, MCC)
- Confusion matrices (CSV)
- ROC-AUC scores
- Summary CSV aggregating all tasks

## Requirements

- Python 3.11+
- CUDA 12.9+ (for GPU training)
- PyTorch 2.8.0, torchvision 0.23.0, torchaudio 2.8.0
- numpy, pandas, matplotlib, scikit-learn, tqdm, timm
- tensorboard, pyyaml, umap-learn, soundfile, librosa, seaborn
- accelerate, huggingface-hub, safetensors
- kagglehub, opensmile, resampy, laion_clap
- plotly, dash, dash-bootstrap-components
