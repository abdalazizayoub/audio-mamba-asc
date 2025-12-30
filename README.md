# Audio Mamba for Acoustic Scene Classification

**DCASE2025 - Task 1 - Audio Mamba Implementation**

[![GitHub](https://img.shields.io/badge/GitHub-audio--mamba--asc-blue)](https://github.com/abdalazizayoub/audio-mamba-asc)

---

## Overview

This repository implements **Audio Mamba (AuM)** for the DCASE2025 Challenge Task 1: Low-Complexity Acoustic Scene Classification. It integrates state-space models (Mamba) into the acoustic scene classification pipeline, replacing traditional CNN architectures with efficient bidirectional Mamba blocks.

The project consists of two main components:
- **`AUM/`**: Audio Mamba model implementation (state-space models for audio)
- **`dcase2025_task1_baseline/`**: Adapted DCASE2025 baseline training pipeline

---

## Repository Structure
```
audio-mamba-asc/
├── AUM/                           # Audio Mamba model implementation
│   └── src/
│       └── models/
│           └── mamba_models.py    # Core Mamba architecture
│
├── dcase2025_task1_baseline/      # Training pipeline (adapted from DCASE2025)
│   ├── dataset/
│   │   ├── dcase25.py            # Dataset loading and preprocessing
│   │   └── meta_data_preprocessing.py  # Audio validation utility
│   │
│   ├── models/
│   │   ├── net.py                # Audio Mamba model wrapper
│   │   └── multi_device_model.py # Multi-device fine-tuning container
│   │
│   ├── helpers/
│   │   ├── complexity.py         # MAC and parameter counting
│   │   ├── utils.py              # MixStyle augmentation
│   │   └── init.py               # Random seed initialization
│   │
│   ├── train_base.py             # Step 1: General model training
│   ├── train_device_specific.py  # Step 2: Device-specific fine-tuning
│   ├── test_only.py              # Standalone evaluation script
│   ├── requirements.txt          # Python dependencies
│   └── README.md                 # Original DCASE2025 baseline documentation
│
├── .gitignore
└── README.md                      # This file
```

---

## Key Features

### 🎯 Audio Mamba Integration
- Bidirectional Mamba blocks for temporal modeling
- Linear complexity for efficient long-sequence processing
- Custom classification head for 10-class scene recognition

### 🔧 Training Pipeline
- **Two-stage training**: General model → Device-specific fine-tuning
- Data augmentation: Time rolling, frequency/time masking, MixStyle
- Mixed-precision training support
- Weights & Biases integration for experiment tracking

### 📊 Model Configurations

| Model | Embed Dim | Depth | Parameters | MACs |
|-------|-----------|-------|------------|------|
| AuM-Small | 384 | 24 | ~23M | TBD |
| AuM-Base | 768 | 24 | ~86M | TBD |

---

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/abdalazizayoub/audio-mamba-asc.git
cd audio-mamba-asc
```

### 2. Environment Setup
```bash
# Create conda environment
conda create -n audio_mamba python=3.13
conda activate audio_mamba

# Install PyTorch (adjust for your CUDA version)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
cd dcase2025_task1_baseline
pip3 install -r requirements.txt

# Install Mamba dependencies
pip install causal-conv1d>=1.1.0
pip install mamba-ssm
```

### 3. Dataset Setup

1. Download [TAU Urban Acoustic Scenes 2022 Mobile dataset](https://zenodo.org/records/6337421)
2. Extract to your preferred location
3. Update `dataset_dir` in `dcase2025_task1_baseline/dataset/dcase25.py`:
```python
dataset_dir = "/path/to/TAU_dataset"
```

### 4. Data Validation (Recommended)
```bash
cd dcase2025_task1_baseline
python dataset/meta_data_preprocessing.py
```

This validates audio files and generates `meta_clean.csv`.

### 5. Weights & Biases
```bash
wandb login
```

---

## Training

### Step 1: Train General Model
```bash
cd dcase2025_task1_baseline

python train_base.py \
    --project_name "DCASE25_Task1" \
    --experiment_name "AUM_Small_Scratch" \
    --embed_dim 384 \
    --depth 24 \
    --patch_size "16,16" \
    --n_epochs 150 \
    --batch_size 64 \
    --lr 0.0005 \
    --precision "32"
```

**For AuM-Base**: Change `--embed_dim 768`

### Step 2: Device-Specific Fine-Tuning
```bash
python train_device_specific.py \
    --ckpt_id <wandb_run_id_from_step_1> \
    --experiment_name "AUM_Small_DeviceSpecific" \
    --embed_dim 384 \
    --depth 24 \
    --n_epochs 50 \
    --lr 0.0005
```

---

## Evaluation

### Test Best Checkpoint
```bash
python test_only.py \
    --ckpt_path "path/to/best-checkpoint.ckpt" \
    --embed_dim 384 \
    --depth 24 \
    --patch_size "16,16" \
    --batch_size 64 \
    --precision "32"
```

---

## Model Architecture

### Audio Mamba Block
```
Input Spectrogram (256 mel-bins × 1024 frames)
    ↓
Patch Embedding (16×16 patches)
    ↓
Bidirectional Mamba Blocks (depth=24)
    ↓
Global Average Pooling
    ↓
Classification Head:
    - LayerNorm
    - Dropout(0.1)
    - Linear(embed_dim → 512)
    - ReLU
    - Dropout(0.1)
    - Linear(512 → 10 classes)
```

### Key Design Choices

- **Bidirectional Mamba**: Processes sequences forward and backward for better context
- **Absolute Position Embeddings**: Disabled to avoid initialization issues
- **Patch Size**: 16×16 balances local feature capture and sequence length
- **Custom Head**: Multi-layer classifier for better scene discrimination

---

## Dataset Information

### TAU Urban Acoustic Scenes 2022

- **10 scene classes**: airport, bus, metro, metro_station, park, public_square, shopping_mall, street_pedestrian, street_traffic, tram
- **Recording devices**: 
  - Real devices: a, b, c
  - Simulated seen: s1, s2, s3
  - Simulated unseen: s4, s5, s6
- **Training subset**: 25% split (per DCASE2025 rules)
- **Audio format**: 10-second clips, 44.1 kHz, mono

---

## Complexity Constraints (DCASE2025)

⚠️ **Challenge Requirements**:
- Maximum parameters: **128 kB** (when converted to 16-bit precision)
- Maximum MACs: **30 million** for 1-second audio

⚠️ **Current Status**: Audio Mamba models exceed these limits and require optimization for official submission.

**Potential Solutions**:
- Model quantization (8-bit or lower)
- Knowledge distillation to smaller models
- Pruning redundant parameters
- Architecture modifications (reduced depth/width)

---

## Citations

### Audio Mamba (AuM)
```bibtex
@article{xie2024audio,
  title={Audio Mamba: Bidirectional State Space Model for Audio Representation Learning},
  author={Xie, Yumeng and Zhang, Shuo and Wang, Guangliang and Wang, Wenwu},
  journal={arXiv preprint arXiv:2406.03344},
  year={2024}
}
```

### Mamba Architecture
```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

### DCASE2025 Task 1 Baseline
```bibtex
@misc{dcase2025_task1_baseline,
  author = {Schmid, Florian},
  title = {DCASE2025 Task 1 Baseline System},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/CPJKU/dcase2025_task1_baseline}}
}
```

### TAU Dataset
```bibtex
@dataset{tau2022,
  author = {Heittola, Toni and Mesaros, Annamaria and Virtanen, Tuomas},
  title = {TAU Urban Acoustic Scenes 2022 Mobile, Development dataset},
  year = {2022},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.6337421}
}
```

---

## Acknowledgments

- **DCASE2025 Baseline**: Florian Schmid (Johannes Kepler University Linz)
- **Audio Mamba**: Yumeng Xie et al. ([AUM Repository](https://github.com/xieyumc/AUM))
- **Mamba Architecture**: Albert Gu and Tri Dao

---

## License

This project combines code from multiple sources. Please refer to individual component licenses:
- DCASE2025 baseline: [Original License](https://github.com/CPJKU/dcase2025_task1_baseline)
- Audio Mamba (AUM): [AUM License](https://github.com/xieyumc/AUM)

---

## Contact

**Abdalaziz Ayoub**  
📧 K12341559@students.jku.at 
🔗 [GitHub](https://github.com/abdalazizayoub)

For DCASE2025 baseline questions: Florian Schmid (florian.schmid@jku.at)

---

## Future Work

- [ ] Implement quantization for complexity compliance
- [ ] Knowledge distillation experiments
- [ ] RoPE (Rotary Position Embeddings) integration
- [ ] Benchmark against CP-Mobile baseline
- [ ] Ablation studies on Mamba depth and embedding dimensions