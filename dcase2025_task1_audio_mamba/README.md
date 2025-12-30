# Audio Mamba for Acoustic Scene Classification

**DCASE2025 - Task 1 - Audio Mamba Implementation**

Contact: **Abdalaziz Ayoub** - K12341559@students.jku.at

---

## Overview

This repository adapts the [DCASE2025 Task 1 Baseline](https://github.com/CPJKU/dcase2025_task1_baseline) to use **Audio Mamba (AuM)** architecture for Low-Complexity Acoustic Scene Classification. The original baseline used a simplified CP-Mobile architecture; this implementation replaces it with state-space models (Mamba) for improved temporal modeling of audio scenes.

### Key Modifications

1. **Architecture Replacement**: Replaced the CP-Mobile CNN-based model with Audio Mamba (AuM-Small and AuM-Base configurations)
2. **Added Audio Mamba Integration**: Integrated the Audio Mamba model from the [AUM repository](https://github.com/xieyumc/AUM)
3. **Custom Classification Head**: Implemented a multi-layer classifier with LayerNorm and dropout for DCASE25's 10-class scene classification
4. **Test-Only Evaluation**: Added `test_only.py` for evaluating trained checkpoints
5. **Data Validation**: Added `meta_data_preprocessing.py` to detect and filter corrupted audio files

---

## What is Audio Mamba?

Audio Mamba applies the Mamba state-space model architecture to audio classification tasks. Unlike traditional CNNs or Transformers, Mamba processes sequences with linear complexity, making it efficient for long audio sequences while maintaining strong temporal modeling capabilities.

**Key advantages for ASC:**
- Efficient processing of long-form audio spectrograms
- Bidirectional temporal modeling with Mamba blocks
- Reduced computational complexity compared to self-attention mechanisms

---

## Repository Structure

### Modified Files

- **`models/net.py`**: Complete rewrite to integrate Audio Mamba
  - Removed CP-Mobile architecture
  - Added `AudioMambaModel` class with configurable depth and embedding dimensions
  - Implemented custom classification head
  - Disabled absolute positional embeddings to avoid initialization issues

- **`train_base.py`**: Updated training script
  - Modified to pass Audio Mamba-specific parameters (`embed_dim`, `depth`, `patch_size`)
  - Added precision control for mixed-precision training
  - Implemented checkpoint tracking for best model selection
  - Fixed preprocessing consistency between validation and test phases

- **`train_device_specific.py`**: Updated device-specific fine-tuning
  - Adapted for Audio Mamba architecture
  - Maintains multi-device training pipeline from original baseline
  - Loads pre-trained weights from general model (Step 1)

### New Files

- **`test_only.py`**: Standalone evaluation script
  - Loads trained checkpoints and evaluates on test set
  - Supports both AuM-Small and AuM-Base configurations
  - Command-line interface for checkpoint path specification

- **`dataset/meta_data_preprocessing.py`**: Data validation utility
  - Validates audio file integrity (shape, NaN, Inf values)
  - Generates cleaned metadata CSV (`meta_clean.csv`)
  - Provides summary statistics on corrupted files

### External Dependencies

- **Audio Mamba (AUM)**: The Audio Mamba implementation is imported from:
```python
  import AUM.src.models.mamba_models as mamba_models
```
  Ensure the [AUM repository](https://github.com/kaistmm/Audio-Mamba-AuM/tree/main) is cloned in the parent directory or adjust the import path accordingly.

---

## Model Configurations

### AuM-Small
```python
embed_dim = 384
depth = 24
patch_size = "16,16"
n_mels = 256
target_length = 1024
```

### AuM-Base
```python
embed_dim = 768
depth = 24
patch_size = "16,16"
n_mels = 256
target_length = 1024
```

---

## Getting Started

### 1. Clone Repositories
```bash
# Clone this repository
git clone https://github.com/abdalazizayoub/audio-mamba-asc
cd dcase2025_task1_audio_mamba

# Clone Audio Mamba dependency (if not already available)
cd ..
git clone https://github.com/xieyumc/AUM.git
cd dcase2025_task1_audio_mamba
```

### 2. Environment Setup
```bash
conda create -n d25_aum python=3.13
conda activate d25_aum

# Install PyTorch (adjust for your CUDA version)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip3 install -r requirements.txt

# Install Mamba dependencies
pip install causal-conv1d>=1.1.0
pip install mamba-ssm
```

### 3. Dataset Setup

Download the [TAU Urban Acoustic Scenes 2022 Mobile, Development dataset](https://zenodo.org/records/6337421).

Update the dataset path in `dataset/dcase25.py`:
```python
dataset_dir = "/path/to/your/dataset"
```

### 4. Data Validation (Optional but Recommended)
```bash
python dataset/meta_data_preprocessing.py
```

This generates `meta_clean.csv` with validated audio files.

### 5. Weights & Biases Setup
```bash
wandb login
```

---

## Training

### Step 1: General Model Training

Train a general Audio Mamba model on the 25% subset:
```bash
python train_base.py \
    --experiment_name "AUM_Small_Scratch" \
    --embed_dim 384 \
    --depth 24 \
    --patch_size "16,16" \
    --n_epochs 150 \
    --batch_size 64 \
    --lr 0.0005 \
    --precision "32"
```

For AuM-Base, change `--embed_dim 768`.

### Step 2: Device-Specific Fine-Tuning

Fine-tune the general model for each recording device:
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

### Test Trained Model
```bash
python test_only.py \
    --ckpt_path "path/to/best-checkpoint.ckpt" \
    --embed_dim 384 \
    --depth 24 \
    --patch_size "16,16" \
    --batch_size 64
```

---

## Model Complexity

Audio Mamba models must adhere to DCASE2025 Task 1 complexity constraints:
- **Maximum parameters**: 128 kB (when converted to 16-bit precision)
- **Maximum MACs**: 30 million for 1-second audio

**Note**: The current Audio Mamba configurations may exceed these limits. Further optimizations (quantization, pruning, or architecture modifications) are needed for challenge submission.

---

## Key Differences from Original Baseline

| Aspect | Original Baseline | This Implementation |
|--------|------------------|---------------------|
| **Architecture** | CP-Mobile (CNN-based) | Audio Mamba (State-Space Models) |
| **Parameters** | 61,148 | ~25M (AuM-Small) / ~91M (AuM-Base) |
| **Temporal Modeling** | Convolutional | Bidirectional Mamba blocks |
| **Preprocessing** | Same mel-spectrogram pipeline | Same (maintained compatibility) |
| **Training Process** | Two-stage (general + device-specific) | Same (adapted for AuM) |

---

## Citations

### Original DCASE2025 Task 1 Baseline
```bibtex
@misc{dcase2025_task1_baseline,
  author = {Schmid, Florian},
  title = {DCASE2025 Task 1 Baseline System},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/CPJKU/dcase2025_task1_baseline}}
}
```

### Audio Mamba (AuM)
```bibtex
@article{xie2024audio,
  title={Audio Mamba: Bidirectional State Space Model for Audio Representation Learning},
  author={Xie, Yumeng and Zhang, Shuo and Wang, Guangliang and Wang, Wenwu},
  journal={arXiv preprint arXiv:2406.03344},
  year={2024}
}
```

### CP-Mobile (Original Baseline Architecture)
```bibtex
@inproceedings{schmid2023cp,
  title={CP-Mobile: A low-complexity neural network for acoustic scene classification},
  author={Schmid, Florian and Koutini, Khaled and Widmer, Gerhard},
  booktitle={DCASE2023 Workshop},
  year={2023}
}
```

### Mamba: Linear-Time Sequence Modeling
```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```

### TAU Urban Acoustic Scenes 2022 Dataset
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

## Known Issues and Future Work

1. **Complexity Constraints**: Current Audio Mamba models exceed DCASE2025 complexity limits
   - **Solution**: Implement quantization, knowledge distillation, or design smaller Mamba variants

2. **Absolute Positional Embeddings**: Disabled to avoid initialization errors
   - **Impact**: May affect performance on variable-length inputs
   - **Alternative**: Consider RoPE (Rotary Position Embeddings) if needed

3. **Memory Usage**: AuM-Base requires significant GPU memory
   - **Recommendation**: Use mixed-precision training (`--precision "16-mixed"`) or gradient accumulation

---

## Acknowledgments

This work builds upon:
- The DCASE2025 Task 1 baseline by Florian Schmid (Johannes Kepler University Linz)
- Audio Mamba implementation by Yumeng Xie et al.
- The Mamba architecture by Albert Gu and Tri Dao

---

## License

This project inherits the license from the original DCASE2025 baseline repository. Please refer to the original repository for licensing details.

---

## Contact

For questions or issues specific to this Audio Mamba implementation:
- **Abdalaziz Ayoub** - K12341559@students.jku.at
For questions about the original baseline:
- **Florian Schmid** - florian.schmid@jku.at

For Audio Mamba architecture questions:
- Refer to the [AUM repository](https://github.com/xieyumc/AUM)