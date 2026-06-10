# MitosBench: A Pipeline and Controlled Benchmark for Evaluating Pathology Foundation Models for Mitotic Figure Detection in Histopathology Whole-Slide Images

![MitosBench Graphical Abstract](graphical%20abstruct.png)

## Overview

**MitosBench** is a comprehensive pipeline and controlled benchmark for evaluating pathology foundation models in the context of mitotic figure detection in whole-slide histopathology images. This project integrates data from multiple sources (TUPAC16, MIDOG, MITOS-WSI-CMC, MITOS-WSI-CCMCT) across both human and canine tumors, providing a robust framework for:

- **Data Integration & Preprocessing**: Tiling and patch extraction from whole-slide images (WSIs) at 40× magnification (~0.25 μm/pixel)
- **Instance Segmentation**: Nuclei segmentation using PathOSAM (a histopathology-specialized variant of Segment Anything Model)
- **Classification**: Binary classification (Mitotic vs. Non-Mitotic nuclei) using multiple foundation models with optional mask fusion and LoRA fine-tuning
- **Benchmarking**: Systematic evaluation of 13+ foundation models for robustness and cross-domain generalization

The pipeline processes **629 WSIs/ROIs** containing approximately **171k mitotic figures** with curated hard negatives for reliable evaluation.

## Key Features

✨ **Comprehensive Dataset Integration**
- Multiple datasets: TUPAC16, MIDOG, MITOS-WSI-CMC, MITOS-WSI-CCMCT
- Species diversity: Human and canine tumors
- ~171k mitotic figures with carefully curated hard negatives

🔬 **Two-Stage Pipeline**
1. **Segmentation**: Instance-level nuclei segmentation using PathOSAM
2. **Classification**: Foundation model-based classification with optional mask fusion

🤖 **Foundation Model Support**
- UNI v2, Virchow2, CONCH v1, Hibou-L, and others
- LoRA fine-tuning for improved F1 and balanced precision/recall
- Mask-image and mask-text fusion strategies

📊 **Robust Evaluation**
- Largest benchmark for pathology foundation models (13+ FMs, 4 cohorts)
- Cross-domain robustness testing
- Systematic comparison of architectures and fine-tuning strategies

## Citation

If you use this code or benchmark, please cite the original paper:

```bibtex
@ARTICLE{11480928,
  author={Albahri, Mohamed and Kukuk, Markus and Nensa, Felix and Lodde, Georg and Livingstone, Elisabeth and Schadendorf, Dirk},
  journal={IEEE Access}, 
  title={MitosBench: A Pipeline and Controlled Benchmark for Evaluating Pathology Foundation Models for Mitotic Figure Detection in Histopathology Whole-Slide Images}, 
  year={2026},
  volume={14},
  number={},
  pages={59466--59485},
  keywords={Benchmark;computational pathology;domain shift;deep learning;foundation models;mitotic figure detection;segmentation and detection},
  doi={10.1109/ACCESS.2026.3683719}
}
```

## Acknowledgments

This work builds upon and is inspired by the methodology presented in:

**Virchow: A Large Vision Model for Computational Pathology** ([https://doi.org/10.1038/s42003-024-07398-6](https://doi.org/10.1038/s42003-024-07398-6))

We acknowledge the contributions of the computational pathology community and the foundation model development efforts that made this benchmark possible.

## Directory Structure

```
MitosBench/
├── Training/                    # Training pipeline for mitotic figure classification
│   ├── classify.py             # Main training script using PyTorch Lightning
│   └── SAM.py                  # Data module for training
├── Inference/                  # Inference and deployment scripts
│   ├── PathoSam_Inference.py  # Classification inference using trained models
│   ├── Segment_PathoSam_Inference.py  # Instance segmentation using PathOSAM
│   ├── SimpleWsiReader.py      # Utilities for reading WSI files
│   └── Inference_Config.ini    # Configuration for inference
├── Models/                     # Model architectures
│   └── Classifier.py           # Unified classifier supporting multiple backbones
├── Utils/                      # Utility functions and tools
│   ├── ColourAugment.py        # Color-based data augmentation
│   ├── GeneratingNRRD.py       # NRRD file generation utilities
│   ├── SAM_utils.py            # Utilities for Segment Anything Model
│   ├── constants.py            # Normalization constants for different models
│   └── model_zoo/              # Pre-trained model utilities
├── Dataloader/                 # Data loading modules
│   ├── Dataloader.py           # Custom PyTorch DataLoader
│   └── SAM.py                  # PyTorch Lightning DataModule
├── dataset/                    # Dataset metadata and annotations
│   ├── masks_dataset.csv       # Dataset with mask annotations
│   └── masks_dataset_with_source.csv  # Extended annotations with data source
├── patho-sam/                  # PathOSAM (Segment Anything Model for histopathology)
│   ├── patho_sam/             # Core PathOSAM library
│   ├── scripts/               # Utility scripts
│   ├── examples/              # Example notebooks and usage
│   └── environment.yaml       # PathOSAM environment
├── Configs/                    # Configuration files
│   └── classify_config.ini    # Training and model configuration
├── envs/                       # Conda environment specifications
│   ├── classification_no_builds.yml
│   └── pathoSam1_no_builds.yml
├── run_train_slurm.sh         # SLURM job submission for training
├── run_script_Inference.sh    # Inference execution script
└── classify_config.ini        # Main configuration file
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (for GPU acceleration)
- Conda (recommended for environment management)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd MitosBench
   ```

2. **Create Conda environments**
   
   For PathOSAM-based segmentation:
   ```bash
   conda env create -f envs/pathoSam1_no_builds.yml -n pathoSam1
   ```
   
   For classification:
   ```bash
   conda env create -f envs/classification_no_builds.yml -n classification
   ```

3. **Install additional dependencies** (if needed)
   ```bash
   # Install PathOSAM if required
   cd patho-sam
   pip install -e .
   cd ..
   ```

4. **Download pre-trained models**
   
   The pipeline supports multiple foundation models:
   - **UNI v2**: Automatically downloaded from HuggingFace Hub
   - **Virchow2**: Automatically downloaded from HuggingFace Hub
   - **CONCH v1**: Requires authentication (see CONCH documentation)
   - **Hibou-L**: Automatically downloaded
   - **PathOSAM**: Included in patho-sam/ directory

## Usage

### 1. Prepare Your Data

Organize your whole-slide images in a directory structure and create a CSV file with annotations:

```csv
image_path,mask_path,label,dataset_source
/path/to/wsi1.tiff,/path/to/mask1.npy,mitotic,MIDOG
/path/to/wsi2.tiff,/path/to/mask2.npy,non_mitotic,TUPAC16
```

### 2. Configure the Pipeline

Edit `Configs/classify_config.ini` to match your environment:

```ini
[DATA]
Dataframe = '/path/to/your/dataset.csv'
nrrd_path = '/path/to/nrrd/files/'
masks_path = '/path/to/segmentation/masks/'

[BASEMODEL]
Backbone = "uni_v2"  # or virchow2, conch_v1, hibou_l, etc.
LoRA = true
Mask_Input = true

[OPTIMIZER]
lr = 5e-4

[AUGMENTATION]
horizontalflip = 0.5
randombrightnesscontrast = 0.2
```

### 3. Training

#### Option A: Local Training
```bash
conda activate classification
python Training/classify.py Configs/classify_config.ini
```

#### Option B: SLURM Cluster Training
```bash
sbatch run_train_slurm.sh
```

Edit `run_train_slurm.sh` to adjust:
- GPU allocation (`--gpus=1`)
- Time limit (`--time=1-00:00:00`)
- Output paths for logs

### 4. Inference

#### Segmentation + Classification Pipeline

```bash
# Set configuration paths
export Config="path/to/Inference_Config.ini"
export Image="path/to/wsi.tiff"
export Output="path/to/output/directory"
export Checkpoint="path/to/checkpoint.ckpt"

# Run full pipeline
bash run_script_Inference.sh
```

Or manually:

```bash
# Step 1: Segmentation (PathOSAM)
conda activate pathoSam1
python Inference/Segment_PathoSam_Inference.py "$Config" "$Image" "$Output"

# Step 2: Classification
conda activate classification
python Inference/PathoSam_Inference.py "$Config" "$Image" "$Output" "$Checkpoint"
```

### 5. Evaluation

The pipeline outputs:
- **Segmentation masks**: Binary instance masks for detected nuclei
- **Classification results**: CSV with predictions and confidence scores
- **Overlay images**: Visualizations of detected mitotic figures
- **Metrics**: F1-score, precision, recall, and other evaluation metrics

## Configuration Details

### Key Configuration Parameters

**Segmentation (SAM_MODEL)**
- `Model_Type`: "vit_b_histopathology" (optimized for histopathology)
- `points_per_side`: 32 (grid density for prompt generation)
- `pred_iou_thresh`: 0.86 (prediction IoU threshold)
- `stability_score_thresh`: 0.92 (mask stability threshold)
- `box_nms_thresh`: 0.1 (non-maximum suppression)

**Classification (BASEMODEL)**
- `Backbone`: Foundation model choice (uni_v2, virchow2, conch_v1, etc.)
- `LoRA`: Enable LoRA fine-tuning for efficient parameter tuning
- `Mask_Input`: Use segmentation masks as auxiliary input
- `Mask_Fusion`: Strategy (concat, add, or film) for combining features

**Training**
- `Batch_Size`: 64 (adjust based on GPU memory)
- `Max_Epochs`: 30
- `lr`: 5e-4 (learning rate)
- `Precision`: "16-mixed" (mixed precision training for efficiency)

**Data**
- `Patch_Size`: [256, 256] (input patch size from WSI)
- `Input_Size`: [64, 64] (network input size)
- `balancing_strategy`: WeightedRandomSampler for class imbalance

## Output Files

### Training
- **Lightning checkpoints**: `lightning_logs/Classifier/{backbone}/Mask_Input_{True/False}/{timestamp}/`
- **TensorBoard logs**: Scalars, histograms, and metrics
- **Model weights**: `.ckpt` files with full model state

### Inference
- **Predictions CSV**: Mitotic figure locations, confidence scores, predictions
- **Overlay images**: Segmentation masks and classification overlays
- **Metrics JSON**: Performance statistics

## Supported Foundation Models

The pipeline supports the following backbone architectures:

| Model | Source | Precision | Resolution |
|-------|--------|-----------|-----------|
| UNI v2 | HuggingFace | 32-bit | 256×256 |
| Virchow2 | HuggingFace | 16-bit | 256×256 |
| CONCH v1 | CONCH GitHub | 32-bit | 224×224 |
| Hibou-L | HuggingFace | 32-bit | 224×224 |
| ResNet18 | torchvision | 32-bit | 64×64 |

## System Requirements

### Minimum
- GPU: 8GB VRAM (for batch size 16)
- RAM: 32GB
- Storage: 50GB (for datasets and models)

### Recommended
- GPU: 12-24GB VRAM (for larger batch sizes)
- RAM: 64GB
- Storage: 500GB
- High-speed SSD for WSI processing

## Performance Metrics

The benchmark evaluates models on:
- **F1-score**: Harmonic mean of precision and recall
- **Balanced Accuracy**: Mean recall for each class
- **Precision & Recall**: For clinical reliability assessment
- **Cross-domain robustness**: Performance across different datasets and domains

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce batch size in config
Batch_Size = 32  # or lower

# Or use gradient accumulation
Accumulate_Grad_Batches = 2
```

### WandB Timeout Issues
```bash
export WANDB__SERVICE_WAIT=300
```

### PathOSAM Model Loading
Ensure you have the correct CUDA version and torch installation:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## License

This project is distributed under the [LICENSE](LICENSE) file included in the repository.

## Contact & Support

For questions, issues, or contributions, please contact:

**Mohamed Albahri**
- Institution: University/Research Center
- Email: [contact@email.com]
- GitHub: [username]

## References

- **Original Paper**: Albahri et al., IEEE Access, 2026. DOI: 10.1109/ACCESS.2026.3683719
- **PathOSAM**: Segment Anything Model adapted for histopathology
- **Inspiration**: Virchow: A Large Vision Model for Computational Pathology. Nature Communications. DOI: 10.1038/s42003-024-07398-6

---

**Last Updated**: June 2026
**Version**: 1.0
**Status**: Active Development
