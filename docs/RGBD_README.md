# RGB-D Domain-Generalized 3D Object Detection

This extension adds RGB-D (depth) support and domain generalization capabilities to Cube R-CNN.

## Overview

The RGB-D extension enables:
1. **Dual Encoder Architecture**: Separate backbones for RGB and depth with different freezing strategies
2. **Multiple Fusion Strategies**: Concatenation, gated fusion, and depth-aware attention fusion (DAAF)
3. **Domain Generalization**: FSDR augmentations, photometric jitter, depth dropout
4. **Cross-Domain Evaluation**: Train on synthetic (Hypersim), test on real (SUNRGBD)

## Architecture

```
RGB Image → RGB Encoder (frozen) → RGB Features ─┐
                                                 ├─→ Fusion → FPN → RPN → ROI Heads → 3D Boxes
Depth Map → Depth Encoder (partial) → Depth Features ─┘
```

### Key Components

- **Dual Encoder Backbone** (`cubercnn/modeling/backbone/dual_encoder.py`): Separate ResNet encoders for RGB and depth
- **Fusion Modules** (`cubercnn/modeling/fusion/`):
  - `ConcatFusion`: Simple concatenation + projection
  - `GatedFusion`: Attention-based gating
  - `DepthAwareAttentionFusion`: Geometry-aware attention masking
- **RCNN3D_RGBD** (`cubercnn/modeling/meta_arch/rcnn3d_rgbd.py`): Extended meta-architecture for RGB-D input
- **DatasetMapper3D_RGBD** (`cubercnn/data/dataset_mapper_rgbd.py`): Data loading with depth + DG augmentations

## Installation

The RGB-D extension uses the same dependencies as Cube R-CNN:

```bash
# Install detectron2 and pytorch3d first (see main README)

# Install cubercnn
pip install -e .
```

## Dataset Preparation

### Expected Format

The dataset should follow the Omni3D format with an additional `depth_file_name` field:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "path/to/rgb.jpg",
      "depth_file_name": "path/to/depth.npy",  // NEW: depth file
      "height": 480,
      "width": 640,
      "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    }
  ],
  "annotations": [...],
  "categories": [...]
}
```

### Supported Depth Formats

- `.npy`: NumPy arrays (recommended)
- `.npz`: Compressed NumPy arrays
- `.png`: 16-bit depth images (mm)
- `.exr`: OpenEXR depth maps

## Training

### Basic Training

```bash
# Single GPU
python tools/train_rgbd.py --config-file configs/rgbd/rgbd_hypersim_to_sunrgbd.yaml

# Multi-GPU (4 GPUs)
python tools/train_rgbd.py --config-file configs/rgbd/rgbd_hypersim_to_sunrgbd.yaml --num-gpus 4
```

### Configuration Options

Key config options in `configs/rgbd/Base_RGBD.yaml`:

```yaml
MODEL:
  META_ARCHITECTURE: "RCNN3D_RGBD"
  USE_DUAL_ENCODER: True
  
  # Backbone freezing
  RGB_FREEZE_AT: 4      # Fully frozen RGB encoder
  DEPTH_FREEZE_AT: 2    # Partially frozen depth encoder
  
  # Fusion type: 'concat', 'add', 'gated'
  FUSION_TYPE: "concat"
  
  # Domain generalization
  USE_DG_AUGMENTATIONS: True
  USE_FSDR: True
  FSDR_PROB: 0.5
  USE_DEPTH_DROPOUT: True
  DEPTH_DROPOUT_PROB: 0.3
```

### Fusion Types

1. **Concatenation** (`concat`): Simple baseline, concatenate features and project
2. **Gated Fusion** (`gated`): Cross-attention + learned gating weights
3. **Add Fusion** (`add`): Element-wise addition (requires same dimensions)

## Evaluation

```bash
python tools/train_rgbd.py --config-file configs/rgbd/rgbd_hypersim_to_sunrgbd.yaml --eval-only \
    MODEL.WEIGHTS /path/to/checkpoint.pth
```

## Domain Generalization Augmentations

### FSDR (Frequency Space Domain Randomization)

Perturbs image amplitude and phase in frequency domain to simulate domain shift while preserving semantics.

### Depth Dropout

Randomly zeros out depth regions to simulate missing depth values common in real sensors.

### Photometric Jitter

Random brightness, contrast, saturation, and hue variations for RGB images.

## Citation

If you use this RGB-D extension, please cite:

```bibtex
@article{brazil2023omni3d,
  title={Omni3D: A Large Benchmark and Model for 3D Object Detection in the Wild},
  author={Brazil, Garrick and others},
  journal={CVPR},
  year={2023}
}
```

## License

Same as Cube R-CNN - see LICENSE.md in the root directory.
