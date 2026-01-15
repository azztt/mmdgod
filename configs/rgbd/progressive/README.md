# Progressive Domain Generalization Configs for RGB-D 3D Detection

This directory contains a series of configuration files that progressively add
domain generalization (DG) capabilities to Cube R-CNN for RGB-D 3D object detection.

## Configuration Progression

Use these configs in order to systematically evaluate the contribution of each component:

| Config | Description | Key Changes |
|--------|-------------|-------------|
| `00_baseline_rgb_only.yaml` | RGB-only baseline | Standard Cube R-CNN, no depth |
| `01_rgbd_concat.yaml` | RGB-D concat fusion | + Dual encoder, + Concat fusion |
| `02_rgbd_gated.yaml` | RGB-D gated fusion | + Learned gating/attention |
| `03_rgbd_windowed.yaml` | MultiMAE-style fusion | + Windowed cross-modal attention |
| `04_rgbd_fsdr.yaml` | + FSDR augmentation | + Frequency domain randomization |
| `05_rgbd_object_style_swap.yaml` | + Object style swap | + Object-wise appearance transfer |
| `06_rgbd_full_dg.yaml` | Full DG stack | All components enabled |

## Architecture Components

### 1. Dual Encoder Backbone
- **RGB Encoder**: ResNet50 with heavy freezing (stages 1-4) to preserve domain-invariant features
- **Depth Encoder**: ResNet50 with light freezing (stages 1-2) to learn depth representation
- Separate normalization for each modality

### 2. Fusion Modules

#### Concat Fusion (Simple)
```
Features = [RGB_features; Depth_features]
```

#### Gated Fusion
```
gate = sigmoid(MLP([RGB; Depth]))
Features = gate * RGB + (1 - gate) * Depth
```
With optional cross-attention for modality interaction.

#### Adaptive MultiMAE-style Windowed Fusion
```
1. Partition features into windows (adaptive sizes per modality)
2. Cross-modal attention within windows
3. Unpartition and fuse
```
Handles resolution asymmetry between modalities.

### 3. Domain Generalization Augmentations

#### FSDR (Frequency Space Domain Randomization)
- Decomposes image into frequency bands via DCT
- Histogram matches selected bands (low + high frequencies)
- Preserves semantic structure while varying appearance

#### Object Style Swap
- Builds category index of objects across dataset
- For each object, finds same-category object from pool
- Transfers appearance via histogram matching
- Alpha blending for smooth integration

## Usage

### Training with Progressive Configs

```bash
# Step 0: Establish RGB baseline
python tools/train_net.py --config-file configs/rgbd/progressive/00_baseline_rgb_only.yaml

# Step 1: Add depth with simple fusion
python tools/train_rgbd.py --config-file configs/rgbd/progressive/01_rgbd_concat.yaml

# Step 2: Upgrade fusion
python tools/train_rgbd.py --config-file configs/rgbd/progressive/02_rgbd_gated.yaml

# ... and so on

# Final: Full DG stack
python tools/train_rgbd.py --config-file configs/rgbd/progressive/06_rgbd_full_dg.yaml
```

### Cross-Domain Transfer

For Hypersim → SUNRGBD experiments:
```bash
python tools/train_rgbd.py --config-file configs/rgbd/progressive/hypersim_to_sunrgbd.yaml
```

## Expected Results

Progressive addition should show:

| Config | In-Domain mAP | Cross-Domain mAP |
|--------|---------------|------------------|
| 00 RGB-only | Baseline | Poor |
| 01 + Depth | +2-5% | +1-3% |
| 02 + Gated | +1-3% | +2-4% |
| 03 + Windowed | +1-2% | +2-3% |
| 04 + FSDR | ≈ | +3-5% |
| 05 + Style Swap | ≈ | +1-2% |
| 06 Full | +5-10% total | +10-15% total |

*Values are illustrative; actual results depend on dataset and evaluation.*

## Configuration Details

### Key Hyperparameters

```yaml
MODEL:
  # Encoder freezing
  RGB_FREEZE_AT: 4  # Heavy freeze for domain invariance
  DEPTH_FREEZE_AT: 2  # Light freeze for adaptation
  
  # Fusion
  FUSION_TYPE: "gated"  # or "concat", "windowed"
  
DG:
  FSDR:
    PROBABILITY: 0.5
    VARIANT_BANDS: [[0, 2], [32, 64]]
  
  OBJECT_STYLE_SWAP:
    PROBABILITY: 0.3
    BLEND_MODE: "alpha"
    ALPHA: 0.7
  
  DEPTH_DROPOUT:
    PROBABILITY: 0.1
```

### Memory Considerations

| Config | Approx Memory (batch=1) |
|--------|------------------------|
| RGB-only | ~6GB |
| RGB-D concat | ~10GB |
| RGB-D gated | ~11GB |
| RGB-D windowed | ~13GB |

Reduce batch size as complexity increases.

## Ablation Study Guide

To isolate component contributions:

1. **Depth contribution**: Compare 00 vs 01
2. **Fusion method**: Compare 01 vs 02 vs 03
3. **FSDR contribution**: Compare 03 vs 04
4. **Object Style Swap**: Compare 04 vs 05
5. **Full DG vs individual**: Compare 06 vs each prior

## References

- FSDR: Inspired by Fourier Domain Adaptation methods
- Windowed Fusion: Inspired by MultiMAE, CuTR architectures
- Object Style Swap: Inspired by instance-level domain randomization
