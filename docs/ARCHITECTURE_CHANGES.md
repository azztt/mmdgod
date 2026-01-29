# Architecture Changes Summary

This document summarizes the major architectural changes made to the Omni3D codebase for domain-generalized RGB-D 3D object detection.

## 1. TransformerDecoder3DHead (replaces CubeHead)

**Location**: `cubercnn/modeling/roi_heads/transformer_3d_head.py`

### Architecture
- Replaces the original MLP-based CubeHead with a transformer encoder architecture
- Uses Flash Attention (PyTorch 2.0+) via `scaled_dot_product_attention` for efficiency
- **Same interface as CubeHead**: Input flattened ROI features, output per-class predictions

### Components
1. **Input Projection**: Linear + LayerNorm + ReLU + Dropout
2. **Transformer Encoder Layers** (default 6 layers):
   - Self-attention (Flash Attention when available)
   - Feed-forward network (2048 dim)
   - Layer normalization
3. **Output Heads** (same outputs as CubeHead):
   - `box_2d_deltas`: (N, num_classes, 2) - 2D center offsets
   - `box_z`: (N, num_classes, 1) - depth
   - `box_dims`: (N, num_classes, 3) - dimensions
   - `box_pose`: (N, num_classes, 3, 3) - rotation matrices
   - `box_uncert`: (N, num_classes) - confidence

### Usage
Set in config:
```yaml
MODEL:
  ROI_CUBE_HEAD:
    USE_TRANSFORMER: True  # Automatically selects TransformerDecoder3DHead
    FEATURE_DIM: 256
    NUM_DECODER_LAYERS: 6
    NUM_HEADS: 8
    DIM_FEEDFORWARD: 2048
    DROPOUT: 0.1
    USE_FLASH_ATTENTION: True
```

## 2. Auxiliary Depth Completion Branch

**Location**: `cubercnn/modeling/auxiliary_heads.py`

### Purpose
Self-supervised auxiliary task for domain generalization:
- Sparsifies depth maps randomly during training
- Trains model to reconstruct dense depth from sparse input
- Encourages learning robust depth-aware features

### Components

1. **DepthSparsifier**
   - Randomly masks depth pixels (keeps 5-20% by default)
   - Random density per sample for augmentation

2. **DepthCompletionHead**
   - CNN decoder with progressive upsampling
   - Input: FPN features (p2 level)
   - Output: Dense depth prediction

3. **DepthCompletionLoss**
   - L1/L2/BerHu loss at masked locations only
   - Self-supervised: GT is the original dense depth

### Usage
Set in config:
```yaml
MODEL:
  DEPTH_COMPLETION:
    ENABLED: True
    FEATURE_DIM: 256
    NUM_DECODER_LAYERS: 4
    USE_SKIP_CONNECTIONS: True
    SPARSITY: 0.1        # Keep 10% of pixels
    SPARSITY_MIN: 0.05   # Min sparsity
    SPARSITY_MAX: 0.2    # Max sparsity
    LOSS_TYPE: "l1"      # l1, l2, or berhu
    LOSS_WEIGHT: 1.0
```

## 3. Integration in RCNN3D_RGBD

**Location**: `cubercnn/modeling/meta_arch/rcnn3d_rgbd.py`

### Changes
- Added `auxiliary_branch` parameter in `__init__`
- Modified `from_config` to build AuxiliaryBranch when enabled
- Modified `forward` to:
  1. Store original depth before normalization
  2. Call auxiliary branch with FPN features
  3. Add auxiliary losses to total loss dict

### Loss Output
When depth completion is enabled, the loss dict includes:
- `loss_depth_completion`: Depth reconstruction loss

## 4. Config Defaults

**Location**: `cubercnn/config/config.py`

Added defaults for:
- Transformer head settings (`ROI_CUBE_HEAD.USE_TRANSFORMER`, etc.)
- Depth completion settings (`DEPTH_COMPLETION.*`)

## 5. Updated Base_RGBD.yaml

**Location**: `configs/rgbd/Base_RGBD.yaml`

Changes:
- **Removed backbone freezing**: `RGB_FROZEN: False`, `DEPTH_FROZEN: False`
  - Will freeze later when using DINO backbones
- **Enabled transformer head**: `ROI_CUBE_HEAD.USE_TRANSFORMER: True`
- **Enabled depth completion**: `DEPTH_COMPLETION.ENABLED: True`

## Files Modified

| File | Changes |
|------|---------|
| `cubercnn/modeling/roi_heads/transformer_3d_head.py` | New file - TransformerDecoder3DHead |
| `cubercnn/modeling/auxiliary_heads.py` | New file - Depth completion branch |
| `cubercnn/modeling/roi_heads/__init__.py` | Added transformer head import |
| `cubercnn/modeling/roi_heads/cube_head.py` | Modified `build_cube_head` to support USE_TRANSFORMER |
| `cubercnn/modeling/meta_arch/rcnn3d_rgbd.py` | Integrated auxiliary branch |
| `cubercnn/config/config.py` | Added config defaults |
| `configs/rgbd/Base_RGBD.yaml` | Updated config |

## Testing

All components pass unit tests:
```python
# Transformer head test
head = build_cube_head(cfg, input_shape)
box_2d_deltas, box_z, box_dims, box_pose, box_uncert = head(input_features)

# Auxiliary branch test
aux_branch = AuxiliaryBranch(...)
outputs = aux_branch(depth_images, fpn_features, feature_key='p2')
# outputs: {'sparse_depth', 'pred_depth', 'loss_depth_completion', ...}
```
