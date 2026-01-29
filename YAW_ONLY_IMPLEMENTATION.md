# Yaw-Only Rotation Implementation

## Overview

Switched from 6D rotation representation to 1D yaw-only (gravity-aligned) rotation prediction.

## Motivation

- **Coordinate System Discovery**: `R_cam` in the dataset is **gravity-aligned world orientation**, not camera-frame rotation
- **Simplification**: Objects are already stored with gravity-aligned Y-axis (vertical)
- **No Extrinsics Needed**: R_cam is already in the correct frame - no camera extrinsics transformation required
- **Domain Generalization**: Gravity alignment is consistent across different camera orientations and datasets

## Key Insight

The dataset uses a **mixed coordinate system**:
- **Position** (`center_cam`, `bbox3D_cam`): Camera coordinates
- **Orientation** (`R_cam`): Gravity-aligned world frame (Y-axis vertical)
- This is why all objects have consistent Y-axis: `[-0.018, 0.9996, -0.020]`

## Implementation Changes

### 1. Added Utility Functions

**File**: `cubercnn/modeling/meta_arch/pure_detr3d.py`

```python
def rotation_matrix_to_yaw(R):
    """Extract yaw from gravity-aligned rotation matrix.
    For Y-axis rotation: θ = atan2(R[0,2], R[2,2])
    """
    return torch.atan2(R[..., 0, 2], R[..., 2, 2])

def yaw_to_rotation_matrix(yaw):
    """Convert yaw to Y-axis rotation matrix.
    R_y(θ) = [[cos(θ), 0, sin(θ)],
              [0,      1, 0     ],
              [-sin(θ), 0, cos(θ)]]
    """
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)
    
    R = torch.stack([
        torch.stack([cos_yaw, zeros, sin_yaw], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sin_yaw, zeros, cos_yaw], dim=-1)
    ], dim=-2)
    
    return R
```

### 2. Updated Pose Head Output Dimension

**File**: `cubercnn/modeling/meta_arch/pure_detr3d.py` (Line ~390)

```python
# Changed from:
pose_dim = 6 if pose_type == '6d' else ...

# To:
pose_dim = 1 if pose_type == 'yaw' else 6 if pose_type == '6d' else ...
```

The pose head now outputs 1D (yaw angle in radians) instead of 6D.

### 3. Updated GT Loading and Loss Computation

**File**: `cubercnn/modeling/meta_arch/pure_detr3d.py` (Line ~750)

```python
# Extract yaw from rotation matrix instead of 6D representation
if self.pose_type == 'yaw':
    gt_yaw = rotation_matrix_to_yaw(matched_gt_poses).unsqueeze(-1)  # (N, 1)
    losses['loss_pose'] += F.l1_loss(matched_pred_pose, gt_yaw)
else:
    gt_pose_6d = matched_gt_poses[:, :, :2].reshape(-1, 6)
    losses['loss_pose'] += F.l1_loss(matched_pred_pose, gt_pose_6d)
```

### 4. Updated GIoU Loss Computation

**File**: `cubercnn/modeling/meta_arch/pure_detr3d.py` (Line ~709)

```python
# Convert predicted pose to rotation matrix
if self.pose_type == 'yaw':
    pred_R = yaw_to_rotation_matrix(matched_pred_pose.squeeze(-1))
else:
    pred_R = rotation_6d_to_matrix(matched_pred_pose)
```

### 5. Updated Inference

**File**: `cubercnn/modeling/meta_arch/pure_detr3d.py` (Line ~930)

```python
# Convert pose to rotation matrix during inference
if self.pose_type == 'yaw':
    pred_R = yaw_to_rotation_matrix(pred_pose.squeeze(-1))
else:
    pred_R = rotation_6d_to_matrix(pred_pose)
```

### 6. Updated Config

**Files**: 
- `configs/rgbd/dinov3_base_aug.yaml`
- `configs/rgbd/dinov3_base_aug_debug.yaml` (inherits from base)

```yaml
ROI_CUBE_HEAD:
  POSE_TYPE: "yaw"  # Changed from "6d"
```

## Box Reconstruction Process

1. **Predict**: center2d (u,v), depth (d), dimensions (w,h,l), **yaw (θ)**
2. **Unproject center**: `center_cam = unproject(center2d, depth, K)` → Camera coordinates
3. **Create canonical corners**: `corners_obj = get_corners(dimensions)` → Object frame
4. **Apply rotation**: `R_y = yaw_to_rotation_matrix(yaw)` → Gravity-aligned rotation
5. **Transform corners**: `corners_rotated = (R_y @ corners_obj.T).T`
6. **Translate**: `corners_cam = corners_rotated + center_cam` → Camera coordinates
7. **Project for visualization**: `corners_2d = project_3d_to_2d(corners_cam, K)`

## Validation

Tested yaw extraction and reconstruction:
- ✓ 0° (0.0000 rad): Perfect reconstruction
- ✓ 45° (0.7854 rad): Perfect reconstruction
- ✓ 90° (1.5708 rad): Perfect reconstruction
- ✓ 180° (±3.1416 rad): Equivalent representations (expected)
- ✓ -45° (-0.7854 rad): Perfect reconstruction

## Training

Old checkpoint (trained with 6D) has been moved to backup:
- `output/dinov3_base_aug/old_6d_checkpoint/model_recent.pth`

Start fresh training with yaw:
```bash
# Debug config (500 samples, 26 epochs)
./quick_train.sh

# Or directly:
conda activate cube
accelerate launch --config_file accelerate_configs/single_gpu.yaml \
  src/train.py --config-name=config_debug
```

## Benefits

1. **Simpler**: 1D output instead of 6D (6× parameter reduction in pose head)
2. **More Interpretable**: Yaw angle directly interpretable
3. **Domain Invariant**: Works across different camera orientations
4. **Accurate**: Matches dataset's gravity-aligned representation
5. **No Extrinsics**: No need for camera extrinsics transformation

## Expected Behavior

- Yaw predictions in range [-π, π] radians
- Loss should converge faster (1D vs 6D)
- Rotation consistency across camera views
- Better cross-dataset generalization
