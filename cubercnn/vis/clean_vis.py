"""
Clean 3D Bounding Box Visualization
====================================

Provides clean side-by-side visualization of GT and predicted 3D bounding boxes.
Based on the rotation verification visualization style.

Features:
- Side-by-side GT (left) and Predictions (right) comparison
- Clean rendering with distinct colors for each box
- Class labels and scores displayed
- Consistent sample selection across epochs
- Integrates seamlessly with detectron2 evaluation pipeline
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import torch
import cv2
from typing import List, Dict, Optional, Tuple
import logging

from cubercnn.util import math_util as util

logger = logging.getLogger(__name__)


# Color palette for bounding boxes (distinct colors)
BOX_COLORS = [
    '#FF6B6B',  # Red
    '#4ECDC4',  # Teal
    '#45B7D1',  # Blue
    '#FFA07A',  # Light Salmon
    '#98D8C8',  # Mint
    '#F7DC6F',  # Yellow
    '#BB8FCE',  # Purple
    '#85C1E2',  # Sky Blue
    '#F8B195',  # Peach
    '#6C5CE7',  # Indigo
]


def project_3d_to_2d(corners_3d, K):
    """
    Project 3D corners to 2D image coordinates.
    
    Args:
        corners_3d: (N, 8, 3) or (8, 3) array - 3D corners in camera coords
        K: (3, 3) array - camera intrinsics matrix
    
    Returns:
        corners_2d: (N, 8, 2) or (8, 2) array - 2D corners in image coords
    """
    if corners_3d.ndim == 2:
        corners_3d = corners_3d[np.newaxis, ...]  # (1, 8, 3)
        squeeze_output = True
    else:
        squeeze_output = False
    
    # Project: [u, v, d] = K @ [x, y, z]
    corners_2d_h = np.einsum('ij,nkj->nki', K, corners_3d)  # (N, 8, 3)
    
    # Normalize by depth
    corners_2d = corners_2d_h[:, :, :2] / (corners_2d_h[:, :, 2:3] + 1e-8)
    
    if squeeze_output:
        corners_2d = corners_2d[0]  # (8, 2)
    
    return corners_2d


def draw_3d_box_on_image(ax, corners_2d, color, label=None, score=None, linewidth=2, clip_bounds=None):
    """
    Draw a 3D bounding box on the image using matplotlib.
    
    Args:
        ax: matplotlib axis
        corners_2d: (8, 2) array - 2D corners [u, v]
        color: str - color for the box
        label: str - class label (optional)
        score: float - confidence score (optional)
        linewidth: int - line width for edges
        clip_bounds: tuple (W, H) - image dimensions for bounds checking
    """
    # Check if box is mostly outside image bounds
    if clip_bounds is not None:
        W, H = clip_bounds
        valid_corners = (
            (corners_2d[:, 0] >= -50) & (corners_2d[:, 0] <= W + 50) &
            (corners_2d[:, 1] >= -50) & (corners_2d[:, 1] <= H + 50)
        )
        if valid_corners.sum() < 2:
            # Skip boxes that are completely outside
            return
    
    # Define the 12 edges of the 3D box (connecting corners)
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Bottom face
        (0, 4), (1, 5), (2, 6), (3, 7),  # Vertical edges
    ]
    
    # Draw edges
    for edge in edges:
        i, j = edge
        x = [corners_2d[i, 0], corners_2d[j, 0]]
        y = [corners_2d[i, 1], corners_2d[j, 1]]
        ax.plot(x, y, color=color, linewidth=linewidth, alpha=0.8)
    
    # Draw filled polygon for front face (edges 0-1-5-4)
    front_face = corners_2d[[0, 1, 5, 4], :]
    front_polygon = mpatches.Polygon(
        front_face, closed=True, 
        edgecolor=color, facecolor=color, 
        alpha=0.15, linewidth=linewidth
    )
    ax.add_patch(front_polygon)
    
    # Add label
    if label is not None:
        text = label
        if score is not None:
            text = f"{label}: {score:.2f}"
        
        # Position text at top-left corner of box, but keep inside image bounds
        text_pos = corners_2d[0]  # Top-front-left corner
        # Clip position to keep label visible (with padding for text size)
        if clip_bounds:
            W, H = clip_bounds
            text_x = np.clip(text_pos[0], 10, W - 100)  # Leave room for text width
            text_y = np.clip(text_pos[1] - 10, 20, H - 10)  # Position above corner
        else:
            text_x = text_pos[0]
            text_y = text_pos[1] - 10
        
        ax.text(
            text_x, text_y,
            text,
            fontsize=9, fontweight='bold',
            color=color,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=color, alpha=0.7, linewidth=1.5)
        )


def visualize_gt_and_predictions(
    image: np.ndarray,
    depth: Optional[np.ndarray],
    K: np.ndarray,
    gt_boxes3d: np.ndarray,
    gt_classes: np.ndarray,
    gt_poses: np.ndarray,
    pred_boxes3d: Optional[np.ndarray] = None,
    pred_classes: Optional[np.ndarray] = None,
    pred_scores: Optional[np.ndarray] = None,
    pred_poses: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    title: str = "",
) -> np.ndarray:
    """
    Create side-by-side visualization of GT (left) and Predictions (right).
    
    Args:
        image: (H, W, 3) RGB image
        depth: (H, W) depth map (optional, for visualization)
        K: (3, 3) camera intrinsics matrix
        gt_boxes3d: (N, 9) array - GT boxes [u, v, z, w, h, l, X, Y, Z]
        gt_classes: (N,) array - GT class indices
        gt_poses: (N, 3, 3) array - GT rotation matrices
        pred_boxes3d: (M, 8, 3) array - predicted 3D box corners (optional)
        pred_classes: (M,) array - predicted class indices (optional)
        pred_scores: (M,) array - prediction scores (optional)
        pred_poses: (M, 3, 3) array - predicted rotation matrices (optional)
        class_names: list of str OR dict mapping class_id->name for labels
        title: str - title for the plot
    
    Returns:
        vis_image: (H, W*2, 3) RGB visualization image
    """
    H, W = image.shape[:2]
    
    # Handle class_names as either list or dict
    def get_class_name(cls_idx):
        if class_names is None:
            return f"Class {cls_idx}"
        elif isinstance(class_names, dict):
            return class_names.get(cls_idx, f"Class {cls_idx}")
        elif isinstance(class_names, list):
            # Assume list index matches class_idx
            if 0 <= cls_idx < len(class_names):
                return class_names[cls_idx]
            else:
                return f"Class {cls_idx}"
        else:
            return f"Class {cls_idx}"
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # ========== Left: Ground Truth ==========
    ax_gt = axes[0]
    ax_gt.imshow(image)
    ax_gt.set_title(f"Ground Truth ({len(gt_boxes3d)} boxes)", 
                    fontsize=14, fontweight='bold')
    ax_gt.set_xlim(0, W)
    ax_gt.set_ylim(H, 0)  # Inverted y-axis for image coordinates
    ax_gt.axis('off')
    
    # Draw GT boxes
    for i, (box3d, cls_idx, pose_or_corners) in enumerate(zip(gt_boxes3d, gt_classes, gt_poses)):
        # Extract box parameters
        center_cam = box3d[6:9]  # [X, Y, Z]
        dimensions = box3d[3:6]  # [w, h, l]
        
        # Check if pose_or_corners is rotation matrix (3x3) or corners (8x3)
        if pose_or_corners.shape == (3, 3):
            # It's a rotation matrix - reconstruct corners using cubercnn util
            R = pose_or_corners
            box3d_tensor = torch.tensor(np.concatenate([center_cam, dimensions])).unsqueeze(0)
            R_tensor = torch.tensor(R).unsqueeze(0)
            corners_3d, _ = util.get_cuboid_verts_faces(box3d_tensor, R_tensor)
            corners_3d = corners_3d[0].numpy()
        elif pose_or_corners.shape == (8, 3):
            # It's already the corners - use directly!
            corners_3d = pose_or_corners
        else:
            logger.warning(f"Unexpected pose shape: {pose_or_corners.shape}, skipping box")
            continue
        
        # Project to 2D
        corners_2d = project_3d_to_2d(corners_3d, K)
        
        # Get color and label
        color = BOX_COLORS[i % len(BOX_COLORS)]
        label = get_class_name(cls_idx)
        
        # Draw box (with bounds checking)
        draw_3d_box_on_image(ax_gt, corners_2d, color, label=label, linewidth=2, clip_bounds=(W, H))
    
    # ========== Right: Predictions ==========
    ax_pred = axes[1]
    ax_pred.imshow(image)
    ax_pred.set_xlim(0, W)
    ax_pred.set_ylim(H, 0)  # Inverted y-axis for image coordinates
    
    if pred_boxes3d is not None and len(pred_boxes3d) > 0:
        ax_pred.set_title(f"Predictions ({len(pred_boxes3d)} boxes)", 
                         fontsize=14, fontweight='bold')
        
        # Draw predicted boxes
        for i, corners_3d in enumerate(pred_boxes3d):
            # Project to 2D
            corners_2d = project_3d_to_2d(corners_3d, K)
            
            # Get color and label
            color = BOX_COLORS[i % len(BOX_COLORS)]
            cls_idx = pred_classes[i] if pred_classes is not None else 0
            score = pred_scores[i] if pred_scores is not None else None
            label = get_class_name(cls_idx)
            
            # Draw box (with bounds checking)
            draw_3d_box_on_image(ax_pred, corners_2d, color, label=label, 
                               score=score, linewidth=2, clip_bounds=(W, H))
    else:
        ax_pred.set_title("Predictions (0 boxes)", 
                         fontsize=14, fontweight='bold')
    
    ax_pred.axis('off')
    
    # Add overall title
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Convert to numpy array
    fig.canvas.draw()
    # Use buffer_rgba() for modern matplotlib
    buf = fig.canvas.buffer_rgba()
    vis_image = np.asarray(buf)
    vis_image = vis_image[:, :, :3]  # Drop alpha channel (RGBA -> RGB)
    
    plt.close(fig)
    
    return vis_image


def visualize_from_instances(
    instances: Dict,
    dataset,
    dataset_name: str,
    output_dir: str,
    class_names: List[str],
    iteration: int = 0,
    num_samples: int = 8,
    fixed_indices: Optional[List[int]] = None,
):
    """
    Create visualizations from predicted instances and save to output directory.
    
    Args:
        instances: dict mapping image_id -> Instance predictions
        dataset: dataset object with annotations
        dataset_name: name of the dataset
        output_dir: directory to save visualizations
        class_names: list of class names
        iteration: current training iteration
        num_samples: number of samples to visualize
        fixed_indices: list of fixed dataset indices to visualize (for consistency)
    
    Returns:
        log_str: string describing what was visualized
    """
    os.makedirs(output_dir, exist_ok=True)
    # Handle iteration as string (e.g., 'final') or int
    iter_str = f'iter_{iteration:07d}' if isinstance(iteration, int) else f'iter_{iteration}'
    vis_dir = os.path.join(output_dir, 'visualizations', iter_str)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Build category_id -> class_name mapping from dataset
    # class_names might be a list (0-indexed) but category_ids might not match indices
    if hasattr(dataset, 'dataset_dicts') and len(dataset.dataset_dicts) > 0:
        # Try to get category info from first sample
        sample_0 = dataset.dataset_dicts[0]
        if 'annotations' in sample_0 and len(sample_0['annotations']) > 0:
            # Build mapping from annotations
            cat_id_to_name = {}
            for sample in dataset.dataset_dicts[:100]:  # Check first 100 samples
                for ann in sample.get('annotations', []):
                    cat_id = ann.get('category_id')
                    cat_name = ann.get('category_name')
                    if cat_id is not None and cat_name is not None:
                        cat_id_to_name[cat_id] = cat_name
            if cat_id_to_name:
                logger.info(f"Built category mapping with {len(cat_id_to_name)} categories from dataset")
                class_names_mapping = cat_id_to_name
            else:
                # Fallback: assume list indices match
                class_names_mapping = class_names if isinstance(class_names, list) else None
        else:
            class_names_mapping = class_names if isinstance(class_names, list) else None
    else:
        class_names_mapping = class_names if isinstance(class_names, list) else None
    
    # Select samples to visualize
    if fixed_indices is not None:
        # Use fixed indices for consistency across epochs
        sample_indices = [i for i in fixed_indices if i < len(dataset)][:num_samples]
    else:
        # Random sampling (old behavior)
        sample_indices = np.random.choice(
            len(dataset), 
            size=min(num_samples, len(dataset)), 
            replace=False
        ).tolist()
    
    num_visualized = 0
    
    for idx in sample_indices:
        try:
            # Get dataset sample
            sample = dataset[idx]
            image_id = sample['image_id']
            
            # Load image (handle both file_name and file_path)
            image_path = sample.get('file_name') or sample.get('file_path')
            if not image_path or not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Failed to load image: {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load depth (optional, handle both field names)
            depth = None
            depth_path = sample.get('depth_file') or sample.get('depth_file_path')
            if depth_path and os.path.exists(depth_path):
                if depth_path.endswith('.npy'):
                    depth = np.load(depth_path)
                else:
                    depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            
            # Get camera intrinsics
            K = np.array(sample['K']).reshape(3, 3)
            
            # Get GT annotations (following DATA.md format)
            annotations = sample['annotations']
            if len(annotations) == 0:
                logger.warning(f"No annotations for sample {idx}")
                continue
            
            # Extract GT data from annotations
            gt_boxes3d_list = []
            gt_classes_list = []
            gt_poses_list = []
            
            for ann in annotations:
                # Skip invalid 3D annotations
                if not ann.get('valid3D', True):
                    continue
                
                # Use bbox3D_cam directly (8 corners) - this is the ground truth!
                bbox3D_cam = np.array(ann['bbox3D_cam'])  # (8, 3)
                center_cam = np.array(ann['center_cam'])  # (3,)
                dimensions = np.array(ann['dimensions'])  # [w, h, l]
                
                # We still need R_cam for the pose, but use bbox3D_cam for corners
                # The annotations have the correct corners already computed
                R_cam = np.array(ann['R_cam'])  # (3, 3)
                
                # Project center to 2D for our format
                center_2d_h = K @ center_cam
                u, v = center_2d_h[0] / center_2d_h[2], center_2d_h[1] / center_2d_h[2]
                
                # Create our format: [u, v, z, w, h, l, X, Y, Z]
                # But we'll pass bbox3D_cam separately for visualization
                gt_box = np.concatenate([
                    [u, v, center_cam[2]],  # u, v, z
                    dimensions,              # w, h, l
                    center_cam               # X, Y, Z
                ])
                
                gt_boxes3d_list.append(gt_box)
                gt_classes_list.append(ann['category_id'])
                gt_poses_list.append(bbox3D_cam)  # Pass the actual corners instead of R_cam!
            
            if len(gt_boxes3d_list) == 0:
                logger.warning(f"No valid 3D annotations for sample {idx}")
                continue
            
            gt_boxes3d = np.array(gt_boxes3d_list)
            gt_classes = np.array(gt_classes_list)
            gt_poses = np.array(gt_poses_list)
            
            # Get predictions
            pred_boxes3d = None
            pred_classes = None
            pred_scores = None
            pred_poses = None
            
            if image_id in instances:
                inst = instances[image_id]
                if hasattr(inst, 'pred_bbox3D') and len(inst.pred_bbox3D) > 0:
                    pred_boxes3d = inst.pred_bbox3D.cpu().numpy()
                    pred_classes = inst.pred_classes.cpu().numpy()
                    pred_scores = inst.scores.cpu().numpy()
                    if hasattr(inst, 'pred_pose'):
                        pred_poses = inst.pred_pose.cpu().numpy()
            
            # Create visualization
            vis_image = visualize_gt_and_predictions(
                image=image,
                depth=depth,
                K=K,
                gt_boxes3d=gt_boxes3d,
                gt_classes=gt_classes,
                gt_poses=gt_poses,
                pred_boxes3d=pred_boxes3d,
                pred_classes=pred_classes,
                pred_scores=pred_scores,
                pred_poses=pred_poses,
                class_names=class_names_mapping,
                title=f"{dataset_name} - Sample {idx} (ID: {image_id})",
            )
            
            # Save visualization
            save_path = os.path.join(vis_dir, f'sample_{idx:04d}_id_{image_id}.png')
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            num_visualized += 1
            
        except Exception as e:
            logger.warning(f"Failed to visualize sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    log_str = f"Visualized {num_visualized}/{len(sample_indices)} samples to {vis_dir}"
    logger.info(log_str)
    
    return log_str


def get_fixed_visualization_indices(dataset, num_samples: int = 8, seed: int = 42):
    """
    Get fixed indices for consistent visualization across epochs.
    
    Args:
        dataset: dataset object
        num_samples: number of samples to select
        seed: random seed for reproducibility
    
    Returns:
        indices: list of fixed indices
    """
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    return indices.tolist()
