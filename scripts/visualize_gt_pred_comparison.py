#!/usr/bin/env python3
"""
Side-by-side visualization of ground truth and predictions.

This script creates visualizations showing:
- Left: Ground truth bounding boxes
- Right: Model predictions

Can be used to verify evaluation is working correctly.
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2
from collections import defaultdict
from pathlib import Path

# Add project root
sys.path.insert(0, '/mnt/data/users/anweshan/omni3d')

from cubercnn import util


def get_color_for_category(cat_idx, num_cats=31):
    """Generate a unique color for each category."""
    # Use HSV colorspace for better distinction
    hue = (cat_idx * 180 // num_cats) % 180
    color = cv2.cvtColor(np.array([[[hue, 255, 200]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in color)


def draw_2d_box(im, bbox, color=(0, 255, 0), thickness=2, label=None, score=None):
    """Draw 2D bounding box on image.
    
    Args:
        im: Image to draw on
        bbox: [x, y, w, h] format
        color: BGR color tuple
        thickness: Line thickness
        label: Category label
        score: Confidence score
    """
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(im, (x, y), (x + w, y + h), color, thickness)
    
    if label:
        text = label
        if score is not None:
            text = f"{label}: {score:.2f}"
        
        # Draw text background
        (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(im, (x, y - text_h - 5), (x + text_w, y), color, -1)
        cv2.putText(im, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_3d_box_corners(im, corners_2d, color=(0, 255, 0), thickness=2):
    """Draw 3D box from 2D projected corners.
    
    Args:
        im: Image to draw on
        corners_2d: 8x2 array of projected corner coordinates
        color: BGR color tuple
        thickness: Line thickness
    """
    if corners_2d is None or len(corners_2d) != 8:
        return
    
    corners = corners_2d.astype(int)
    
    # Draw bottom face (corners 0-3)
    for i in range(4):
        cv2.line(im, tuple(corners[i]), tuple(corners[(i+1)%4]), color, thickness)
    
    # Draw top face (corners 4-7)
    for i in range(4):
        cv2.line(im, tuple(corners[i+4]), tuple(corners[(i+1)%4+4]), color, thickness)
    
    # Draw vertical edges
    for i in range(4):
        cv2.line(im, tuple(corners[i]), tuple(corners[i+4]), color, thickness)


def project_3d_to_2d(center_cam, dimensions, K, pose=None):
    """Project 3D box to 2D corners.
    
    Args:
        center_cam: [x, y, z] center in camera coordinates
        dimensions: [w, h, l] box dimensions
        K: 3x3 camera intrinsic matrix
        pose: 3x3 rotation matrix (optional)
    
    Returns:
        8x2 array of 2D corner coordinates, or None if behind camera
    """
    center_cam = np.array(center_cam).flatten()
    dimensions = np.array(dimensions).flatten()
    
    if len(center_cam) != 3 or len(dimensions) != 3:
        return None
    
    x, y, z = float(center_cam[0]), float(center_cam[1]), float(center_cam[2])
    w, h, l = float(dimensions[0]), float(dimensions[1]), float(dimensions[2])
    
    if z <= 0:
        return None
    
    # Define 3D box corners relative to center
    corners_3d = np.array([
        [-w/2, -h/2, -l/2],
        [+w/2, -h/2, -l/2],
        [+w/2, -h/2, +l/2],
        [-w/2, -h/2, +l/2],
        [-w/2, +h/2, -l/2],
        [+w/2, +h/2, -l/2],
        [+w/2, +h/2, +l/2],
        [-w/2, +h/2, +l/2],
    ])
    
    # Apply rotation if provided
    if pose is not None:
        pose = np.array(pose)
        if pose.shape == (3, 3):
            corners_3d = (pose @ corners_3d.T).T
    
    # Translate to world position
    corners_3d = corners_3d + np.array([x, y, z])
    
    # Project to 2D
    K = np.array(K)
    corners_2d = (K @ corners_3d.T).T
    
    # Check if any corner is behind camera
    if np.any(corners_2d[:, 2] <= 0):
        return None
    
    corners_2d = corners_2d[:, :2] / corners_2d[:, 2:3]
    
    return corners_2d


def create_side_by_side_visualization(
    image_path,
    depth_path,
    gt_annotations,
    predictions,
    cat_id_to_name,
    K,
    output_path,
    model_categories,
    draw_3d=True,
):
    """Create side-by-side visualization of GT and predictions.
    
    Args:
        image_path: Path to RGB image
        depth_path: Path to depth file (optional)
        gt_annotations: List of GT annotation dicts
        predictions: List of prediction dicts
        cat_id_to_name: Dict mapping category IDs to names
        K: Camera intrinsic matrix
        output_path: Path to save visualization
        model_categories: List of model category names
        draw_3d: Whether to draw 3D boxes
    """
    # Load image
    im = cv2.imread(image_path)
    if im is None:
        print(f"Failed to load image: {image_path}")
        return False
    
    h, w = im.shape[:2]
    
    # Create two copies for GT and predictions
    im_gt = im.copy()
    im_pred = im.copy()
    
    # Draw ground truth
    for ann in gt_annotations:
        cat_id = ann['category_id']
        cat_name = cat_id_to_name.get(cat_id, f"unk_{cat_id}")
        
        # Get color based on category
        if cat_name in model_categories:
            cat_idx = model_categories.index(cat_name)
        else:
            cat_idx = hash(cat_name) % 31
        color = get_color_for_category(cat_idx)
        
        # Draw 2D box if available
        if 'bbox' in ann:
            draw_2d_box(im_gt, ann['bbox'], color=color, thickness=2, label=cat_name)
        
        # Draw 3D box if available
        if draw_3d and 'center_cam' in ann and 'dimensions' in ann:
            pose = ann.get('pose', None)
            corners_2d = project_3d_to_2d(ann['center_cam'], ann['dimensions'], K, pose)
            if corners_2d is not None:
                draw_3d_box_corners(im_gt, corners_2d, color=color, thickness=2)
                # Also draw 2D bbox from 3D projection
                x1, y1 = corners_2d.min(axis=0)
                x2, y2 = corners_2d.max(axis=0)
                if 'bbox' not in ann:
                    draw_2d_box(im_gt, [x1, y1, x2-x1, y2-y1], color=color, thickness=2, label=cat_name)
    
    # Draw predictions
    for pred in predictions:
        cat_id = pred['category_id']
        score = pred.get('score', 1.0)
        
        # For predictions, cat_id is already dataset category ID
        cat_name = cat_id_to_name.get(cat_id, f"unk_{cat_id}")
        
        if cat_name in model_categories:
            cat_idx = model_categories.index(cat_name)
        else:
            cat_idx = hash(cat_name) % 31
        color = get_color_for_category(cat_idx)
        
        # Draw 2D box
        if 'bbox' in pred:
            draw_2d_box(im_pred, pred['bbox'], color=color, thickness=2, 
                       label=cat_name, score=score)
        
        # Draw 3D box if available
        if draw_3d and 'bbox3D' in pred:
            # bbox3D format: [x3d, y3d, z3d, w, h, l, rx, ry] or similar
            bbox3d = pred['bbox3D']
            if len(bbox3d) >= 6:
                center_cam = bbox3d[:3]
                dimensions = bbox3d[3:6]
                pose = None  # Simplified for now
                corners_2d = project_3d_to_2d(center_cam, dimensions, K, pose)
                if corners_2d is not None:
                    draw_3d_box_corners(im_pred, corners_2d, color=color, thickness=2)
    
    # Add labels
    label_height = 40
    im_gt_labeled = np.zeros((h + label_height, w, 3), dtype=np.uint8)
    im_pred_labeled = np.zeros((h + label_height, w, 3), dtype=np.uint8)
    
    im_gt_labeled[label_height:, :] = im_gt
    im_pred_labeled[label_height:, :] = im_pred
    
    # Add text labels
    cv2.rectangle(im_gt_labeled, (0, 0), (w, label_height), (50, 50, 50), -1)
    cv2.rectangle(im_pred_labeled, (0, 0), (w, label_height), (50, 50, 50), -1)
    
    cv2.putText(im_gt_labeled, f"Ground Truth ({len(gt_annotations)} objects)", 
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(im_pred_labeled, f"Predictions ({len(predictions)} detections)", 
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    
    # Combine side by side
    combined = np.hstack([im_gt_labeled, im_pred_labeled])
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, combined)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Create side-by-side GT/prediction visualizations")
    parser.add_argument("--dataset", type=str, default="sunrgbd_val",
                       choices=["hypersim_val", "sunrgbd_val", "multiscan_val", "scannetpp_val"],
                       help="Dataset to visualize")
    parser.add_argument("--output-dir", type=str, 
                       default="/mnt/data/users/anweshan/omni3d/output/rgbd_basic_aug/vis_comparison",
                       help="Output directory for visualizations")
    parser.add_argument("--num-images", type=int, default=20,
                       help="Number of images to visualize")
    parser.add_argument("--predictions-dir", type=str,
                       default="/mnt/data/users/anweshan/omni3d/output/rgbd_basic_aug/inference/iter_final",
                       help="Directory containing prediction results")
    parser.add_argument("--min-score", type=float, default=0.1,
                       help="Minimum score threshold for predictions")
    args = parser.parse_args()
    
    # Paths
    manifest_root = "/mnt/data/users/anweshan/omni3d/data/unidet3d_format/cleaned_manifests"
    data_root = "/mnt/data/users/anweshan/omni3d/data"
    
    manifest_map = {
        "hypersim_val": "hypersim_val_filtered.json",
        "sunrgbd_val": "sunrgbd_val_filtered.json",
        "multiscan_val": "multiscan_val_filtered.json",
        "scannetpp_val": "scannetpp_val_filtered.json",
    }
    
    # Model categories
    model_categories = [
        'person', 'books', 'chair', 'towel', 'blinds', 'window', 'lamp', 
        'shelves', 'mirror', 'sink', 'cabinet', 'bathtub', 'door', 'toilet',
        'desk', 'box', 'bookcase', 'picture', 'table', 'counter', 'bed',
        'night stand', 'dresser', 'pillow', 'sofa', 'television', 'floor mat',
        'curtain', 'clothes', 'stationery', 'refrigerator'
    ]
    
    # Load manifest
    manifest_path = os.path.join(manifest_root, manifest_map[args.dataset])
    print(f"Loading manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    cat_id_to_name = {c['id']: c['name'] for c in manifest['categories']}
    img_id_to_info = {img['id']: img for img in manifest['images']}
    
    # Group annotations by image
    anns_by_image = defaultdict(list)
    for ann in manifest['annotations']:
        anns_by_image[ann['image_id']].append(ann)
    
    # Load predictions
    pred_dataset_name = f"{args.dataset}_rgbd"
    predictions_path = os.path.join(args.predictions_dir, pred_dataset_name, "omni_instances_results.json")
    
    if os.path.exists(predictions_path):
        print(f"Loading predictions: {predictions_path}")
        with open(predictions_path) as f:
            all_predictions = json.load(f)
        
        # Group predictions by image
        preds_by_image = defaultdict(list)
        for pred in all_predictions:
            if pred['score'] >= args.min_score:
                preds_by_image[pred['image_id']].append(pred)
    else:
        print(f"Warning: No predictions found at {predictions_path}")
        preds_by_image = defaultdict(list)
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize images
    images_to_vis = manifest['images'][:args.num_images]
    
    print(f"\nVisualizing {len(images_to_vis)} images...")
    
    success_count = 0
    for i, img_info in enumerate(images_to_vis):
        img_id = img_info['id']
        img_path = os.path.join(data_root, img_info['file_path'])
        depth_path = os.path.join(data_root, img_info.get('depth_file_path', ''))
        
        K = np.array(img_info.get('K', [[500, 0, 320], [0, 500, 240], [0, 0, 1]]))
        
        gt_anns = anns_by_image.get(img_id, [])
        predictions = preds_by_image.get(img_id, [])
        
        output_path = os.path.join(output_dir, f"{i:04d}_img{img_id}.jpg")
        
        success = create_side_by_side_visualization(
            img_path, depth_path, gt_anns, predictions,
            cat_id_to_name, K, output_path, model_categories,
            draw_3d=True
        )
        
        if success:
            success_count += 1
            print(f"  [{i+1}/{len(images_to_vis)}] Image {img_id}: {len(gt_anns)} GT, {len(predictions)} preds -> {output_path}")
        else:
            print(f"  [{i+1}/{len(images_to_vis)}] Image {img_id}: FAILED")
    
    print(f"\nSuccessfully created {success_count}/{len(images_to_vis)} visualizations")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()
