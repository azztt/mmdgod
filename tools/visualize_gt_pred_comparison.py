#!/usr/bin/env python3
"""
Visualize 3D bounding box predictions vs ground truth side by side.
Projects 3D boxes onto 2D images for comparison.
"""
import os
import json
import argparse
import numpy as np
import cv2
from pathlib import Path


# Category colors (consistent across GT and predictions)
CATEGORY_COLORS = {}


def get_category_color(category_id):
    """Get consistent color for a category."""
    if category_id not in CATEGORY_COLORS:
        np.random.seed(category_id * 17 + 42)
        CATEGORY_COLORS[category_id] = tuple(int(c) for c in np.random.randint(50, 255, 3))
    return CATEGORY_COLORS[category_id]


def project_3d_box_to_2d(bbox3d, K):
    """Project 3D bounding box corners to 2D image coordinates.
    
    Args:
        bbox3d: (8, 3) array of 3D corner coordinates in camera frame
        K: (3, 3) camera intrinsic matrix
        
    Returns:
        corners_2d: (8, 2) array of 2D pixel coordinates
    """
    corners_3d = np.array(bbox3d)
    
    # Filter out points behind camera
    valid = corners_3d[:, 2] > 0.1
    if not np.any(valid):
        return None
    
    # Project to 2D
    corners_2d = np.zeros((8, 2))
    for i in range(8):
        if corners_3d[i, 2] > 0.1:
            p = K @ corners_3d[i]
            corners_2d[i] = p[:2] / p[2]
        else:
            corners_2d[i] = [np.nan, np.nan]
    
    return corners_2d


def draw_3d_box(image, corners_2d, color, thickness=2):
    """Draw 3D bounding box edges on image.
    
    Args:
        image: Image to draw on
        corners_2d: (8, 2) array of projected 2D corners
        color: BGR color tuple
        thickness: Line thickness
    """
    if corners_2d is None:
        return
    
    # Define edges of a 3D box (connecting 8 corners)
    # Standard order: 4 bottom corners, 4 top corners
    edges = [
        # Bottom face
        (0, 1), (1, 2), (2, 3), (3, 0),
        # Top face
        (4, 5), (5, 6), (6, 7), (7, 4),
        # Vertical edges
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    h, w = image.shape[:2]
    
    for i, j in edges:
        pt1 = corners_2d[i]
        pt2 = corners_2d[j]
        
        # Skip if any point is invalid
        if np.any(np.isnan(pt1)) or np.any(np.isnan(pt2)):
            continue
        
        # Clip to image bounds (with margin)
        pt1 = (int(np.clip(pt1[0], -w, 2*w)), int(np.clip(pt1[1], -h, 2*h)))
        pt2 = (int(np.clip(pt2[0], -w, 2*w)), int(np.clip(pt2[1], -h, 2*h)))
        
        cv2.line(image, pt1, pt2, color, thickness)


def draw_3d_boxes(image, boxes_3d, labels, colors, K, alpha=0.7):
    """Draw 3D bounding boxes projected onto image."""
    overlay = image.copy()
    
    for box_3d, label, color in zip(boxes_3d, labels, colors):
        # Project 3D box to 2D
        corners_2d = project_3d_box_to_2d(box_3d, K)
        if corners_2d is None:
            continue
        
        # Draw 3D box edges
        draw_3d_box(overlay, corners_2d, color, thickness=2)
        
        # Add label at centroid
        valid_corners = corners_2d[~np.any(np.isnan(corners_2d), axis=1)]
        if len(valid_corners) > 0:
            centroid = valid_corners.mean(axis=0).astype(int)
            h, w = image.shape[:2]
            if 0 <= centroid[0] < w and 0 <= centroid[1] < h:
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, 
                             (centroid[0], centroid[1] - text_h - 4), 
                             (centroid[0] + text_w, centroid[1]), 
                             color, -1)
                cv2.putText(overlay, label, (centroid[0], centroid[1] - 4), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


def load_manifest(manifest_path):
    """Load COCO-style manifest."""
    with open(manifest_path) as f:
        data = json.load(f)
    
    # Build lookup tables
    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    # Group annotations by image
    annotations = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = []
        annotations[img_id].append(ann)
    
    return images, categories, annotations


def load_predictions(pred_path):
    """Load predictions from COCO-format results file."""
    with open(pred_path) as f:
        preds = json.load(f)
    
    # Group by image
    by_image = {}
    for pred in preds:
        img_id = pred['image_id']
        if img_id not in by_image:
            by_image[img_id] = []
        by_image[img_id].append(pred)
    
    return by_image


def visualize_sample(image_path, gt_anns, pred_anns, categories, K, output_path, score_thresh=0.1):
    """Create side-by-side visualization of GT and predictions with 3D boxes."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return False
    
    h, w = image.shape[:2]
    K = np.array(K)
    
    # Create GT visualization
    gt_image = image.copy()
    gt_boxes = []
    gt_labels = []
    gt_colors = []
    
    for ann in gt_anns:
        # Check for valid 3D annotation
        if not ann.get('valid3D', True):  # Skip invalid 3D annotations
            continue
        
        # Try different field names for 3D box
        bbox3d = None
        if 'bbox3D_cam' in ann:
            bbox3d = np.array(ann['bbox3D_cam']).reshape(8, 3)
        elif 'bbox3d' in ann:
            bbox3d = np.array(ann['bbox3d']).reshape(8, 3)
        elif 'bbox3D' in ann:
            bbox3d = np.array(ann['bbox3D']).reshape(8, 3)
        
        if bbox3d is None:
            continue
        cat_id = ann['category_id']
        cat_name = categories.get(cat_id, f"id={cat_id}")
        
        gt_boxes.append(bbox3d)
        gt_labels.append(cat_name)
        gt_colors.append(get_category_color(cat_id))
    
    gt_image = draw_3d_boxes(gt_image, gt_boxes, gt_labels, gt_colors, K)
    
    # Create prediction visualization
    pred_image = image.copy()
    pred_boxes = []
    pred_labels = []
    pred_colors = []
    
    for pred in pred_anns:
        score = pred.get('score', 1.0)
        if score < score_thresh:
            continue
        
        # Try different field names for 3D box
        bbox3d = None
        if 'bbox3D' in pred:
            bbox3d = np.array(pred['bbox3D']).reshape(8, 3)
        elif 'bbox3d' in pred:
            bbox3d = np.array(pred['bbox3d']).reshape(8, 3)
        elif 'bbox3D_cam' in pred:
            bbox3d = np.array(pred['bbox3D_cam']).reshape(8, 3)
        
        if bbox3d is None:
            continue
        cat_id = pred['category_id']
        cat_name = categories.get(cat_id, f"id={cat_id}")
        
        pred_boxes.append(bbox3d)
        pred_labels.append(f"{cat_name}:{score:.2f}")
        pred_colors.append(get_category_color(cat_id))
    
    pred_image = draw_3d_boxes(pred_image, pred_boxes, pred_labels, pred_colors, K)
    
    # Add titles
    cv2.putText(gt_image, f"Ground Truth ({len(gt_boxes)} boxes)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(pred_image, f"Predictions ({len(pred_boxes)} boxes, thresh={score_thresh})", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Combine side by side
    combined = np.hstack([gt_image, pred_image])
    
    # Save
    cv2.imwrite(output_path, combined)
    return True


def main():
    parser = argparse.ArgumentParser(description="Visualize 3D bbox GT vs Predictions")
    parser.add_argument("--manifest", required=True, help="Path to manifest JSON")
    parser.add_argument("--predictions", required=True, help="Path to predictions JSON")
    parser.add_argument("--data-root", required=True, help="Root path for image files")
    parser.add_argument("--output-dir", required=True, help="Output directory for visualizations")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of samples to visualize")
    parser.add_argument("--score-thresh", type=float, default=0.1, help="Score threshold for predictions")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sample selection")
    parser.add_argument("--require-valid3d", action="store_true", help="Only show images with valid 3D GT annotations")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading manifest: {args.manifest}")
    images, categories, annotations = load_manifest(args.manifest)
    
    print(f"Loading predictions: {args.predictions}")
    predictions = load_predictions(args.predictions)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Filter for images with valid 3D GT if requested
    if args.require_valid3d:
        valid3d_images = set()
        for img_id, anns in annotations.items():
            if any(ann.get('valid3D', False) for ann in anns):
                valid3d_images.add(img_id)
        print(f"Images with valid 3D GT: {len(valid3d_images)}")
        image_ids = list(valid3d_images & set(predictions.keys()))
    else:
        image_ids = list(set(annotations.keys()) & set(predictions.keys()))
    
    # Select samples that have both GT and predictions
    np.random.seed(args.random_seed)
    
    if len(image_ids) == 0:
        print("No images with both GT and predictions found!")
        print(f"GT image IDs: {list(annotations.keys())[:10]}...")
        print(f"Pred image IDs: {list(predictions.keys())[:10]}...")
        return
    
    selected = np.random.choice(image_ids, min(args.num_samples, len(image_ids)), replace=False)
    
    print(f"Visualizing {len(selected)} samples...")
    
    success_count = 0
    for i, img_id in enumerate(selected):
        img_info = images[img_id]
        
        # Get image path
        if 'file_path' in img_info:
            image_path = os.path.join(args.data_root, img_info['file_path'])
        elif 'file_name' in img_info:
            image_path = os.path.join(args.data_root, img_info['file_name'])
        else:
            print(f"No file path found for image {img_id}")
            continue
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        # Get camera intrinsics
        K = img_info.get('K', [[500, 0, 320], [0, 500, 240], [0, 0, 1]])
        
        # Get GT and predictions
        gt_anns = annotations.get(img_id, [])
        pred_anns = predictions.get(img_id, [])
        
        # Output path
        output_path = os.path.join(args.output_dir, f"sample_{i:03d}_id{img_id}.jpg")
        
        if visualize_sample(image_path, gt_anns, pred_anns, categories, K, 
                           output_path, args.score_thresh):
            success_count += 1
            print(f"  [{i+1}/{len(selected)}] Saved: {output_path}")
    
    print(f"\nDone! Visualized {success_count} samples to {args.output_dir}")


if __name__ == "__main__":
    main()
