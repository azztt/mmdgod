#!/usr/bin/env python3
"""
Quick test script to generate visualizations from sampled manifests.
"""

import os
import json
import numpy as np
import cv2
import torch
from cubercnn.vis.clean_vis import visualize_gt_and_predictions
from cubercnn.util import math_util as util

# Paths
MANIFEST_DIR = "/mnt/data/users/anweshan/omni3d/data/unidet3d_format/cleaned_manifests"
OUTPUT_DIR = "/mnt/data/users/anweshan/omni3d/test_viz"

def visualize_samples(manifest_path, num_samples=5):
    """Generate visualizations from a manifest."""
    
    # Load manifest
    print(f"\nLoading manifest: {manifest_path}")
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    images = {img['id']: img for img in data['images']}
    
    # Group annotations by image
    anns_by_image = {}
    for ann in data['annotations']:
        if ann.get('valid3D', False):
            img_id = ann['image_id']
            if img_id not in anns_by_image:
                anns_by_image[img_id] = []
            anns_by_image[img_id].append(ann)
    
    # Get category mapping
    cat_id_to_name = {cat['id']: cat['name'] for cat in data['categories']}
    
    print(f"Found {len(anns_by_image)} images with valid 3D annotations")
    
    # Sample images to visualize
    image_ids = list(anns_by_image.keys())[:num_samples]
    
    # Create output directory
    dataset_name = os.path.basename(manifest_path).replace('.json', '')
    output_subdir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(output_subdir, exist_ok=True)
    
    for idx, img_id in enumerate(image_ids):
        try:
            img_info = images[img_id]
            annotations = anns_by_image[img_id]
            
            # Load image (handle both file_name and file_path)
            image_path = img_info.get('file_name') or img_info.get('file_path')
            if not image_path or not os.path.exists(image_path):
                print(f"  Image not found: {image_path}")
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"  Failed to load: {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Load depth if available
            depth = None
            depth_path = img_info.get('depth_file') or img_info.get('depth_file_path')
            if depth_path:
                if os.path.exists(depth_path):
                    if depth_path.endswith('.npy'):
                        depth = np.load(depth_path)
                    else:
                        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
            
            # Get camera intrinsics
            K = np.array(img_info['K']).reshape(3, 3)
            
            # Extract GT data
            gt_boxes3d_list = []
            gt_classes_list = []
            gt_poses_list = []
            
            for ann in annotations:
                # Use bbox3D_cam directly (ground truth corners)
                bbox3D_cam = np.array(ann['bbox3D_cam'])  # (8, 3)
                center_cam = np.array(ann['center_cam'])  # (3,)
                dimensions = np.array(ann['dimensions'])  # [w, h, l]
                
                # Project center to 2D
                center_2d_h = K @ center_cam
                u, v = center_2d_h[0] / center_2d_h[2], center_2d_h[1] / center_2d_h[2]
                
                # Create box format: [u, v, z, w, h, l, X, Y, Z]
                gt_box = np.concatenate([
                    [u, v, center_cam[2]],  # u, v, z
                    dimensions,              # w, h, l
                    center_cam               # X, Y, Z
                ])
                
                gt_boxes3d_list.append(gt_box)
                gt_classes_list.append(ann['category_id'])
                gt_poses_list.append(bbox3D_cam)  # Pass corners directly
            
            if len(gt_boxes3d_list) == 0:
                continue
            
            gt_boxes3d = np.array(gt_boxes3d_list)
            gt_classes = np.array(gt_classes_list)
            gt_poses = np.array(gt_poses_list)
            
            # Create visualization (GT only, no predictions)
            vis_image = visualize_gt_and_predictions(
                image=image,
                depth=depth,
                K=K,
                gt_boxes3d=gt_boxes3d,
                gt_classes=gt_classes,
                gt_poses=gt_poses,
                pred_boxes3d=None,  # No predictions for now
                pred_classes=None,
                pred_scores=None,
                pred_poses=None,
                class_names=cat_id_to_name,
                title=f"{dataset_name} - Image {img_id} ({len(annotations)} boxes)",
            )
            
            # Save
            save_path = os.path.join(output_subdir, f'sample_{idx:03d}_id_{img_id}.png')
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"  ✓ Saved: {save_path}")
            
        except Exception as e:
            print(f"  ✗ Error on image {img_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nVisualizations saved to: {output_subdir}")


if __name__ == "__main__":
    # Find all manifests
    manifests = [
        os.path.join(MANIFEST_DIR, f) 
        for f in os.listdir(MANIFEST_DIR) 
        if f.endswith('.json')
    ]
    
    print(f"Found {len(manifests)} manifests:")
    for m in manifests:
        print(f"  - {os.path.basename(m)}")
    
    # Process each manifest
    for manifest_path in manifests:
        visualize_samples(manifest_path, num_samples=5)
    
    print("\n✓ All done!")
