import os
import json
import cv2
import numpy as np
import torch
from cubercnn.vis.clean_vis import visualize_gt_and_predictions

# Paths
MANIFEST_PATH = "/mnt/data/users/anweshan/omni3d/data/unidet3d_format/cleaned_manifests/hypersim_train_sampled.json"
HYPERSIM_BASE = "/mnt/data/users/anweshan/omni3d/data/hypersim"
OUTPUT_DIR = "/mnt/data/users/anweshan/omni3d/test_viz"

print(f"Loading manifest from: {MANIFEST_PATH}")
with open(MANIFEST_PATH, 'r') as f:
    data = json.load(f)

# Build category mapping
categories = {cat['id']: cat['name'] for cat in data['categories']}
print(f"Found {len(categories)} categories")

# Get images and annotations
images = {img['id']: img for img in data['images']}
print(f"Found {len(images)} images")

# Group annotations by image
annotations_by_image = {}
for ann in data['annotations']:
    img_id = ann['image_id']
    if img_id not in annotations_by_image:
        annotations_by_image[img_id] = []
    annotations_by_image[img_id].append(ann)

print(f"Found annotations for {len(annotations_by_image)} images")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Visualize first 5 samples with annotations
samples = [img for img in data['images'] if img['id'] in annotations_by_image][:5]
print(f"\nVisualizing {len(samples)} samples...")

for i, sample in enumerate(samples):
    image_id = sample['id']
    
    # Get image path (handle both file_name and file_path)
    image_path = sample.get('file_name') or sample.get('file_path')
    if not image_path:
        print(f"No image path for sample {i}, skipping...")
        continue
    
    # Make path absolute if relative - remove duplicate 'hypersim' from path
    if not os.path.isabs(image_path):
        # Remove leading 'hypersim/' if present
        if image_path.startswith('hypersim/'):
            image_path = image_path[len('hypersim/'):]
        image_path = os.path.join(HYPERSIM_BASE, image_path)
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}, skipping...")
        continue
    
    print(f"\n[{i+1}/{len(samples)}] Processing image_id={image_id}")
    print(f"  Image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"  Failed to load image, skipping...")
        continue
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load depth if available
    depth = None
    depth_path = sample.get('depth_file') or sample.get('depth_file_path')
    if depth_path:
        if not os.path.isabs(depth_path):
            # Remove leading 'hypersim/' if present
            if depth_path.startswith('hypersim/'):
                depth_path = depth_path[len('hypersim/'):]
            depth_path = os.path.join(HYPERSIM_BASE, depth_path)
        if os.path.exists(depth_path):
            if depth_path.endswith('.npy'):
                depth = np.load(depth_path)
            else:
                depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)
    
    # Get camera intrinsics
    K = np.array(sample['K']).reshape(3, 3)
    
    # Get annotations
    annotations = annotations_by_image[image_id]
    print(f"  Found {len(annotations)} annotations")
    
    # Extract GT data
    gt_boxes3d_list = []
    gt_classes_list = []
    gt_poses_list = []
    
    for ann in annotations:
        if not ann.get('valid3D', True):
            continue
        
        # Use bbox3D_cam directly
        bbox3D_cam = np.array(ann['bbox3D_cam'])  # (8, 3)
        center_cam = np.array(ann['center_cam'])  # (3,)
        dimensions = np.array(ann['dimensions'])  # [w, h, l]
        
        # Project center to 2D
        center_2d_h = K @ center_cam
        u, v = center_2d_h[0] / center_2d_h[2], center_2d_h[1] / center_2d_h[2]
        
        # Create box format: [u, v, z, w, h, l, X, Y, Z]
        gt_box = np.concatenate([
            [u, v, center_cam[2]],
            dimensions,
            center_cam
        ])
        
        gt_boxes3d_list.append(gt_box)
        gt_classes_list.append(ann['category_id'])
        gt_poses_list.append(bbox3D_cam)  # Pass corners directly
    
    if len(gt_boxes3d_list) == 0:
        print(f"  No valid 3D annotations, skipping...")
        continue
    
    gt_boxes3d = np.array(gt_boxes3d_list)
    gt_classes = np.array(gt_classes_list)
    gt_poses = np.array(gt_poses_list)
    
    print(f"  Visualizing {len(gt_boxes3d)} valid 3D boxes")
    
    # Create visualization (GT only, no predictions)
    vis_image = visualize_gt_and_predictions(
        image=image,
        depth=depth,
        K=K,
        gt_boxes3d=gt_boxes3d,
        gt_classes=gt_classes,
        gt_poses=gt_poses,
        pred_boxes3d=None,
        pred_classes=None,
        pred_scores=None,
        pred_poses=None,
        class_names=categories,
        title=f"Hypersim Sample {i+1} (ID: {image_id})",
    )
    
    # Save visualization
    save_path = os.path.join(OUTPUT_DIR, f'sample_{i+1:02d}_id_{image_id}.png')
    cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    print(f"  Saved to: {save_path}")

print(f"\nDone! Visualizations saved to {OUTPUT_DIR}")
