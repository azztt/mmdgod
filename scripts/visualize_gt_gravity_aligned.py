#!/usr/bin/env python3
"""
Visualize ground truth bounding boxes assuming gravity alignment.

This script:
1. Loads annotations from manifest
2. Extracts yaw angle from R_cam (assuming gravity-aligned boxes)
3. Visualizes 3D boxes projected onto images
4. Verifies that gravity alignment assumption is valid

Usage:
    python scripts/visualize_gt_gravity_aligned.py
"""

import json
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from pathlib import Path
import argparse


def rotation_matrix_to_yaw(R):
    """
    Extract yaw angle from rotation matrix assuming gravity alignment.
    
    For gravity-aligned boxes in camera coordinates:
    - Camera: X=right, Y=down, Z=forward
    - Yaw is rotation around Y-axis (vertical)
    - R represents rotation from object frame to camera frame
    
    For rotation around Y-axis: R_y(θ) = [[cos(θ), 0, sin(θ)],
                                           [0,      1,  0     ],
                                           [-sin(θ), 0, cos(θ)]]
    
    Args:
        R: 3x3 rotation matrix
        
    Returns:
        yaw: Yaw angle in radians (rotation around Y)
        is_aligned: Whether object appears gravity-aligned (Y-axis mostly vertical)
        y_alignment: How aligned the Y-axis is with vertical (0-1, 1=perfect)
    """
    R = np.array(R)
    
    # Check alignment by looking at Y-axis column (should be ~[0, ±1, 0])
    y_axis = R[:, 1]  # Y-axis after rotation
    y_alignment = abs(y_axis[1])  # Should be close to 1 for gravity-aligned
    
    # Extract yaw from R[0,2] and R[2,2] (assuming Y-axis rotation)
    # For R_y: R[0,2] = sin(yaw), R[2,2] = cos(yaw)
    yaw = np.arctan2(R[0, 2], R[2, 2])
    
    # Gravity-aligned if Y-axis is mostly vertical (y_alignment > 0.95)
    is_aligned = y_alignment > 0.95
    
    # Additional check: X and Z components of Y-axis should be small
    y_horizontal_mag = np.sqrt(y_axis[0]**2 + y_axis[2]**2)
    
    return yaw, is_aligned, y_alignment, y_horizontal_mag


def yaw_to_rotation_matrix(yaw):
    """
    Convert yaw angle to rotation matrix (gravity-aligned).
    
    Rotation around Y-axis (vertical):
    R_y(yaw) = [[cos(yaw),  0, sin(yaw)],
                [0,         1, 0       ],
                [-sin(yaw), 0, cos(yaw)]]
    
    Args:
        yaw: Yaw angle in radians
        
    Returns:
        R: 3x3 rotation matrix
    """
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array([
        [c,  0, s],
        [0,  1, 0],
        [-s, 0, c]
    ])


def get_box_corners_3d(center, dims, yaw):
    """
    Get 8 corners of 3D bounding box.
    
    Args:
        center: (3,) center in camera coordinates [x, y, z]
        dims: (3,) dimensions [width, height, depth]
        yaw: Yaw angle in radians
        
    Returns:
        corners: (8, 3) corner coordinates
    """
    # Create box in canonical orientation (aligned with axes)
    w, h, d = dims
    x_corners = [w/2, w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2]
    y_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
    z_corners = [d/2, -d/2, -d/2, d/2, d/2, -d/2, -d/2, d/2]
    corners = np.array([x_corners, y_corners, z_corners])  # (3, 8)
    
    # Rotate by yaw
    R = yaw_to_rotation_matrix(yaw)
    corners = R @ corners  # (3, 8)
    
    # Translate to center
    corners = corners.T + center  # (8, 3)
    
    return corners


def project_3d_to_2d(points_3d, K):
    """
    Project 3D points to 2D image coordinates.
    
    Args:
        points_3d: (N, 3) points in camera coordinates
        K: (3, 3) camera intrinsic matrix
        
    Returns:
        points_2d: (N, 2) image coordinates
    """
    points_3d = np.array(points_3d)
    K = np.array(K)
    
    # Project: [u, v, 1]^T = K * [x, y, z]^T / z
    points_2d = K @ points_3d.T  # (3, N)
    points_2d = points_2d[:2] / points_2d[2]  # (2, N)
    
    return points_2d.T  # (N, 2)


def draw_box_3d(img, corners_2d, color=(0, 255, 0), thickness=2):
    """
    Draw 3D bounding box on image.
    
    Args:
        img: Image array
        corners_2d: (8, 2) projected corners
        color: Box color (B, G, R)
        thickness: Line thickness
    """
    corners_2d = corners_2d.astype(int)
    
    # Draw front face (indices 0,1,2,3)
    for i in range(4):
        cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[(i+1)%4]), color, thickness)
    
    # Draw back face (indices 4,5,6,7)
    for i in range(4):
        cv2.line(img, tuple(corners_2d[i+4]), tuple(corners_2d[(i+1)%4+4]), color, thickness)
    
    # Draw connections between front and back
    for i in range(4):
        cv2.line(img, tuple(corners_2d[i]), tuple(corners_2d[i+4]), color, thickness)
    
    # Draw front face in different color to show orientation
    cv2.line(img, tuple(corners_2d[0]), tuple(corners_2d[1]), (0, 0, 255), thickness+1)
    
    return img


def visualize_samples(manifest_path, data_root, num_samples=5, output_dir="vis_gt_gravity"):
    """
    Visualize ground truth bounding boxes with gravity alignment.
    """
    print(f"Loading manifest: {manifest_path}")
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    images_info = {img['id']: img for img in data['images']}
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Statistics
    total_boxes = 0
    aligned_boxes = 0
    yaw_angles = []
    y_alignments = []
    y_horizontal_mags = []
    
    print(f"\nVisualizing {num_samples} samples...")
    
    for sample_idx in range(num_samples):
        # Get random image with annotations
        img_id = data['images'][sample_idx * 100]['id']
        img_info = images_info[img_id]
        
        # Load image
        img_path = os.path.join(data_root, img_info['file_path'])
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue
            
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        K = np.array(img_info['K'])
        
        # Get annotations for this image
        anns = [ann for ann in data['annotations'] if ann['image_id'] == img_id]
        
        print(f"\nSample {sample_idx + 1}: Image {img_id}, {len(anns)} objects")
        
        for ann in anns:
            if not ann.get('valid3D', False):
                continue
                
            total_boxes += 1
            
            # Extract GT data
            R_cam = np.array(ann['R_cam'])
            center = np.array(ann['center_cam'])
            dims = np.array(ann['dimensions'])
            
            # Extract yaw and check alignment
            yaw, is_aligned, y_alignment, y_horiz_mag = rotation_matrix_to_yaw(R_cam)
            
            yaw_angles.append(yaw)
            y_alignments.append(y_alignment)
            y_horizontal_mags.append(y_horiz_mag)
            
            if is_aligned:
                aligned_boxes += 1
                color = (0, 255, 0)  # Green for aligned
            else:
                color = (0, 165, 255)  # Orange for not aligned
            
            # Get corners using GT rotation matrix
            corners_3d_gt = np.array(ann['bbox3D_cam'])
            corners_2d_gt = project_3d_to_2d(corners_3d_gt, K)
            
            # Get corners using yaw-only rotation
            corners_3d_yaw = get_box_corners_3d(center, dims, yaw)
            corners_2d_yaw = project_3d_to_2d(corners_3d_yaw, K)
            
            # Draw GT box (dashed) and yaw-only box (solid)
            img = draw_box_3d(img, corners_2d_gt, color=(128, 128, 128), thickness=1)
            img = draw_box_3d(img, corners_2d_yaw, color=color, thickness=2)
            
            # Add text
            text = f"{ann['category_name']}: yaw={np.degrees(yaw):.1f}°"
            if not is_aligned:
                text += f" (y_align={y_alignment:.2f})"
            
            cv2.putText(img, text, (int(corners_2d_yaw[0, 0]), int(corners_2d_yaw[0, 1])-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Save visualization
        out_path = os.path.join(output_dir, f"sample_{sample_idx:03d}.jpg")
        cv2.imwrite(out_path, img)
        print(f"Saved: {out_path}")
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"GRAVITY ALIGNMENT STATISTICS")
    print(f"{'='*60}")
    print(f"Total boxes analyzed: {total_boxes}")
    print(f"Gravity-aligned boxes (y_align>0.95): {aligned_boxes} ({aligned_boxes/max(total_boxes,1)*100:.1f}%)")
    print(f"\nStatistics:")
    print(f"  Yaw angle:         mean={np.degrees(np.mean(yaw_angles)):.1f}°, std={np.degrees(np.std(yaw_angles)):.1f}°")
    print(f"  Y-axis alignment:  mean={np.mean(y_alignments):.3f}, std={np.std(y_alignments):.3f} (1.0=perfect)")
    print(f"  Y horiz. magnitude: mean={np.mean(y_horizontal_mags):.3f}, std={np.std(y_horizontal_mags):.3f} (0.0=perfect)")
    
    # Plot distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(np.degrees(yaw_angles), bins=30, edgecolor='black')
    axes[0].set_title('Yaw Distribution')
    axes[0].set_xlabel('Yaw (degrees)')
    axes[0].set_ylabel('Count')
    axes[0].axvline(0, color='r', linestyle='--', label='Zero')
    axes[0].legend()
    
    axes[1].hist(y_alignments, bins=30, edgecolor='black')
    axes[1].set_title('Y-Axis Alignment (should be near 1.0)')
    axes[1].set_xlabel('Y-axis alignment')
    axes[1].set_ylabel('Count')
    axes[1].axvline(1.0, color='r', linestyle='--', label='Perfect')
    axes[1].axvspan(0.95, 1.0, alpha=0.2, color='green', label='Aligned (>0.95)')
    axes[1].legend()
    
    axes[2].hist(y_horizontal_mags, bins=30, edgecolor='black')
    axes[2].set_title('Y-Axis Horizontal Magnitude (should be near 0)')
    axes[2].set_xlabel('Horizontal magnitude')
    axes[2].set_ylabel('Count')
    axes[2].axvline(0, color='r', linestyle='--', label='Zero')
    axes[2].axvspan(0, 0.1, alpha=0.2, color='green', label='Aligned (<0.1)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'angle_distributions.png'), dpi=150)
    print(f"\nSaved angle distribution plot: {os.path.join(output_dir, 'angle_distributions.png')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, 
                       default='/mnt/data/users/anweshan/omni3d/data/unidet3d_format/cleaned_manifests/hypersim_train_filtered.json',
                       help='Path to manifest JSON file')
    parser.add_argument('--data-root', type=str,
                       default='/mnt/data/users/anweshan/omni3d/data',
                       help='Root directory for data files')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to visualize')
    parser.add_argument('--output-dir', type=str, default='vis_gt_gravity',
                       help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    visualize_samples(args.manifest, args.data_root, args.num_samples, args.output_dir)
