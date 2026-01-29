#!/usr/bin/env python3
"""
Verify yaw extraction with camera extrinsics.

This script:
1. Loads camera extrinsics for Hypersim and SUNRGBD
2. Transforms object rotations from camera frame to world frame
3. Extracts yaw in world coordinates
4. Compares gravity alignment in camera vs world frame
"""

import json
import numpy as np
import os
import h5py
from pathlib import Path
import argparse


def load_hypersim_camera_extrinsics(scene_path, frame_idx):
    """
    Load camera extrinsics for Hypersim scene.
    
    Args:
        scene_path: Path to scene (e.g., 'hypersim/ai_001_001')
        frame_idx: Frame index (e.g., 0)
        
    Returns:
        R_world_to_cam: (3, 3) rotation from world to camera
        t_world_to_cam: (3,) translation from world to camera
    """
    # Extract scene and camera names
    # e.g., 'hypersim/ai_001_001/images/scene_cam_00_final_preview/frame.0000.rgb.png'
    # -> scene: ai_001_001, cam: cam_00
    
    parts = scene_path.split('/')
    scene_name = parts[1]  # ai_001_001
    
    # Camera is in the images path
    detail_path = f'/mnt/data/users/anweshan/omni3d/data/hypersim/{scene_name}/_detail/cam_00'
    
    # Load frame indices
    frame_indices_file = os.path.join(detail_path, 'camera_keyframe_frame_indices.hdf5')
    if not os.path.exists(frame_indices_file):
        return None, None
    
    with h5py.File(frame_indices_file, 'r') as f:
        frame_indices = f['dataset'][:]
    
    # Find which keyframe corresponds to this frame
    keyframe_idx = np.where(frame_indices == frame_idx)[0]
    if len(keyframe_idx) == 0:
        # Frame is not a keyframe, need interpolation (skip for now)
        return None, None
    keyframe_idx = keyframe_idx[0]
    
    # Load orientation (already a 3x3 rotation matrix!)
    orientations_file = os.path.join(detail_path, 'camera_keyframe_orientations.hdf5')
    with h5py.File(orientations_file, 'r') as f:
        R_world_to_cam = f['dataset'][keyframe_idx]  # Already a 3x3 matrix
    
    # Load position
    positions_file = os.path.join(detail_path, 'camera_keyframe_positions.hdf5')
    with h5py.File(positions_file, 'r') as f:
        t_world_to_cam = f['dataset'][keyframe_idx]
    
    return R_world_to_cam, t_world_to_cam


def load_sunrgbd_camera_extrinsics(extrinsics_path):
    """
    Load camera extrinsics for SUNRGBD.
    
    Args:
        extrinsics_path: Path to extrinsics file
        
    Returns:
        R_world_to_cam: (3, 3) rotation from world to camera
    """
    if not os.path.exists(extrinsics_path):
        return None
    
    # Load 3x4 extrinsics matrix [R | t]
    extrinsics = np.loadtxt(extrinsics_path)
    R_world_to_cam = extrinsics[:3, :3]
    
    return R_world_to_cam


def extract_yaw_in_world(R_cam, R_world_to_cam):
    """
    Extract yaw angle in world coordinates.
    
    Args:
        R_cam: (3, 3) object rotation in camera frame
        R_world_to_cam: (3, 3) camera rotation (world to camera)
        
    Returns:
        yaw: Yaw angle in world frame (radians)
        is_aligned: Whether gravity-aligned in world frame
        y_alignment: Y-axis alignment score
    """
    # Transform object rotation to world frame
    # R_world = R_cam_to_world @ R_cam
    # where R_cam_to_world = R_world_to_cam.T
    R_cam_to_world = R_world_to_cam.T
    R_world = R_cam_to_world @ R_cam
    
    # Extract yaw assuming gravity is world Y-axis
    # For rotation around Y-axis: R_y(θ) = [[cos(θ), 0, sin(θ)],
    #                                        [0,      1,  0     ],
    #                                        [-sin(θ), 0, cos(θ)]]
    yaw = np.arctan2(R_world[0, 2], R_world[2, 2])
    
    # Check alignment by looking at Y-axis column (should be ~[0, ±1, 0] in world)
    y_axis_world = R_world[:, 1]
    y_alignment = abs(y_axis_world[1])
    
    is_aligned = y_alignment > 0.95
    
    return yaw, is_aligned, y_alignment


def extract_yaw_in_camera(R_cam):
    """
    Extract yaw angle in camera coordinates (for comparison).
    """
    yaw = np.arctan2(R_cam[0, 2], R_cam[2, 2])
    y_axis = R_cam[:, 1]
    y_alignment = abs(y_axis[1])
    is_aligned = y_alignment > 0.95
    
    return yaw, is_aligned, y_alignment


def verify_datasets(manifest_path, data_root, num_samples=50):
    """
    Verify yaw extraction with camera extrinsics.
    """
    print(f"Loading manifest: {manifest_path}")
    with open(manifest_path, 'r') as f:
        data = json.load(f)
    
    images_info = {img['id']: img for img in data['images']}
    
    # Separate by dataset
    hypersim_stats = {
        'total': 0, 'loaded_extrinsics': 0,
        'aligned_in_cam': 0, 'aligned_in_world': 0,
        'yaw_diffs': [], 'y_align_cam': [], 'y_align_world': []
    }
    
    sunrgbd_stats = {
        'total': 0, 'loaded_extrinsics': 0,
        'aligned_in_cam': 0, 'aligned_in_world': 0,
        'yaw_diffs': [], 'y_align_cam': [], 'y_align_world': []
    }
    
    print(f"\nProcessing {num_samples} samples...")
    
    sample_count = 0
    for ann in data['annotations']:
        if sample_count >= num_samples:
            break
        
        if not ann.get('valid3D', False):
            continue
        
        img_info = images_info[ann['image_id']]
        file_path = img_info['file_path']
        
        R_cam = np.array(ann['R_cam'])
        
        # Extract yaw in camera frame
        yaw_cam, aligned_cam, y_align_cam = extract_yaw_in_camera(R_cam)
        
        # Try to load camera extrinsics
        R_world_to_cam = None
        
        if 'hypersim' in file_path:
            stats = hypersim_stats
            stats['total'] += 1
            
            # Extract frame number from path
            # e.g., 'hypersim/ai_001_001/images/scene_cam_00_final_preview/frame.0000.rgb.png'
            frame_str = file_path.split('frame.')[-1].split('.')[0]
            frame_idx = int(frame_str)
            
            R_world_to_cam, _ = load_hypersim_camera_extrinsics(file_path, frame_idx)
            
        elif 'SUNRGBD' in file_path or 'sunrgbd' in file_path:
            stats = sunrgbd_stats
            stats['total'] += 1
            
            # Extract extrinsics path from file_path
            # file_path structure: 'SUNRGBD/.../000008_2014-05-26_14-30-06_260595134347_rgbf000060-resize/image/0000060.jpg'
            # extrinsics: 'SUNRGBD/.../000008_2014-05-26_14-30-06_260595134347_rgbf000060-resize/extrinsics/*.txt'
            
            path_parts = file_path.split('/')
            scene_dir = '/'.join(path_parts[:-2])  # Remove 'image/0000060.jpg'
            extrinsics_dir = os.path.join(data_root, scene_dir, 'extrinsics')
            
            if os.path.exists(extrinsics_dir):
                # Get first .txt file (usually only one)
                extrinsics_files = [f for f in os.listdir(extrinsics_dir) if f.endswith('.txt')]
                if extrinsics_files:
                    extrinsics_path = os.path.join(extrinsics_dir, extrinsics_files[0])
                    R_world_to_cam = load_sunrgbd_camera_extrinsics(extrinsics_path)
        else:
            continue
        
        # Store camera frame stats
        stats['y_align_cam'].append(y_align_cam)
        if aligned_cam:
            stats['aligned_in_cam'] += 1
        
        # If we have extrinsics, compute world frame stats
        if R_world_to_cam is not None:
            stats['loaded_extrinsics'] += 1
            
            yaw_world, aligned_world, y_align_world = extract_yaw_in_world(R_cam, R_world_to_cam)
            
            stats['y_align_world'].append(y_align_world)
            if aligned_world:
                stats['aligned_in_world'] += 1
            
            # Store yaw difference
            yaw_diff = np.abs(yaw_world - yaw_cam)
            # Wrap to [-pi, pi]
            yaw_diff = (yaw_diff + np.pi) % (2 * np.pi) - np.pi
            stats['yaw_diffs'].append(np.abs(yaw_diff))
        
        sample_count += 1
    
    # Print results
    print("\n" + "="*70)
    print("HYPERSIM RESULTS")
    print("="*70)
    print(f"Total objects: {hypersim_stats['total']}")
    print(f"Loaded extrinsics: {hypersim_stats['loaded_extrinsics']}/{hypersim_stats['total']}")
    
    if len(hypersim_stats['y_align_cam']) > 0:
        print(f"\nCamera frame:")
        print(f"  Aligned (>0.95): {hypersim_stats['aligned_in_cam']}/{len(hypersim_stats['y_align_cam'])} "
              f"({hypersim_stats['aligned_in_cam']/len(hypersim_stats['y_align_cam'])*100:.1f}%)")
        print(f"  Y-alignment: {np.mean(hypersim_stats['y_align_cam']):.3f} ± {np.std(hypersim_stats['y_align_cam']):.3f}")
    
    if len(hypersim_stats['y_align_world']) > 0:
        print(f"\nWorld frame (with extrinsics):")
        print(f"  Aligned (>0.95): {hypersim_stats['aligned_in_world']}/{len(hypersim_stats['y_align_world'])} "
              f"({hypersim_stats['aligned_in_world']/len(hypersim_stats['y_align_world'])*100:.1f}%)")
        print(f"  Y-alignment: {np.mean(hypersim_stats['y_align_world']):.3f} ± {np.std(hypersim_stats['y_align_world']):.3f}")
        print(f"  Yaw difference (cam vs world): {np.degrees(np.mean(hypersim_stats['yaw_diffs'])):.1f}° ± "
              f"{np.degrees(np.std(hypersim_stats['yaw_diffs'])):.1f}°")
    
    print("\n" + "="*70)
    print("SUNRGBD RESULTS")
    print("="*70)
    print(f"Total objects: {sunrgbd_stats['total']}")
    print(f"Loaded extrinsics: {sunrgbd_stats['loaded_extrinsics']}/{sunrgbd_stats['total']}")
    
    if len(sunrgbd_stats['y_align_cam']) > 0:
        print(f"\nCamera frame:")
        print(f"  Aligned (>0.95): {sunrgbd_stats['aligned_in_cam']}/{len(sunrgbd_stats['y_align_cam'])} "
              f"({sunrgbd_stats['aligned_in_cam']/len(sunrgbd_stats['y_align_cam'])*100:.1f}%)")
        print(f"  Y-alignment: {np.mean(sunrgbd_stats['y_align_cam']):.3f} ± {np.std(sunrgbd_stats['y_align_cam']):.3f}")
    
    if len(sunrgbd_stats['y_align_world']) > 0:
        print(f"\nWorld frame (with extrinsics):")
        print(f"  Aligned (>0.95): {sunrgbd_stats['aligned_in_world']}/{len(sunrgbd_stats['y_align_world'])} "
              f"({sunrgbd_stats['aligned_in_world']/len(sunrgbd_stats['y_align_world'])*100:.1f}%)")
        print(f"  Y-alignment: {np.mean(sunrgbd_stats['y_align_world']):.3f} ± {np.std(sunrgbd_stats['y_align_world']):.3f}")
        print(f"  Yaw difference (cam vs world): {np.degrees(np.mean(sunrgbd_stats['yaw_diffs'])):.1f}° ± "
              f"{np.degrees(np.std(sunrgbd_stats['yaw_diffs'])):.1f}°")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    total_cam = len(hypersim_stats['y_align_cam']) + len(sunrgbd_stats['y_align_cam'])
    total_world = len(hypersim_stats['y_align_world']) + len(sunrgbd_stats['y_align_world'])
    
    if total_cam > 0:
        all_y_align_cam = hypersim_stats['y_align_cam'] + sunrgbd_stats['y_align_cam']
        all_aligned_cam = hypersim_stats['aligned_in_cam'] + sunrgbd_stats['aligned_in_cam']
        print(f"\nCamera frame (all datasets):")
        print(f"  Aligned: {all_aligned_cam}/{total_cam} ({all_aligned_cam/total_cam*100:.1f}%)")
        print(f"  Mean Y-alignment: {np.mean(all_y_align_cam):.3f}")
    
    if total_world > 0:
        all_y_align_world = hypersim_stats['y_align_world'] + sunrgbd_stats['y_align_world']
        all_aligned_world = hypersim_stats['aligned_in_world'] + sunrgbd_stats['aligned_in_world']
        all_yaw_diffs = hypersim_stats['yaw_diffs'] + sunrgbd_stats['yaw_diffs']
        print(f"\nWorld frame (all datasets):")
        print(f"  Aligned: {all_aligned_world}/{total_world} ({all_aligned_world/total_world*100:.1f}%)")
        print(f"  Mean Y-alignment: {np.mean(all_y_align_world):.3f}")
        print(f"  Mean yaw error: {np.degrees(np.mean(all_yaw_diffs)):.1f}°")
        
        # Conclusion
        improvement = all_aligned_world - all_aligned_cam
        print(f"\n{'='*70}")
        if improvement > 0:
            print(f"✓ Using world frame IMPROVES alignment by {improvement} objects ({improvement/total_cam*100:.1f}%)")
            print(f"  Recommendation: Use camera extrinsics for better gravity alignment")
        else:
            print(f"✗ Using world frame provides NO IMPROVEMENT (or worse)")
            print(f"  Recommendation: Camera frame is sufficient, skip extrinsics")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=str, 
                       default='/mnt/data/users/anweshan/omni3d/data/unidet3d_format/cleaned_manifests/hypersim_train_sampled.json',
                       help='Path to manifest JSON file')
    parser.add_argument('--data-root', type=str,
                       default='/mnt/data/users/anweshan/omni3d/data',
                       help='Root directory for data files')
    parser.add_argument('--num-samples', type=int, default=100,
                       help='Number of samples to process')
    
    args = parser.parse_args()
    
    verify_datasets(args.manifest, args.data_root, args.num_samples)
