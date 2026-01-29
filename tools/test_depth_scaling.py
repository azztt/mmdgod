#!/usr/bin/env python3
"""
Test script to validate depth distribution hypothesis.
Scales SUNRGBD depth to match Hypersim distribution and checks if predictions improve.

Hypothesis: Model trained on Hypersim (mean depth 6.81m) expects that depth range.
SUNRGBD has mean depth 2.75m, so we scale it up by ~2.5x before inference,
then scale predictions back down.
"""

import json
import numpy as np
import torch
from collections import defaultdict
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import DatasetCatalog, MetadataCatalog

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, RCNN3D_RGBD
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone, build_dual_encoder_fpn_backbone
from cubercnn.data import (
    DatasetMapper3D_RGBD,
    register_and_store_rgbd_model_metadata,
)

import cv2
from PIL import Image


def compute_recall(predictions, gt_file, name, score_thresh=0.3):
    """Compute recall at different IoU thresholds."""
    gt = json.load(open(gt_file))
    
    # Build GT dict by image
    gt_by_image = defaultdict(list)
    for ann in gt['annotations']:
        if ann.get('valid3D', False):
            gt_by_image[ann['image_id']].append(ann)
    
    def box3d_iou(pred_corners, gt_corners):
        """Axis-aligned bounding box IoU"""
        pred_min = np.min(pred_corners, axis=0)
        pred_max = np.max(pred_corners, axis=0)
        gt_min = np.min(gt_corners, axis=0)
        gt_max = np.max(gt_corners, axis=0)
        
        inter_min = np.maximum(pred_min, gt_min)
        inter_max = np.minimum(pred_max, gt_max)
        inter_dims = np.maximum(inter_max - inter_min, 0)
        inter_vol = np.prod(inter_dims)
        
        pred_vol = np.prod(pred_max - pred_min)
        gt_vol = np.prod(gt_max - gt_min)
        union_vol = pred_vol + gt_vol - inter_vol
        
        return inter_vol / (union_vol + 1e-6)
    
    total_gt, matched_gt15, matched_gt25 = 0, 0, 0
    pred_depths = []
    
    for img_id, preds in predictions.items():
        img_preds = [p for p in preds if p['score'] > score_thresh]
        img_gt = gt_by_image.get(img_id, [])
        total_gt += len(img_gt)
        
        for p in img_preds:
            pred_depths.append(p['depth'])
        
        for g in img_gt:
            gt_corners = np.array(g['bbox3D_cam'])
            best_iou = 0
            for p in img_preds:
                pred_corners = np.array(p['bbox3D'])
                iou = box3d_iou(pred_corners, gt_corners)
                best_iou = max(best_iou, iou)
            if best_iou > 0.15:
                matched_gt15 += 1
            if best_iou > 0.25:
                matched_gt25 += 1
    
    gt_depths = [a['center_cam'][2] for a in gt['annotations'] if a.get('valid3D', False)]
    
    print(f'{name}:')
    print(f'  GT: {total_gt}, Preds: {sum(len(v) for v in predictions.values())} (score>{score_thresh}: {len(pred_depths)})')
    print(f'  Recall@15: {matched_gt15/max(total_gt,1)*100:.2f}%')
    print(f'  Recall@25: {matched_gt25/max(total_gt,1)*100:.2f}%')
    if pred_depths:
        print(f'  Pred depth: mean={np.mean(pred_depths):.2f}, median={np.median(pred_depths):.2f}')
    print(f'  GT depth:   mean={np.mean(gt_depths):.2f}, median={np.median(gt_depths):.2f}')
    print()
    
    return matched_gt15 / max(total_gt, 1)


def run_inference_with_scaled_depth(cfg, checkpoint, manifest_file, depth_scale, depth_max=8.0, max_images=100):
    """
    Run inference with scaled depth input.
    
    Args:
        depth_scale: Factor to multiply depth by before feeding to model
        After inference, predictions are scaled back by 1/depth_scale
    """
    print(f"\n{'='*60}")
    print(f"Testing with depth_scale={depth_scale:.2f}")
    print(f"{'='*60}")
    
    # Load manifest
    manifest = json.load(open(manifest_file))
    images_by_id = {img['id']: img for img in manifest['images']}
    
    # Group annotations by image
    anns_by_image = defaultdict(list)
    for ann in manifest['annotations']:
        anns_by_image[ann['image_id']].append(ann)
    
    # Setup config
    cfg = cfg.clone()
    cfg.MODEL.WEIGHTS = checkpoint
    cfg.MODEL.DEPTH_MAX = depth_max
    cfg.freeze()
    
    # Build model
    from detectron2.modeling import build_model
    from detectron2.checkpoint import DetectionCheckpointer
    
    model = build_model(cfg)
    model.eval()
    
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(checkpoint)
    
    predictions = {}
    
    # Get image IDs to process
    image_ids = list(images_by_id.keys())[:max_images]
    
    print(f"Processing {len(image_ids)} images...")
    
    for i, img_id in enumerate(image_ids):
        if i % 20 == 0:
            print(f"  Processing image {i+1}/{len(image_ids)}")
        
        img_info = images_by_id[img_id]
        
        # Load RGB image
        rgb_path = img_info['file_path']
        if not os.path.isabs(rgb_path):
            rgb_path = os.path.join(cfg.DATASETS.DATA_ROOT, rgb_path)
        
        image = cv2.imread(rgb_path)
        if image is None:
            print(f"  Warning: Could not load {rgb_path}")
            continue
        
        # Load depth
        depth_path = img_info.get('depth_file_path', img_info.get('depth_path', ''))
        if not depth_path:
            print(f"  Warning: No depth path for image {img_id}")
            predictions[img_id] = []
            continue
        if not os.path.isabs(depth_path):
            depth_path = os.path.join(cfg.DATASETS.DATA_ROOT, depth_path)
        
        if depth_path.endswith('.npy'):
            depth = np.load(depth_path)
        else:
            depth = np.array(Image.open(depth_path)).astype(np.float32)
            # SUNRGBD depth is in mm, convert to meters
            depth = depth / 1000.0
        
        # Handle NaN/inf values
        depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale depth
        depth_scaled = depth * depth_scale
        
        # Normalize depth for model input
        depth_normalized = np.clip(depth_scaled / depth_max, 0, 1)
        depth_normalized = (depth_normalized * 2) - 1  # Scale to [-1, 1]
        
        # Prepare input
        height, width = image.shape[:2]
        
        # Get camera intrinsics
        K = np.array(img_info.get('K', [[500, 0, width/2], [0, 500, height/2], [0, 0, 1]]))
        
        inputs = {
            "image": torch.as_tensor(image.transpose(2, 0, 1).astype("float32")),
            "depth": torch.as_tensor(depth_normalized.astype("float32")).unsqueeze(0),
            "height": height,
            "width": width,
            "K": torch.as_tensor(K.astype("float32")),
            "image_id": img_id,
        }
        
        # Run inference
        with torch.no_grad():
            outputs = model([inputs])[0]
        
        # Extract predictions
        instances = outputs.get("instances", None)
        if instances is None or len(instances) == 0:
            predictions[img_id] = []
            continue
        
        img_preds = []
        for j in range(len(instances)):
            pred = {
                'score': instances.scores[j].item(),
                'category_id': instances.pred_classes[j].item(),
            }
            
            # Get 3D box and scale back
            if hasattr(instances, 'pred_bbox3D'):
                bbox3D = instances.pred_bbox3D[j].cpu().numpy()
                # Scale Z coordinates back by 1/depth_scale
                bbox3D[:, 2] = bbox3D[:, 2] / depth_scale
                pred['bbox3D'] = bbox3D.tolist()
                pred['depth'] = np.mean(bbox3D[:, 2])
            
            if hasattr(instances, 'pred_center_cam'):
                center = instances.pred_center_cam[j].cpu().numpy()
                center[2] = center[2] / depth_scale
                pred['center_cam'] = center.tolist()
                pred['depth'] = center[2]
            
            img_preds.append(pred)
        
        predictions[img_id] = img_preds
    
    return predictions


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", default="output/rgbd_basic_aug/config.yaml")
    parser.add_argument("--checkpoint", default="output/rgbd_basic_aug/model_recent.pth")
    parser.add_argument("--manifest-dir", default="data/unidet3d_format/cleaned_manifests")
    parser.add_argument("--max-images", type=int, default=100)
    args = parser.parse_args()
    
    # Setup config
    cfg = get_cfg()
    get_cfg_defaults(cfg)
    cfg.merge_from_file(args.config_file)
    
    # Depth statistics from our analysis
    HYPERSIM_MEAN_DEPTH = 6.81  # meters
    SUNRGBD_MEAN_DEPTH = 2.75   # meters
    
    # Calculate scale factor to match distributions
    depth_scale = HYPERSIM_MEAN_DEPTH / SUNRGBD_MEAN_DEPTH
    print(f"\nDepth distribution analysis:")
    print(f"  Hypersim mean depth: {HYPERSIM_MEAN_DEPTH:.2f}m")
    print(f"  SUNRGBD mean depth: {SUNRGBD_MEAN_DEPTH:.2f}m")
    print(f"  Scale factor: {depth_scale:.2f}x")
    
    sunrgbd_manifest = os.path.join(args.manifest_dir, "sunrgbd_val_filtered.json")
    
    # Test 1: No scaling (baseline)
    print("\n" + "="*60)
    print("TEST 1: No depth scaling (baseline)")
    print("="*60)
    preds_baseline = run_inference_with_scaled_depth(
        cfg, args.checkpoint, sunrgbd_manifest, 
        depth_scale=1.0, 
        depth_max=8.0,
        max_images=args.max_images
    )
    recall_baseline = compute_recall(preds_baseline, sunrgbd_manifest, "SUNRGBD (no scaling)")
    
    # Test 2: Scale depth to match Hypersim
    print("\n" + "="*60)
    print(f"TEST 2: Scale depth by {depth_scale:.2f}x to match Hypersim")
    print("="*60)
    preds_scaled = run_inference_with_scaled_depth(
        cfg, args.checkpoint, sunrgbd_manifest,
        depth_scale=depth_scale,
        depth_max=8.0,
        max_images=args.max_images
    )
    recall_scaled = compute_recall(preds_scaled, sunrgbd_manifest, f"SUNRGBD (scaled {depth_scale:.2f}x)")
    
    # Test 3: Try different scale factors
    print("\n" + "="*60)
    print("TEST 3: Grid search for optimal scale factor")
    print("="*60)
    
    best_recall = 0
    best_scale = 1.0
    
    for scale in [1.5, 2.0, 2.5, 3.0]:
        preds = run_inference_with_scaled_depth(
            cfg, args.checkpoint, sunrgbd_manifest,
            depth_scale=scale,
            depth_max=8.0,
            max_images=args.max_images
        )
        recall = compute_recall(preds, sunrgbd_manifest, f"SUNRGBD (scale={scale})")
        if recall > best_recall:
            best_recall = recall
            best_scale = scale
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Baseline (no scaling): {recall_baseline*100:.2f}% Recall@15")
    print(f"Scaled ({depth_scale:.2f}x): {recall_scaled*100:.2f}% Recall@15")
    print(f"Best scale ({best_scale}x): {best_recall*100:.2f}% Recall@15")
    
    if recall_scaled > recall_baseline * 1.5:
        print("\n✅ HYPOTHESIS CONFIRMED: Depth scaling significantly improves performance!")
        print("   The model is sensitive to depth distribution mismatch.")
        print("   Retraining with proper depth normalization is recommended.")
    else:
        print("\n❌ Hypothesis not strongly confirmed. Other factors may be at play.")


if __name__ == "__main__":
    main()
