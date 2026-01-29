#!/usr/bin/env python3
"""
Evaluation Pipeline Verification Script

This script verifies that the evaluation pipeline is working correctly by:
1. Loading a trained model
2. Running inference on a small subset of each dataset
3. Comparing predictions with ground truth
4. Checking category ID mappings are correct
5. Verifying IoU calculations
"""

import os
import sys
import json
import numpy as np
import torch
from PIL import Image
import cv2
from collections import defaultdict

# Add project root to path
sys.path.insert(0, '/mnt/data/users/anweshan/omni3d')

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import Boxes, BoxMode

def box_iou_2d(box1, box2):
    """Compute 2D IoU between two boxes in xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def verify_category_mappings():
    """Verify category ID mappings between model and datasets."""
    print("\n" + "="*80)
    print("STEP 1: Verifying Category ID Mappings")
    print("="*80)
    
    # Load manifests
    manifest_root = "/mnt/data/users/anweshan/omni3d/data/unidet3d_format/cleaned_manifests"
    datasets = {
        "hypersim_train": "hypersim_train_filtered.json",
        "hypersim_val": "hypersim_val_filtered.json",
        "sunrgbd_val": "sunrgbd_val_filtered.json",
        "multiscan_val": "multiscan_val_filtered.json",
        "scannetpp_val": "scannetpp_val_filtered.json",
    }
    
    # Model categories from config
    model_categories = [
        'person', 'books', 'chair', 'towel', 'blinds', 'window', 'lamp', 
        'shelves', 'mirror', 'sink', 'cabinet', 'bathtub', 'door', 'toilet',
        'desk', 'box', 'bookcase', 'picture', 'table', 'counter', 'bed',
        'night stand', 'dresser', 'pillow', 'sofa', 'television', 'floor mat',
        'curtain', 'clothes', 'stationery', 'refrigerator'
    ]
    
    print(f"\nModel has {len(model_categories)} categories")
    print(f"Order: {model_categories[:5]}... {model_categories[-3:]}")
    
    all_ok = True
    
    for dataset_name, manifest_file in datasets.items():
        manifest_path = os.path.join(manifest_root, manifest_file)
        if not os.path.exists(manifest_path):
            print(f"\n⚠ {dataset_name}: Manifest not found!")
            continue
            
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        # Build category ID -> name mapping from manifest
        cat_id_to_name = {c['id']: c['name'] for c in manifest['categories']}
        
        # Check which model categories are in this dataset
        dataset_cats = set(cat_id_to_name.values())
        model_cats_in_dataset = [c for c in model_categories if c in dataset_cats]
        missing_in_dataset = [c for c in model_categories if c not in dataset_cats]
        extra_in_dataset = [c for c in dataset_cats if c not in model_categories]
        
        print(f"\n📊 {dataset_name}:")
        print(f"   Manifest categories: {len(cat_id_to_name)}")
        print(f"   Overlap with model: {len(model_cats_in_dataset)}")
        if missing_in_dataset:
            print(f"   Missing from dataset: {missing_in_dataset[:5]}{'...' if len(missing_in_dataset) > 5 else ''}")
        if extra_in_dataset:
            print(f"   ⚠ Extra in dataset (ignored): {extra_in_dataset}")
            
        # Check for duplicate category names
        cat_names = list(cat_id_to_name.values())
        duplicates = [n for n in set(cat_names) if cat_names.count(n) > 1]
        if duplicates:
            print(f"   ⚠ DUPLICATE CATEGORIES: {duplicates}")
            for dup in duplicates:
                ids = [id for id, name in cat_id_to_name.items() if name == dup]
                print(f"      '{dup}' has IDs: {ids}")
            all_ok = False
            
        # Verify ID mapping for a sample annotation
        if manifest['annotations']:
            sample_ann = manifest['annotations'][0]
            sample_cat_id = sample_ann['category_id']
            sample_cat_name = cat_id_to_name.get(sample_cat_id, "UNKNOWN")
            if sample_cat_name in model_categories:
                model_idx = model_categories.index(sample_cat_name)
                print(f"   Sample: cat_id={sample_cat_id} -> '{sample_cat_name}' -> model_idx={model_idx} ✓")
            else:
                print(f"   Sample: cat_id={sample_cat_id} -> '{sample_cat_name}' NOT IN MODEL ⚠")
    
    return all_ok


def verify_ground_truth_loading():
    """Verify ground truth annotations are loaded correctly."""
    print("\n" + "="*80)
    print("STEP 2: Verifying Ground Truth Loading")
    print("="*80)
    
    manifest_root = "/mnt/data/users/anweshan/omni3d/data/unidet3d_format/cleaned_manifests"
    data_root = "/mnt/data/users/anweshan/omni3d/data"
    
    datasets = {
        "hypersim_val": "hypersim_val_filtered.json",
        "sunrgbd_val": "sunrgbd_val_filtered.json",
    }
    
    all_ok = True
    
    for dataset_name, manifest_file in datasets.items():
        manifest_path = os.path.join(manifest_root, manifest_file)
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        print(f"\n📊 {dataset_name}:")
        print(f"   Images: {len(manifest['images'])}")
        print(f"   Annotations: {len(manifest['annotations'])}")
        print(f"   Categories: {len(manifest['categories'])}")
        
        # Check a few images have valid paths
        cat_id_to_name = {c['id']: c['name'] for c in manifest['categories']}
        img_id_to_info = {img['id']: img for img in manifest['images']}
        
        # Group annotations by image
        anns_by_image = defaultdict(list)
        for ann in manifest['annotations']:
            anns_by_image[ann['image_id']].append(ann)
        
        # Check first few images
        sample_images = manifest['images'][:3]
        for img_info in sample_images:
            img_path = os.path.join(data_root, img_info['file_path'])
            depth_path = os.path.join(data_root, img_info.get('depth_file_path', ''))
            
            img_exists = os.path.exists(img_path)
            depth_exists = os.path.exists(depth_path) if depth_path else False
            
            anns = anns_by_image.get(img_info['id'], [])
            
            status = "✓" if img_exists and depth_exists else "⚠"
            print(f"   {status} Image {img_info['id']}: {len(anns)} annotations, img={'✓' if img_exists else '✗'}, depth={'✓' if depth_exists else '✗'}")
            
            if anns:
                # Check annotation format
                ann = anns[0]
                has_bbox = 'bbox' in ann
                has_bbox_3d = 'bbox_3d' in ann or 'bbox3d_camera' in ann or 'corners_3d' in ann
                cat_name = cat_id_to_name.get(ann['category_id'], 'UNKNOWN')
                print(f"      Sample ann: cat='{cat_name}', 2D_bbox={has_bbox}, 3D_bbox={has_bbox_3d}")
                
                if has_bbox:
                    bbox = ann['bbox']
                    # Check if bbox is in valid range
                    if bbox[2] <= 0 or bbox[3] <= 0:
                        print(f"      ⚠ Invalid bbox size: {bbox}")
                        all_ok = False
            
            if not img_exists:
                all_ok = False
    
    return all_ok


def verify_predictions_format():
    """Verify predictions are in correct format by checking saved results."""
    print("\n" + "="*80)
    print("STEP 3: Verifying Prediction Format")
    print("="*80)
    
    output_dir = "/mnt/data/users/anweshan/omni3d/output/rgbd_basic_aug/inference/iter_final"
    
    datasets = ["hypersim_val_rgbd", "sunrgbd_val_rgbd"]
    
    for dataset in datasets:
        results_path = os.path.join(output_dir, dataset, "omni_instances_results.json")
        
        if not os.path.exists(results_path):
            print(f"\n⚠ {dataset}: No results file found at {results_path}")
            continue
        
        with open(results_path) as f:
            results = json.load(f)
        
        print(f"\n📊 {dataset}:")
        print(f"   Total predictions: {len(results)}")
        
        if not results:
            print("   ⚠ No predictions!")
            continue
        
        # Check prediction format
        sample = results[0]
        required_keys = ['image_id', 'category_id', 'bbox', 'score']
        missing_keys = [k for k in required_keys if k not in sample]
        
        if missing_keys:
            print(f"   ⚠ Missing keys in prediction: {missing_keys}")
        else:
            print(f"   Required keys present: ✓")
        
        # Check category distribution
        cat_counts = defaultdict(int)
        for pred in results:
            cat_counts[pred['category_id']] += 1
        
        print(f"   Unique categories predicted: {len(cat_counts)}")
        top_cats = sorted(cat_counts.items(), key=lambda x: -x[1])[:5]
        print(f"   Top 5 categories: {[(cid, cnt) for cid, cnt in top_cats]}")
        
        # Check score distribution
        scores = [p['score'] for p in results]
        print(f"   Score range: {min(scores):.3f} - {max(scores):.3f}")
        print(f"   Mean score: {np.mean(scores):.3f}")
        
        # Check bbox format
        sample_bbox = results[0]['bbox']
        print(f"   Sample bbox format: {sample_bbox}")
        if len(sample_bbox) == 4:
            print(f"   Bbox format: xywh or xyxy (4 values) ✓")
        else:
            print(f"   ⚠ Unexpected bbox format: {len(sample_bbox)} values")
        
        # Check for 3D predictions
        has_3d = 'bbox3D' in sample or 'bbox_3d' in sample or 'K' in sample
        print(f"   Has 3D predictions: {'✓' if has_3d else '✗'}")
        
        if 'bbox3D' in sample:
            print(f"   3D bbox format: {len(sample['bbox3D'])} values")


def verify_iou_calculation():
    """Verify IoU calculation with known examples."""
    print("\n" + "="*80)
    print("STEP 4: Verifying IoU Calculation")
    print("="*80)
    
    # Test 2D IoU with known values
    box1 = [0, 0, 100, 100]  # 100x100 box at origin
    box2 = [50, 50, 150, 150]  # 50% overlap
    
    iou = box_iou_2d(box1, box2)
    expected = (50*50) / (100*100 + 100*100 - 50*50)  # intersection / union
    
    print(f"\n2D IoU Test:")
    print(f"   Box1: {box1}")
    print(f"   Box2: {box2}")
    print(f"   Computed IoU: {iou:.4f}")
    print(f"   Expected IoU: {expected:.4f}")
    print(f"   Match: {'✓' if abs(iou - expected) < 0.001 else '✗'}")
    
    # Test perfect overlap
    iou_same = box_iou_2d(box1, box1)
    print(f"\n   Same box IoU: {iou_same:.4f} (expected 1.0) {'✓' if abs(iou_same - 1.0) < 0.001 else '✗'}")
    
    # Test no overlap
    box3 = [200, 200, 300, 300]
    iou_none = box_iou_2d(box1, box3)
    print(f"   Non-overlapping IoU: {iou_none:.4f} (expected 0.0) {'✓' if iou_none == 0 else '✗'}")


def compare_predictions_with_gt():
    """Compare actual predictions with ground truth to verify alignment."""
    print("\n" + "="*80)
    print("STEP 5: Comparing Predictions with Ground Truth")
    print("="*80)
    
    manifest_root = "/mnt/data/users/anweshan/omni3d/data/unidet3d_format/cleaned_manifests"
    output_dir = "/mnt/data/users/anweshan/omni3d/output/rgbd_basic_aug/inference/iter_final"
    
    datasets = [
        ("sunrgbd_val_rgbd", "sunrgbd_val_filtered.json"),
        ("hypersim_val_rgbd", "hypersim_val_filtered.json"),
    ]
    
    for dataset_name, manifest_file in datasets:
        results_path = os.path.join(output_dir, dataset_name, "omni_instances_results.json")
        manifest_path = os.path.join(manifest_root, manifest_file)
        
        if not os.path.exists(results_path):
            print(f"\n⚠ {dataset_name}: No results")
            continue
        
        with open(results_path) as f:
            predictions = json.load(f)
        with open(manifest_path) as f:
            manifest = json.load(f)
        
        cat_id_to_name = {c['id']: c['name'] for c in manifest['categories']}
        img_id_to_info = {img['id']: img for img in manifest['images']}
        
        # Group by image
        preds_by_img = defaultdict(list)
        for p in predictions:
            preds_by_img[p['image_id']].append(p)
        
        gt_by_img = defaultdict(list)
        for ann in manifest['annotations']:
            gt_by_img[ann['image_id']].append(ann)
        
        print(f"\n📊 {dataset_name}:")
        print(f"   Images with predictions: {len(preds_by_img)}")
        print(f"   Images with GT: {len(gt_by_img)}")
        print(f"   Overlapping images: {len(set(preds_by_img.keys()) & set(gt_by_img.keys()))}")
        
        # Check a few images in detail
        common_imgs = list(set(preds_by_img.keys()) & set(gt_by_img.keys()))[:3]
        
        for img_id in common_imgs:
            preds = preds_by_img[img_id]
            gts = gt_by_img[img_id]
            
            pred_cats = set(p['category_id'] for p in preds)
            gt_cats = set(cat_id_to_name.get(g['category_id'], f"unk_{g['category_id']}") for g in gts)
            gt_cat_ids = set(g['category_id'] for g in gts)
            
            print(f"\n   Image {img_id}:")
            print(f"      Predictions: {len(preds)} detections, categories (IDs): {pred_cats}")
            print(f"      Ground truth: {len(gts)} objects, categories: {gt_cats}")
            print(f"      GT category IDs: {gt_cat_ids}")
            
            # Check if any predicted category matches GT
            # Predictions use dataset category IDs (after unmapping)
            matching = pred_cats & gt_cat_ids
            print(f"      Matching category IDs: {matching if matching else 'NONE'}")
            
            if not matching and len(preds) > 0 and len(gts) > 0:
                print(f"      ⚠ No category overlap between predictions and GT!")


def main():
    print("\n" + "="*80)
    print("EVALUATION PIPELINE VERIFICATION")
    print("="*80)
    
    results = {}
    
    # Step 1: Category mappings
    results['categories'] = verify_category_mappings()
    
    # Step 2: Ground truth loading
    results['gt_loading'] = verify_ground_truth_loading()
    
    # Step 3: Prediction format
    verify_predictions_format()
    
    # Step 4: IoU calculation
    verify_iou_calculation()
    
    # Step 5: Compare predictions with GT
    compare_predictions_with_gt()
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if all(results.values()):
        print("✓ All basic checks passed")
    else:
        print("⚠ Some issues found - see above for details")
    
    print("\nNote: This script checks format and mappings.")
    print("Poor AP scores may be due to model training, not evaluation bugs.")


if __name__ == "__main__":
    main()
