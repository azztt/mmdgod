#!/usr/bin/env python3
"""Test the actual data loading to see what category_ids are passed through."""
import sys
sys.path.insert(0, '/mnt/data/users/anweshan/omni3d')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from detectron2.config import get_cfg
from cubercnn.config import get_cfg_defaults
from cubercnn.data.datasets_rgbd import register_rgbd_datasets, load_rgbd_json, get_filter_settings_from_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog

# Load config
cfg = get_cfg()
get_cfg_defaults(cfg)
cfg.merge_from_file("configs/rgbd/dinov3_base_aug.yaml")
cfg.freeze()

# Register datasets
print("Registering datasets...")
register_rgbd_datasets(cfg)

# Get the dataset
print("\nLoading dataset...")
dataset_name = cfg.DATASETS.TRAIN[0]
dataset_dicts = DatasetCatalog.get(dataset_name)

# Check first few annotations
print(f"\nDataset: {dataset_name}")
print(f"Number of images: {len(dataset_dicts)}")

# Check category_ids in annotations
all_cat_ids = set()
for d in dataset_dicts[:1000]:
    for ann in d.get('annotations', []):
        all_cat_ids.add(ann['category_id'])

print(f"\nCategory IDs in loaded dataset (first 1000 images):")
print(f"  Min: {min(all_cat_ids)}")
print(f"  Max: {max(all_cat_ids)}")
print(f"  Unique: {sorted(all_cat_ids)}")

# Check if -1 is present (ignored annotations)
if -1 in all_cat_ids:
    print("\n  WARNING: -1 present (ignored annotations)")
    all_cat_ids.remove(-1)
    print(f"  Without -1 - Min: {min(all_cat_ids)}, Max: {max(all_cat_ids)}")

# Check metadata
meta = MetadataCatalog.get(dataset_name)
print(f"\nMetadata thing_classes: {len(meta.thing_classes)} classes")
print(f"Metadata id_map keys range: {min(meta.thing_dataset_id_to_contiguous_id.keys())} - {max(meta.thing_dataset_id_to_contiguous_id.keys())}")
print(f"Metadata id_map values range: {min(meta.thing_dataset_id_to_contiguous_id.values())} - {max(meta.thing_dataset_id_to_contiguous_id.values())}")
