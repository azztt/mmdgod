#!/usr/bin/env python3
"""Debug script to check model predictions."""

import torch
import sys
sys.path.insert(0, '/mnt/data/users/anweshan/omni3d')

from detectron2.config import get_cfg
from cubercnn.config import get_cfg_defaults
from cubercnn.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import json

# Load config
cfg = get_cfg()
get_cfg_defaults(cfg)
cfg.merge_from_file("configs/rgbd/dinov3_base_aug.yaml")
cfg.MODEL.WEIGHTS = "output/dinov3_base_aug/model_recent.pth"
cfg.freeze()

# Build model
model = build_model(cfg)
model.eval()

# Load weights
checkpointer = DetectionCheckpointer(model)
checkpointer.load(cfg.MODEL.WEIGHTS)

print("Model loaded successfully")
print(f"Number of classes: {model.num_classes}")
print(f"Test score threshold: {model.test_score_thresh}")
print(f"Test NMS threshold: {model.test_nms_thresh}")

# Load a sample
with open("datasets/Hypersim/val_rgbd.json") as f:
    data = json.load(f)

print(f"\nDataset: {len(data['images'])} images, {len(data['annotations'])} annotations")

# Check if model has the required attributes
print(f"\nModel class: {model.__class__.__name__}")
if hasattr(model, 'device'):
    print(f"Model device: {model.device}")
