#!/usr/bin/env python3
"""
Standalone evaluation script for RGB-D model.
Evaluates a trained checkpoint on all validation datasets using RGBD mapper.
"""

import os
import sys
import argparse
import torch
import logging

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import MetadataCatalog

import cubercnn.data as data
from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.meta_arch import build_model
from cubercnn.evaluation.omni3d_evaluation import Omni3DEvaluationHelper
from cubercnn.data.dataset_mapper_rgbd import DatasetMapper3D_RGBD
from cubercnn.data.build import build_detection_test_loader
from cubercnn.evaluation import inference_on_dataset
from cubercnn.data.datasets_rgbd import (
    register_rgbd_datasets,
    register_and_store_rgbd_model_metadata,
)


logger = logging.getLogger("eval_rgbd")


def setup_cfg(args):
    """Setup config from args."""
    cfg = get_cfg()
    get_cfg_defaults(cfg)
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    # Override depth_max if specified
    if args.depth_max is not None:
        cfg.MODEL.DEPTH_MAX = args.depth_max
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def do_eval(cfg, model, checkpoint_name="final"):
    """Run evaluation on all test datasets."""
    filter_settings = data.get_filter_settings_from_cfg(cfg)
    filter_settings['visibility_thres'] = cfg.TEST.VISIBILITY_THRES
    filter_settings['truncation_thres'] = cfg.TEST.TRUNCATION_THRES
    filter_settings['min_height_thres'] = 0.0625
    filter_settings['max_depth'] = 1e8

    dataset_names_test = cfg.DATASETS.TEST
    only_2d = cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D == 0.0
    output_folder = os.path.join(cfg.OUTPUT_DIR, "eval_rgbd_mapper", checkpoint_name)
    os.makedirs(output_folder, exist_ok=True)

    eval_helper = Omni3DEvaluationHelper(
        dataset_names_test,
        filter_settings,
        output_folder,
        iter_label=checkpoint_name,
        only_2d=only_2d,
    )

    # Create RGBD mapper for evaluation
    # Use same depth normalization mode as training for consistency
    depth_norm_mode = getattr(cfg.MODEL, 'DEPTH_NORM_MODE', 'fixed')
    depth_norm_percentile = getattr(cfg.MODEL, 'DEPTH_NORM_PERCENTILE', 95)
    
    test_mapper = DatasetMapper3D_RGBD(
        cfg,
        is_train=False,
        depth_max=cfg.MODEL.DEPTH_MAX,
        depth_norm_mode=depth_norm_mode,
        depth_norm_percentile=depth_norm_percentile,
    )
    
    logger.info(f"Using DEPTH_MAX = {cfg.MODEL.DEPTH_MAX}")
    logger.info(f"Using DEPTH_NORM_MODE = {depth_norm_mode}")
    logger.info(f"Using mapper: {type(test_mapper).__name__}")

    for dataset_name in dataset_names_test:
        logger.info(f"Evaluating on {dataset_name}")
        
        # Build test loader with RGBD mapper
        # Pass mapper explicitly to override the default DatasetMapper
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=test_mapper)
        results_json = inference_on_dataset(model, data_loader)

        eval_helper.add_predictions(dataset_name, results_json)
        eval_helper.save_predictions(dataset_name)
        eval_helper.evaluate(dataset_name)

    eval_helper.summarize_all()
    return eval_helper


def main():
    parser = argparse.ArgumentParser(description="Evaluate RGB-D model with RGBD mapper")
    parser.add_argument("--config-file", required=True, help="Path to config file")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--depth-max", type=float, default=None, 
                        help="Override depth_max (default: use config value)")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("opts", nargs=argparse.REMAINDER, help="Additional config options")
    args = parser.parse_args()

    # Setup config
    cfg = setup_cfg(args)
    
    # Override output dir if specified
    if args.output_dir:
        cfg.defrost()
        cfg.OUTPUT_DIR = args.output_dir
        cfg.freeze()
    
    # Get filter settings
    filter_settings = data.get_filter_settings_from_cfg(cfg)
    
    # Get category names from config
    category_names = list(cfg.DATASETS.CATEGORY_NAMES) if cfg.DATASETS.CATEGORY_NAMES else None
    
    # Register RGBD datasets
    register_rgbd_datasets(cfg)
    register_and_store_rgbd_model_metadata(
        list(cfg.DATASETS.TRAIN) + list(cfg.DATASETS.TEST),
        cfg.OUTPUT_DIR,
        filter_settings,
        category_names=category_names
    )
    
    # Build model
    model = build_model(cfg)
    model.eval()
    
    # Load checkpoint
    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.load(args.checkpoint)
    
    checkpoint_name = os.path.splitext(os.path.basename(args.checkpoint))[0]
    logger.info(f"Loaded checkpoint: {args.checkpoint}")
    
    # Run evaluation
    with torch.no_grad():
        do_eval(cfg, model, checkpoint_name)


if __name__ == "__main__":
    main()
