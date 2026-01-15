#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates
# Modified for RGB-D domain-generalized 3D detection
"""
Training script for RGB-D 3D Object Detection with Domain Generalization.

This script extends the base Cube R-CNN training to:
1. Load RGB + Depth data via DatasetMapper3D_RGBD
2. Use dual encoder backbone for domain-generalized features
3. Support various fusion strategies (concat, gated, DAAF)
4. Apply domain generalization augmentations (FSDR, etc.)

Usage:
    # Single GPU
    python tools/train_rgbd.py --config-file configs/rgbd/progressive/hypersim_to_sunrgbd.yaml
    
    # Multi-GPU
    python tools/train_rgbd.py --config-file configs/rgbd/progressive/hypersim_to_sunrgbd.yaml --num-gpus 4
"""
import logging
import os
import sys
import numpy as np
import copy
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import (
    default_argument_parser, 
    default_setup, 
    default_writers, 
    launch
)
from detectron2.solver import build_lr_scheduler
from detectron2.utils.events import EventStorage
from detectron2.utils.logger import setup_logger

logger = logging.getLogger("cubercnn")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.solver import build_optimizer, freeze_bn, PeriodicCheckpointerOnlyOne
from cubercnn.config import get_cfg_defaults
from cubercnn.data import (
    load_omni3d_json,
    build_detection_train_loader,
    build_detection_test_loader,
    get_omni3d_categories,
    simple_register
)
from cubercnn.data.datasets_rgbd import (
    register_rgbd_datasets,
    register_all_rgbd_datasets,
    register_and_store_rgbd_model_metadata,
    get_manifest_file_for_dataset,
)
from cubercnn.data.dataset_mapper_rgbd import DatasetMapper3D_RGBD, DatasetMapper3D_RGBD_DG
from cubercnn.evaluation import (
    Omni3DEvaluator, Omni3Deval,
    Omni3DEvaluationHelper,
    inference_on_dataset
)
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D_RGBD, build_model_rgbd
from cubercnn.modeling.backbone import build_simple_dual_encoder_backbone
from cubercnn import util, vis, data
import cubercnn.vis.logperf as utils_logperf


MAX_TRAINING_ATTEMPTS = 10


def allreduce_dict(input_dict, average=True):
    """All reduce a dict of tensors."""
    world_size = comm.get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def do_test(cfg, model, iteration='final', storage=None):
    """Run evaluation on test datasets."""
    filter_settings = data.get_filter_settings_from_cfg(cfg)    
    filter_settings['visibility_thres'] = cfg.TEST.VISIBILITY_THRES
    filter_settings['truncation_thres'] = cfg.TEST.TRUNCATION_THRES
    filter_settings['min_height_thres'] = 0.0625
    filter_settings['max_depth'] = 1e8

    dataset_names_test = cfg.DATASETS.TEST
    only_2d = cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D == 0.0
    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", 'iter_{}'.format(iteration))

    eval_helper = Omni3DEvaluationHelper(
        dataset_names_test, 
        filter_settings, 
        output_folder, 
        iter_label=iteration,
        only_2d=only_2d,
    )

    for dataset_name in dataset_names_test:
        # Build test loader with RGBD mapper
        data_loader = build_detection_test_loader(cfg, dataset_name)
        results_json = inference_on_dataset(model, data_loader)

        if comm.is_main_process():
            eval_helper.add_predictions(dataset_name, results_json)
            eval_helper.save_predictions(dataset_name)
            eval_helper.evaluate(dataset_name)

            # Visualize some instances
            instances = torch.load(os.path.join(output_folder, dataset_name, 'instances_predictions.pth'))
            log_str = vis.visualize_from_instances(
                instances, data_loader.dataset, dataset_name, 
                cfg.INPUT.MIN_SIZE_TEST, os.path.join(output_folder, dataset_name), 
                MetadataCatalog.get('omni3d_model').thing_classes, iteration
            )
            logger.info(log_str)

    if comm.is_main_process():
        eval_helper.summarize_all()


def do_train(cfg, model, dataset_id_to_unknown_cats, dataset_id_to_src, resume=False):
    """Main training loop for RGB-D model."""
    max_iter = cfg.SOLVER.MAX_ITER
    do_eval = cfg.TEST.EVAL_PERIOD > 0

    model.train()

    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    # Bookkeeping
    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )    
    periodic_checkpointer = PeriodicCheckpointerOnlyOne(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    
    # Create the RGB-D dataloader with domain generalization augmentations
    use_dg_augmentations = getattr(cfg.MODEL, 'USE_DG_AUGMENTATIONS', True)
    
    if use_dg_augmentations:
        data_mapper = DatasetMapper3D_RGBD_DG(
            cfg, 
            is_train=True,
            depth_max=getattr(cfg.MODEL, 'DEPTH_MAX', 8.0),
            use_fsdr=getattr(cfg.MODEL, 'USE_FSDR', True),
            fsdr_prob=getattr(cfg.MODEL, 'FSDR_PROB', 0.5),
            use_depth_dropout=getattr(cfg.MODEL, 'USE_DEPTH_DROPOUT', True),
            depth_dropout_prob=getattr(cfg.MODEL, 'DEPTH_DROPOUT_PROB', 0.3),
            use_photometric=getattr(cfg.MODEL, 'USE_PHOTOMETRIC', True),
            photometric_prob=getattr(cfg.MODEL, 'PHOTOMETRIC_PROB', 0.8),
        )
    else:
        data_mapper = DatasetMapper3D_RGBD(
            cfg, 
            is_train=True,
            depth_max=getattr(cfg.MODEL, 'DEPTH_MAX', 8.0),
        )
    
    data_loader = build_detection_train_loader(
        cfg, mapper=data_mapper, dataset_id_to_src=dataset_id_to_src
    )

    # Give the mapper access to dataset_ids
    data_mapper.dataset_id_to_unknown_cats = dataset_id_to_unknown_cats

    # Load pretrained weights if specified
    if cfg.MODEL.WEIGHTS_PRETRAIN != '':
        checkpointer.load(cfg.MODEL.WEIGHTS_PRETRAIN, checkpointables=[])

    # Determine starting iteration
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
        .get("iteration", -1) + 1
    )
    iteration = start_iter

    logger.info("Starting RGB-D training from iteration {}".format(start_iter))

    if not cfg.MODEL.USE_BN:
        freeze_bn(model)

    world_size = comm.get_world_size()

    # Stabilization parameters
    iterations_success = 0
    iterations_explode = 0
    TOLERANCE = 4.0
    GAMMA = 0.02
    recent_loss = None

    data_iter = iter(data_loader)
    named_params = list(model.named_parameters())

    with EventStorage(start_iter) as storage:
        
        while True:
            data = next(data_iter)
            storage.iter = iteration

            # Forward pass
            loss_dict = model(data)
            losses = sum(loss_dict.values())

            # Reduce losses across GPUs
            loss_dict_reduced = {k: v.item() for k, v in allreduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        
            comm.synchronize()

            if recent_loss is None:
                recent_loss = losses_reduced * 2.0

            # Check for diverging model
            diverging_model = cfg.MODEL.STABILIZE > 0 and (
                losses_reduced > recent_loss * TOLERANCE or 
                not np.isfinite(losses_reduced) or 
                np.isnan(losses_reduced)
            )

            if diverging_model:
                losses = losses.clip(0, 1) 
                logger.warning(
                    'Skipping gradient update: loss {:.2f} vs. rolling mean {:.2f}, Dict-> {}'.format(
                        losses_reduced, recent_loss, loss_dict_reduced
                    )
                )
            else:
                recent_loss = recent_loss * (1 - GAMMA) + losses_reduced * GAMMA
            
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)
        
            # Backward pass
            optimizer.zero_grad()
            losses.backward()

            # Check gradients for NaN/Inf
            if not diverging_model and cfg.MODEL.STABILIZE > 0:
                for name, param in named_params:
                    if param.grad is not None:
                        diverging_model = torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                    if diverging_model:
                        logger.warning('Skipping gradient update due to inf/nan detection')
                        break

            # Sync diverging status across GPUs
            diverging_model = torch.tensor(float(diverging_model)).cuda()
            if world_size > 1:
                dist.all_reduce(diverging_model)
            comm.synchronize()

            if diverging_model > 0:
                optimizer.zero_grad()
                iterations_explode += 1
            else:
                optimizer.step()
                iterations_success += 1

            storage.put_scalar('lr', optimizer.param_groups[0]['lr'], smoothing_hint=False)
            scheduler.step()

            # Log training progress
            if iteration % 50 == 0 and comm.is_main_process():
                success_rate = iterations_success / max(iterations_success + iterations_explode, 1)
                logger.info(
                    f"Iter {iteration}/{max_iter}, Loss: {losses_reduced:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
                    f"Success Rate: {success_rate:.2%}"
                )

            # Checkpointing
            periodic_checkpointer.step(iteration)

            # Evaluation
            if do_eval and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0 and iteration != max_iter:
                logger.info("Running evaluation at iteration {}".format(iteration))
                do_test(cfg, model, iteration=iteration, storage=storage)
                model.train()
                if not cfg.MODEL.USE_BN:
                    freeze_bn(model)

            # Check for convergence failure
            if cfg.MODEL.STABILIZE > 0:
                failure_rate = iterations_explode / max(iterations_success + iterations_explode, 1)
                if failure_rate > 0.5 and iteration > 1000:
                    logger.error(
                        f"Training failing with {failure_rate:.1%} divergence rate. Stopping."
                    )
                    break

            iteration += 1
            if iteration >= max_iter:
                break

    # Final evaluation
    if do_eval:
        logger.info("Running final evaluation")
        do_test(cfg, model, iteration='final', storage=storage)


def setup(args):
    """Setup config and logging."""
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    config_file = args.config_file
    
    # Handle overrides from command line
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    
    default_setup(cfg, args)
    
    setup_logger(
        output=cfg.OUTPUT_DIR, 
        distributed_rank=comm.get_rank(), 
        name="cubercnn"
    )
    
    return cfg


def main(args):
    """Main entry point."""
    cfg = setup(args)

    # Get filter settings for dataset loading
    filter_settings = data.get_filter_settings_from_cfg(cfg)

    # ===========================================================================
    # Register RGB-D datasets from manifests
    # ===========================================================================
    logger.info("Registering RGB-D datasets...")
    logger.info(f"  Data root: {cfg.DATASETS.DATA_ROOT}")
    logger.info(f"  Manifest root: {cfg.DATASETS.MANIFEST_ROOT}")
    
    # Register all datasets specified in config
    register_rgbd_datasets(cfg)
    
    # Store model metadata (category info)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    register_and_store_rgbd_model_metadata(
        list(cfg.DATASETS.TRAIN) + list(cfg.DATASETS.TEST),
        cfg.OUTPUT_DIR,
        filter_settings
    )
    
    # Get category info from metadata
    meta = MetadataCatalog.get('omni3d_model')
    thing_classes = meta.thing_classes
    id_map = meta.thing_dataset_id_to_contiguous_id
    
    logger.info(f"Registered {len(thing_classes)} categories: {thing_classes[:5]}...")
    
    # Create dataset_id mappings
    # For RGBD datasets, we use a simpler mapping since each dataset has unique ID
    dataset_id_to_unknown_cats = {}  # Can be extended for open-vocabulary detection
    dataset_id_to_src = {}
    
    # Populate from training datasets
    for dataset_name in cfg.DATASETS.TRAIN:
        try:
            dataset = DatasetCatalog.get(dataset_name)
            for sample in dataset:
                ds_id = sample.get('dataset_id', 0)
                if ds_id not in dataset_id_to_src:
                    dataset_id_to_src[ds_id] = dataset_name
        except Exception as e:
            logger.warning(f"Could not load dataset {dataset_name}: {e}")
    
    logger.info(f"Dataset ID mapping: {dataset_id_to_src}")

    # Build model
    priors = None
    if hasattr(cfg.MODEL, 'ROI_CUBE_HEAD') and cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_ENABLED:
        # Compute dimension priors from training data
        priors = util.compute_priors(cfg, thing_classes)
    
    model = build_model_rgbd(cfg, priors=priors)
    logger.info("Model:\n{}".format(model))
    
    # Log parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Total parameters: {total_params:,}, "
        f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.1f}%)"
    )

    # Distributed training
    if args.num_gpus > 1:
        model = DistributedDataParallel(
            model, 
            device_ids=[comm.get_local_rank()], 
            broadcast_buffers=False,
            find_unused_parameters=True
        )

    # Train
    do_train(cfg, model, dataset_id_to_unknown_cats, dataset_id_to_src, resume=args.resume)


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    args = parser.parse_args()
    
    print("Command Line Args:", args)
    
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
