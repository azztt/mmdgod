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
5. WandB integration for experiment tracking

Usage:
    # Single GPU
    python tools/train_rgbd.py --config-file configs/rgbd/progressive/hypersim_to_sunrgbd.yaml
    
    # Multi-GPU
    python tools/train_rgbd.py --config-file configs/rgbd/progressive/hypersim_to_sunrgbd.yaml --num-gpus 4
    
    # With WandB
    python tools/train_rgbd.py --config-file configs/rgbd/Base_RGBD.yaml --wandb-project cube-rgbd
"""
import logging
import os
import sys
import numpy as np
import copy
import glob
import shutil
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
from torch.amp import autocast, GradScaler
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

# WandB integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

logger = logging.getLogger("cubercnn")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.solver import build_optimizer, freeze_bn, PeriodicCheckpointerOnlyOne, ModelEMA
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
from cubercnn.vis.clean_vis import (
    visualize_from_instances,
    get_fixed_visualization_indices,
)


# ============================================================================
# WandB Integration
# ============================================================================
class WandBWriter:
    """Writer that logs metrics to Weights & Biases."""
    
    def __init__(self, log_period: int = 50):
        self._log_period = log_period
        self._last_write = -1
    
    def write(self, storage):
        """Log metrics to WandB."""
        if not WANDB_AVAILABLE or wandb.run is None:
            return
            
        iteration = storage.iter
        if iteration <= self._last_write:
            return
            
        self._last_write = iteration
        
        # Collect all scalars from storage (exclude corner loss)
        metrics = {}
        for k, (v, _) in storage.latest_with_smoothing_hint(window_size=20).items():
            # Skip corner loss metrics
            if 'corner' not in k.lower():
                metrics[f"train/{k}"] = v
        
        # Log to wandb
        if metrics:
            wandb.log(metrics, step=iteration)
    
    def close(self):
        pass


class BestCheckpointer:
    """Tracks and saves the best model checkpoint based on evaluation metric."""
    
    def __init__(self, checkpointer, output_dir, metric_name="AP3D", mode="max", keep_last_n=5):
        """
        Args:
            checkpointer: DetectionCheckpointer instance
            output_dir: Directory to save checkpoints
            metric_name: Metric to track for best model
            mode: "max" or "min" - whether higher or lower is better
            keep_last_n: Number of recent checkpoints to keep
        """
        self.checkpointer = checkpointer
        self.output_dir = output_dir
        self.metric_name = metric_name
        self.mode = mode
        self.keep_last_n = keep_last_n
        
        self.best_metric = float('-inf') if mode == "max" else float('inf')
        self.best_iteration = -1
        
    def is_better(self, metric):
        """Check if metric is better than current best."""
        if self.mode == "max":
            return metric > self.best_metric
        else:
            return metric < self.best_metric
    
    def step(self, iteration, metrics_dict):
        """Check metrics and save best model if improved.
        
        Args:
            iteration: Current iteration
            metrics_dict: Dictionary of evaluation metrics
        """
        # Look for the metric in the dict
        metric_value = None
        for key, value in metrics_dict.items():
            if self.metric_name in key:
                metric_value = value
                break
        
        if metric_value is None:
            logger.warning(f"Metric {self.metric_name} not found in evaluation results")
            return False
        
        if self.is_better(metric_value):
            logger.info(
                f"New best {self.metric_name}: {metric_value:.4f} (previous: {self.best_metric:.4f})"
            )
            self.best_metric = metric_value
            self.best_iteration = iteration
            
            # Save best model
            best_path = os.path.join(self.output_dir, "model_best.pth")
            self.checkpointer.save("model_best")
            
            # Log to wandb
            if WANDB_AVAILABLE and wandb.run is not None:
                wandb.run.summary[f"best_{self.metric_name}"] = metric_value
                wandb.run.summary["best_iteration"] = iteration
            
            return True
        return False
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the last N."""
        checkpoint_files = sorted(
            glob.glob(os.path.join(self.output_dir, "model_*.pth")),
            key=os.path.getmtime
        )
        
        # Exclude model_best.pth and model_final.pth from cleanup
        checkpoint_files = [
            f for f in checkpoint_files 
            if not f.endswith("model_best.pth") and not f.endswith("model_final.pth")
        ]
        
        # Remove old checkpoints
        if len(checkpoint_files) > self.keep_last_n:
            for f in checkpoint_files[:-self.keep_last_n]:
                try:
                    os.remove(f)
                    logger.info(f"Removed old checkpoint: {f}")
                except OSError as e:
                    logger.warning(f"Failed to remove checkpoint {f}: {e}")


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
    
    # Create RGBD mapper for evaluation (same as training, but is_train=False)
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

    for dataset_name in dataset_names_test:
        # Build test loader with RGBD mapper so depth is available
        data_loader = build_detection_test_loader(cfg, dataset_name, mapper=test_mapper)
        results_json = inference_on_dataset(model, data_loader)

        if comm.is_main_process():
            eval_helper.add_predictions(dataset_name, results_json)
            eval_helper.save_predictions(dataset_name)
            eval_helper.evaluate(dataset_name)

            # Clean visualization with GT + Pred side-by-side (fixed samples for consistency)
            instances = torch.load(
                os.path.join(output_folder, dataset_name, 'instances_predictions.pth'),
                weights_only=False  # Required for numpy arrays in predictions
            )
            
            # Get or create fixed indices for consistent visualization across epochs
            fixed_indices_file = os.path.join(cfg.OUTPUT_DIR, f'{dataset_name}_visualization_indices.txt')
            if os.path.exists(fixed_indices_file):
                with open(fixed_indices_file, 'r') as f:
                    fixed_indices = [int(line.strip()) for line in f]
            else:
                # First time: create fixed indices
                fixed_indices = get_fixed_visualization_indices(data_loader.dataset, num_samples=8, seed=42)
                with open(fixed_indices_file, 'w') as f:
                    for idx in fixed_indices:
                        f.write(f"{idx}\n")
                logger.info(f"Created fixed visualization indices for {dataset_name}: {fixed_indices}")
            
            log_str = visualize_from_instances(
                instances, data_loader.dataset, dataset_name,
                os.path.join(output_folder, dataset_name),
                MetadataCatalog.get('omni3d_model').thing_classes,
                iteration,
                num_samples=8,
                fixed_indices=fixed_indices
            )
            logger.info(log_str)

    if comm.is_main_process():
        eval_helper.summarize_all()
    
    # Return metrics for best checkpoint tracking
    return eval_helper


def do_val(cfg, model, iteration, storage=None):
    """
    Run validation on the validation split (20% of training data).
    Faster than full test evaluation, runs every epoch.
    """
    filter_settings = data.get_filter_settings_from_cfg(cfg)    
    filter_settings['visibility_thres'] = cfg.TEST.VISIBILITY_THRES
    filter_settings['truncation_thres'] = cfg.TEST.TRUNCATION_THRES
    filter_settings['min_height_thres'] = 0.0625
    filter_settings['max_depth'] = 1e8

    # Use the validation split dataset (stored in cfg.DATASETS.VAL during setup)
    val_dataset_name = cfg.DATASETS.VAL[0] if len(cfg.DATASETS.VAL) > 0 else None
    if val_dataset_name is None:
        logger.warning("No validation split found, skipping validation")
        return None
    only_2d = cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D == 0.0
    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference_val", 'iter_{}'.format(iteration))

    eval_helper = Omni3DEvaluationHelper(
        [val_dataset_name], 
        filter_settings, 
        output_folder, 
        iter_label=iteration,
        only_2d=only_2d,
    )
    
    # Create RGBD mapper for evaluation
    depth_norm_mode = getattr(cfg.MODEL, 'DEPTH_NORM_MODE', 'fixed')
    depth_norm_percentile = getattr(cfg.MODEL, 'DEPTH_NORM_PERCENTILE', 95)
    
    val_mapper = DatasetMapper3D_RGBD(
        cfg, 
        is_train=False,
        depth_max=cfg.MODEL.DEPTH_MAX,
        depth_norm_mode=depth_norm_mode,
        depth_norm_percentile=depth_norm_percentile,
    )

    # Build validation loader
    data_loader = build_detection_test_loader(cfg, val_dataset_name, mapper=val_mapper)
    results_json = inference_on_dataset(model, data_loader)

    if comm.is_main_process():
        eval_helper.add_predictions(val_dataset_name, results_json)
        eval_helper.save_predictions(val_dataset_name)
        eval_helper.evaluate(val_dataset_name)
        
        # Clean visualization with GT + Pred side-by-side (fixed samples for consistency)
        instances = torch.load(
            os.path.join(output_folder, val_dataset_name, 'instances_predictions.pth'),
            weights_only=False
        )
        
        # Get or create fixed indices for consistent visualization across epochs
        fixed_indices_file = os.path.join(cfg.OUTPUT_DIR, 'val_visualization_indices.txt')
        if os.path.exists(fixed_indices_file):
            with open(fixed_indices_file, 'r') as f:
                fixed_indices = [int(line.strip()) for line in f]
        else:
            # First time: create fixed indices
            fixed_indices = get_fixed_visualization_indices(data_loader.dataset, num_samples=8, seed=42)
            with open(fixed_indices_file, 'w') as f:
                for idx in fixed_indices:
                    f.write(f"{idx}\n")
            logger.info(f"Created fixed visualization indices: {fixed_indices}")
        
        log_str = visualize_from_instances(
            instances, data_loader.dataset, val_dataset_name,
            os.path.join(output_folder, val_dataset_name),
            MetadataCatalog.get('omni3d_model').thing_classes,
            iteration,
            num_samples=8,
            fixed_indices=fixed_indices
        )
        logger.info(log_str)

    if comm.is_main_process():
        eval_helper.summarize_all()
    
    return eval_helper


def do_train(cfg, model, dataset_id_to_unknown_cats, dataset_id_to_src, resume=False, wandb_enabled=False):
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
    
    # Setup writers with WandB support
    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []
    if comm.is_main_process() and wandb_enabled and WANDB_AVAILABLE:
        writers.append(WandBWriter(log_period=20))  # Log every 20 iterations
    
    # Best checkpoint tracker - use AP3D@15 from the first test dataset
    # The metric will be matched against keys like "hypersim_val_rgbd/AP3D@15"
    best_checkpointer = BestCheckpointer(
        checkpointer, cfg.OUTPUT_DIR, 
        metric_name="AP3D@15", mode="max", keep_last_n=5
    ) if comm.is_main_process() else None
    
    # Create the RGB-D dataloader with domain generalization augmentations
    # Augmentation settings are now read from cfg.AUG and cfg.DG
    use_dg_augmentations = getattr(cfg.DG, 'ENABLED', False) or getattr(cfg.AUG, 'ENABLED', False)
    
    # Get depth normalization settings
    depth_norm_mode = getattr(cfg.MODEL, 'DEPTH_NORM_MODE', 'fixed')
    depth_norm_percentile = getattr(cfg.MODEL, 'DEPTH_NORM_PERCENTILE', 95)
    
    if use_dg_augmentations:
        data_mapper = DatasetMapper3D_RGBD_DG(
            cfg, 
            is_train=True,
            depth_max=cfg.MODEL.DEPTH_MAX,
            depth_norm_mode=depth_norm_mode,
            depth_norm_percentile=depth_norm_percentile,
        )
    else:
        data_mapper = DatasetMapper3D_RGBD(
            cfg, 
            is_train=True,
            depth_max=cfg.MODEL.DEPTH_MAX,
            depth_norm_mode=depth_norm_mode,
            depth_norm_percentile=depth_norm_percentile,
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

    # Initialize AMP GradScaler if enabled
    use_amp = cfg.SOLVER.get('AMP_ENABLED', False)
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        logger.info("Using Automatic Mixed Precision (AMP) training")

    # Initialize EMA if enabled
    use_ema = cfg.SOLVER.EMA.ENABLED
    model_ema = None
    if use_ema:
        model_ema = ModelEMA(
            model,
            decay=cfg.SOLVER.EMA.DECAY,
            warmup_iters=cfg.SOLVER.EMA.WARMUP_ITERS,
        )
        logger.info(f"Using EMA with decay={cfg.SOLVER.EMA.DECAY}, "
                   f"warmup_iters={cfg.SOLVER.EMA.WARMUP_ITERS}")

    with EventStorage(start_iter) as storage:
        
        while True:
            data = next(data_iter)
            storage.iter = iteration

            # Forward pass with AMP autocast
            with autocast(device_type='cuda', enabled=use_amp):
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
        
            # Backward pass with AMP scaling
            optimizer.zero_grad()
            scaler.scale(losses).backward()

            # Check gradients for NaN/Inf (only when STABILIZE is enabled)
            grads_unscaled = False
            if not diverging_model and cfg.MODEL.STABILIZE > 0:
                # Unscale gradients for inspection when using AMP
                scaler.unscale_(optimizer)
                grads_unscaled = True
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
                # Update scaler even when skipping (to track inf/nan)
                if grads_unscaled:
                    scaler.update()
            else:
                # Step with scaler for AMP (unscale_ called internally if not already done)
                scaler.step(optimizer)
                scaler.update()
                iterations_success += 1
                
                # Update EMA after successful optimizer step
                if model_ema is not None:
                    model_ema.update(model)

            storage.put_scalar('lr', optimizer.param_groups[0]['lr'], smoothing_hint=False)
            scheduler.step()

            # Log training progress and write to all writers (including WandB)
            if iteration % 20 == 0 and comm.is_main_process():
                success_rate = iterations_success / max(iterations_success + iterations_explode, 1)
                storage.put_scalar('success_rate', success_rate, smoothing_hint=False)
                logger.info(
                    f"Iter {iteration}/{max_iter}, Loss: {losses_reduced:.4f}, "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f}, "
                    f"Success Rate: {success_rate:.2%}"
                )
                # Write to all writers (TensorBoard, WandB, etc.)
                for writer in writers:
                    # WandBWriter takes storage as argument, default writers don't
                    if isinstance(writer, WandBWriter):
                        writer.write(storage)
                    else:
                        writer.write()

            # Checkpointing and cleanup
            periodic_checkpointer.step(iteration)
            if best_checkpointer is not None and (iteration + 1) % cfg.SOLVER.CHECKPOINT_PERIOD == 0:
                best_checkpointer.cleanup_old_checkpoints()

            # Evaluation - multi-frequency: val split every epoch, full eval every N epochs
            if do_eval and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0 and iteration != max_iter:
                # Determine which type of evaluation to run
                full_eval_period = getattr(cfg.TEST, 'FULL_EVAL_PERIOD', cfg.TEST.EVAL_PERIOD * 10)
                run_full_eval = (iteration + 1) % full_eval_period == 0
                
                if run_full_eval:
                    logger.info(f"Running FULL evaluation (all test sets) at iteration {iteration}")
                    # Apply EMA weights for evaluation if enabled
                    if model_ema is not None and cfg.SOLVER.EMA.USE_EMA_FOR_EVAL:
                        logger.info("Applying EMA weights for evaluation")
                        model_ema.apply_shadow(model)
                    
                    eval_helper = do_test(cfg, model, iteration=iteration, storage=storage)
                    
                    # Restore training weights after evaluation
                    if model_ema is not None and cfg.SOLVER.EMA.USE_EMA_FOR_EVAL:
                        model_ema.restore(model)
                else:
                    logger.info(f"Running validation split evaluation at iteration {iteration}")
                    # Apply EMA weights for evaluation if enabled
                    if model_ema is not None and cfg.SOLVER.EMA.USE_EMA_FOR_EVAL:
                        model_ema.apply_shadow(model)
                    
                    eval_helper = do_val(cfg, model, iteration=iteration, storage=storage)
                    
                    # Restore training weights after evaluation
                    if model_ema is not None and cfg.SOLVER.EMA.USE_EMA_FOR_EVAL:
                        model_ema.restore(model)
                
                # Track best checkpoint based on evaluation metrics
                if comm.is_main_process() and best_checkpointer is not None:
                    # Get metrics from eval_helper - extract from results_analysis which has flat structure
                    eval_metrics = {}
                    if hasattr(eval_helper, 'results_analysis') and eval_helper.results_analysis:
                        for ds_name, metrics in eval_helper.results_analysis.items():
                            for metric_name, value in metrics.items():
                                if metric_name == "iters":
                                    continue
                                if isinstance(value, (int, float)):
                                    eval_metrics[f"{ds_name}/{metric_name}"] = value
                    
                    # Also extract from results (bbox_2D/bbox_3D dicts)
                    if hasattr(eval_helper, 'results') and eval_helper.results:
                        for ds_name, result_dict in eval_helper.results.items():
                            if 'bbox_2D' in result_dict and isinstance(result_dict['bbox_2D'], dict):
                                for k, v in result_dict['bbox_2D'].items():
                                    if isinstance(v, (int, float)):
                                        eval_metrics[f"{ds_name}/2D_{k}"] = v
                            if 'bbox_3D' in result_dict and isinstance(result_dict['bbox_3D'], dict):
                                for k, v in result_dict['bbox_3D'].items():
                                    if isinstance(v, (int, float)):
                                        eval_metrics[f"{ds_name}/3D_{k}"] = v
                    
                    # Log eval metrics to WandB (exclude corner loss metrics)
                    if wandb_enabled and WANDB_AVAILABLE and wandb.run is not None:
                        # Filter out corner-related metrics
                        wandb_metrics = {k: v for k, v in eval_metrics.items() 
                                       if 'corner' not in k.lower()}
                        # Add prefix based on evaluation type
                        if run_full_eval:
                            wandb.log({f"eval_full/{k}": v for k, v in wandb_metrics.items()}, step=iteration)
                        else:
                            wandb.log({f"eval_val/{k}": v for k, v in wandb_metrics.items()}, step=iteration)
                    
                    # Update best checkpoint (only on full eval or validation split)
                    best_checkpointer.step(iteration, eval_metrics)
                
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
        
        # Apply EMA weights for final evaluation if enabled
        if model_ema is not None and cfg.SOLVER.EMA.USE_EMA_FOR_EVAL:
            logger.info("Applying EMA weights for final evaluation")
            model_ema.apply_shadow(model)
        
        eval_helper = do_test(cfg, model, iteration='final', storage=storage)
        
        # Save EMA model as final model (don't restore training weights)
        if model_ema is not None and cfg.SOLVER.EMA.USE_EMA_FOR_EVAL:
            # Save EMA weights as the final model
            checkpointer.save("model_final_ema")
            logger.info("Saved EMA weights as model_final_ema")
            
            # Also save EMA state separately for potential resumption
            ema_path = os.path.join(cfg.OUTPUT_DIR, "model_ema_state.pth")
            torch.save(model_ema.state_dict(), ema_path)
            logger.info(f"Saved EMA state to {ema_path}")
        
        # Final best checkpoint check
        if comm.is_main_process() and best_checkpointer is not None:
            eval_metrics = {}
            if hasattr(eval_helper, 'results') and eval_helper.results:
                for ds_name, metrics in eval_helper.results.items():
                    for metric_name, value in metrics.items():
                        eval_metrics[f"{ds_name}/{metric_name}"] = value
            best_checkpointer.step(iteration, eval_metrics)
    
    # Close writers
    if comm.is_main_process():
        for writer in writers:
            writer.close()
        
        # Log final summary to WandB
        if wandb_enabled and WANDB_AVAILABLE and wandb.run is not None:
            wandb.run.summary["final_iteration"] = iteration
            if best_checkpointer is not None:
                wandb.run.summary["best_iteration"] = best_checkpointer.best_iteration
                wandb.run.summary["best_metric"] = best_checkpointer.best_metric


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
    
    # Initialize WandB if requested
    wandb_enabled = False
    if comm.is_main_process() and args.wandb_project and WANDB_AVAILABLE:
        # Silence wandb console output
        os.environ["WANDB_SILENT"] = "true"
        
        wandb_config = {
            "config_file": args.config_file,
            "batch_size": cfg.SOLVER.IMS_PER_BATCH,
            "base_lr": cfg.SOLVER.BASE_LR,
            "max_iter": cfg.SOLVER.MAX_ITER,
            "model": cfg.MODEL.META_ARCHITECTURE,
            "backbone": cfg.MODEL.BACKBONE.NAME,
            "fusion_type": getattr(cfg.MODEL, 'FUSION_TYPE', 'concat'),
            "train_datasets": cfg.DATASETS.TRAIN,
            "test_datasets": cfg.DATASETS.TEST,
        }
        
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or os.path.basename(cfg.OUTPUT_DIR),
            config=wandb_config,
            dir=cfg.OUTPUT_DIR,
            resume="allow" if args.resume else None,
            # settings=wandb.Settings(quiet=True),  # Suppress console output
        )
        wandb_enabled = True
        logger.info(f"WandB initialized: {wandb.run.name}")

    # Get filter settings for dataset loading
    filter_settings = data.get_filter_settings_from_cfg(cfg)

    # ===========================================================================
    # Setup global category metadata FIRST (before registering datasets)
    # This ensures all datasets use consistent category ID mapping
    # ===========================================================================
    logger.info("Setting up global category metadata...")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Set up omni3d_model metadata from config BEFORE loading datasets
    # This ensures load_rgbd_json uses the correct category order
    category_names = list(cfg.DATASETS.CATEGORY_NAMES) if hasattr(cfg.DATASETS, 'CATEGORY_NAMES') else None
    if category_names:
        from detectron2.data import MetadataCatalog
        meta = MetadataCatalog.get('omni3d_model')
        meta.thing_classes = category_names
        # id_map will be built during dataset registration
        logger.info(f"Set global model categories: {category_names[:5]}... ({len(category_names)} total)")
    
    # ===========================================================================
    # Register RGB-D datasets from manifests
    # ===========================================================================
    logger.info("Registering RGB-D datasets...")
    logger.info(f"  Data root: {cfg.DATASETS.DATA_ROOT}")
    logger.info(f"  Manifest root: {cfg.DATASETS.MANIFEST_ROOT}")
    
    # Register all datasets specified in config
    # Now datasets will use the global metadata's category order
    register_rgbd_datasets(cfg)
    
    # Create train/val split from training data if VAL_SPLIT is specified
    val_split_ratio = getattr(cfg.TEST, 'VAL_SPLIT', 0.0)
    if val_split_ratio > 0 and len(cfg.DATASETS.TRAIN) > 0:
        logger.info(f"Creating {val_split_ratio*100:.0f}% validation split from training data")
        train_dataset_name = cfg.DATASETS.TRAIN[0]
        
        # Load full training dataset
        train_data = DatasetCatalog.get(train_dataset_name)
        
        # Create train/val split
        import random
        random.seed(42)  # Reproducible split
        indices = list(range(len(train_data)))
        random.shuffle(indices)
        
        val_size = int(len(train_data) * val_split_ratio)
        val_indices = set(indices[:val_size])
        train_indices = set(indices[val_size:])
        
        train_split = [train_data[i] for i in range(len(train_data)) if i in train_indices]
        val_split = [train_data[i] for i in range(len(train_data)) if i in val_indices]
        
        # Register new datasets
        train_split_name = train_dataset_name + "_train_split"
        val_split_name = train_dataset_name + "_val_split"
        
        DatasetCatalog.register(train_split_name, lambda: train_split)
        DatasetCatalog.register(val_split_name, lambda: val_split)
        
        # Copy metadata from original dataset
        train_meta = MetadataCatalog.get(train_dataset_name)
        for split_name in [train_split_name, val_split_name]:
            split_meta = MetadataCatalog.get(split_name)
            if hasattr(train_meta, 'thing_classes'):
                split_meta.thing_classes = train_meta.thing_classes
            if hasattr(train_meta, 'thing_dataset_id_to_contiguous_id'):
                split_meta.thing_dataset_id_to_contiguous_id = train_meta.thing_dataset_id_to_contiguous_id
        
        logger.info(f"  Training split: {len(train_split)} samples ({train_split_name})")
        logger.info(f"  Validation split: {len(val_split)} samples ({val_split_name})")
        
        # Update config to use the training split and store validation split name
        # NOTE: We modify the config tuple, which requires converting to list first
        from detectron2.config import CfgNode
        cfg.defrost()
        cfg.DATASETS.TRAIN = (train_split_name,)
        cfg.DATASETS.VAL = (val_split_name,)  # Store validation split name for do_val to use
        cfg.freeze()
        logger.info(f"Updated DATASETS.TRAIN to use split: {cfg.DATASETS.TRAIN}")
        logger.info(f"Updated DATASETS.VAL to use split: {cfg.DATASETS.VAL}")
    
    # Store complete model metadata (now with id_map from datasets)
    # IMPORTANT: Pass category_names from config to ensure consistent ordering
    register_and_store_rgbd_model_metadata(
        list(cfg.DATASETS.TRAIN) + list(cfg.DATASETS.TEST),
        cfg.OUTPUT_DIR,
        filter_settings,
        category_names=category_names
    )
    
    # Get category info from metadata
    meta = MetadataCatalog.get('omni3d_model')
    thing_classes = meta.thing_classes
    id_map = meta.thing_dataset_id_to_contiguous_id
    
    logger.info(f"Registered {len(thing_classes)} categories: {thing_classes[:5]}...")
    
    # Create dataset_id mappings
    # For RGBD datasets, we use a simpler mapping since each dataset has unique ID
    # Use predefined dataset IDs to avoid loading dataset twice
    dataset_id_to_unknown_cats = {}  # Can be extended for open-vocabulary detection
    dataset_id_to_src = {}
    
    # Predefined dataset ID mapping (from manifest dataset_id field)
    # This avoids loading the full dataset just to get these IDs
    KNOWN_DATASET_IDS = {
        'hypersim_train_rgbd': 9,
        'hypersim_val_rgbd': 9,
        'sunrgbd_val_rgbd': 13,
        'multiscan_val_rgbd': 14,
        'scannetpp_val_rgbd': 15,
    }
    
    for dataset_name in cfg.DATASETS.TRAIN:
        ds_id = KNOWN_DATASET_IDS.get(dataset_name, hash(dataset_name) % 1000)
        dataset_id_to_src[ds_id] = dataset_name
    
    logger.info(f"Dataset ID mapping: {dataset_id_to_src}")

    # Build model
    # Note: For RGB-D training, we skip dimension priors computation for now.
    # The priors require an Omni3D dataset object which we don't have.
    # Priors can be pre-computed and loaded from a file if needed.
    priors = None
    if hasattr(cfg.MODEL, 'ROI_CUBE_HEAD') and cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_ENABLED:
        logger.warning("Dimension priors are enabled but not computed for RGBD training. "
                       "Set DIMS_PRIORS_ENABLED=False or pre-compute priors.")
        # Disable priors for now
        cfg.defrost()
        cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_ENABLED = False
        cfg.freeze()
    
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

    # Evaluation only or training
    if args.eval_only:
        # Load model weights for evaluation
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=False
        )
        # Run evaluation
        do_test(cfg, model)
        return
    
    # Train
    do_train(cfg, model, dataset_id_to_unknown_cats, dataset_id_to_src, resume=args.resume, wandb_enabled=wandb_enabled)
    
    # Finish WandB
    if comm.is_main_process() and wandb_enabled and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    parser = default_argument_parser()
    # Note: --eval-only is already provided by default_argument_parser()
    
    # WandB arguments
    parser.add_argument("--wandb-project", type=str, default=None,
                        help="WandB project name. If not set, WandB logging is disabled.")
    parser.add_argument("--wandb-name", type=str, default=None,
                        help="WandB run name. Defaults to output directory name.")
    
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
