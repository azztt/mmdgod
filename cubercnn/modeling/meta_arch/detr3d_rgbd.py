# Copyright (c) Meta Platforms, Inc. and affiliates
# DINOv2 + DETR3D Architecture for 3D Object Detection
"""
DETR3D_RGBD: DINOv2-based 3D Object Detection.

This meta-architecture combines:
1. DINOv2 dual encoder backbone (RGB frozen, depth partially frozen)
2. RPN for proposal generation on fused multi-scale features
3. ROI Heads with DETR3D-style head (cross-attention instead of ROI pooling)

Uses existing CubeRCNN losses and matching - only changes the feature
extraction and aggregation architecture.

Key differences from RCNN3D_RGBD:
- DINOv2 backbone instead of ResNet
- RGB encoder completely frozen
- Depth encoder partially frozen for geometry adaptation
- Can use DETR3DHead for cross-attention feature aggregation
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
import inspect

from detectron2.layers import ShapeSpec
from detectron2.structures import Instances, ImageList
from detectron2.utils.events import get_event_storage
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY, GeneralizedRCNN
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.data import MetadataCatalog

from cubercnn.modeling.roi_heads import build_roi_heads
# Import both encoders - will use config to decide which one
from cubercnn.modeling.backbone.dino_encoder import DinoDualEncoder, build_dino_dual_encoder


@META_ARCH_REGISTRY.register()
class DETR3D_RGBD(GeneralizedRCNN):
    """3D Object Detection with DINO (DINOv2/DINOv3) backbone.
    
    Architecture:
        RGB Image → DINO Encoder (frozen) ─────────┐
                                                    ├─→ Fusion → FPN-like features
        Depth Map → DINO Encoder (partial freeze) ─┘           ↓
                                                            RPN → Proposals
                                                                 ↓
                                                         ROI Heads (with DETR3DHead)
                                                                 ↓
                                                         3D Box Predictions
    
    Uses existing ROI heads and losses from CubeRCNN, only changes backbone
    to DINO (DINOv2/v3) with proper freezing for domain generalization.
    """
    
    def __init__(
        self,
        cfg=None,
        priors=None,
        *,
        backbone: nn.Module = None,
        proposal_generator: nn.Module = None,
        roi_heads: nn.Module = None,
        pixel_mean: Tuple[float] = None,
        pixel_std: Tuple[float] = None,
        input_format: str = "BGR",
        vis_period: int = 0,
        depth_pixel_mean: float = 0.0,
        depth_pixel_std: float = 1.0,
    ):
        """Initialize DETR3D_RGBD.
        
        Args:
            cfg: Detectron2 config
            priors: CubeRCNN priors for dimension/depth estimation
            backbone: DINOv2 dual encoder backbone
            proposal_generator: RPN
            roi_heads: ROI heads with 3D detection
            pixel_mean: RGB normalization mean
            pixel_std: RGB normalization std
            input_format: Image format (BGR or RGB)
            vis_period: Visualization period
            depth_pixel_mean: Depth normalization mean
            depth_pixel_std: Depth normalization std
        """
        # Handle registry-based construction
        if cfg is not None and backbone is None:
            config_dict = DETR3D_RGBD.from_config(cfg, priors=priors)
            backbone = config_dict["backbone"]
            proposal_generator = config_dict["proposal_generator"]
            roi_heads = config_dict["roi_heads"]
            pixel_mean = config_dict["pixel_mean"]
            pixel_std = config_dict["pixel_std"]
            input_format = config_dict["input_format"]
            vis_period = config_dict["vis_period"]
            depth_pixel_mean = config_dict["depth_pixel_mean"]
            depth_pixel_std = config_dict["depth_pixel_std"]
        
        super().__init__(
            backbone=backbone,
            proposal_generator=proposal_generator,
            roi_heads=roi_heads,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            input_format=input_format,
            vis_period=vis_period,
        )
        
        # Depth normalization
        self.register_buffer(
            "depth_pixel_mean",
            torch.tensor([depth_pixel_mean]).view(-1, 1, 1),
            False
        )
        self.register_buffer(
            "depth_pixel_std",
            torch.tensor([depth_pixel_std]).view(-1, 1, 1),
            False
        )
    
    @classmethod
    def from_config(cls, cfg, priors=None):
        """Build from config.
        
        Args:
            cfg: Detectron2 config
            priors: CubeRCNN priors
            
        Returns:
            Dict of constructor arguments
        """
        # Build backbone using registry (supports both DINOv2 and DINOv3)
        backbone_name = cfg.MODEL.BACKBONE.NAME
        backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, None)
        
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape(), priors=priors),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "depth_pixel_mean": getattr(cfg.MODEL, 'DEPTH_PIXEL_MEAN', 0.0),
            "depth_pixel_std": getattr(cfg.MODEL, 'DEPTH_PIXEL_STD', 1.0),
        }
    
    def preprocess_image(
        self,
        batched_inputs: List[Dict]
    ) -> Tuple[ImageList, Optional[torch.Tensor]]:
        """Preprocess RGB and depth inputs.
        
        Args:
            batched_inputs: List of input dicts with 'image' and optional 'depth'
            
        Returns:
            Tuple of (RGB ImageList, depth tensor or None)
        """
        # Process RGB images
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
            images,
            self.backbone.size_divisibility if hasattr(self.backbone, 'size_divisibility') else 0,
        )
        
        # Process depth maps if available
        depths = None
        if "depth" in batched_inputs[0]:
            depths = [x["depth"].to(self.device) for x in batched_inputs]
            depths = [(d - self.depth_pixel_mean) / self.depth_pixel_std for d in depths]
            depths = ImageList.from_tensors(
                depths,
                self.backbone.size_divisibility if hasattr(self.backbone, 'size_divisibility') else 0,
            ).tensor
        
        return images, depths
    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """Forward pass for training or inference.
        
        Args:
            batched_inputs: List of input dicts containing:
                - image: RGB tensor (C, H, W)
                - depth (optional): Depth tensor (1, H, W)
                - K: Camera intrinsics
                - instances (training): Ground truth
                
        Returns:
            Training: Dict of losses
            Inference: List of Instances with predictions
        """
        if not self.training:
            return self.inference(batched_inputs)
        
        # Preprocess
        images, depths = self.preprocess_image(batched_inputs)
        
        # Image scale ratios
        im_scales_ratio = [
            info['height'] / im.shape[1]
            for (info, im) in zip(batched_inputs, images)
        ]
        
        # Camera intrinsics - handle both tensor and list/array input
        Ks = []
        for info in batched_inputs:
            K = info['K']
            if isinstance(K, torch.Tensor):
                Ks.append(K.float().cpu())  # CubeRCNN ROI heads expect CPU tensors
            else:
                Ks.append(torch.FloatTensor(K))
        
        # Ground truth
        gt_instances = None
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        
        # Extract features (DINOv2 backbone with fusion)
        features = self.backbone(images.tensor, depth=depths)
        
        # RPN proposals
        proposals, rpn_losses = self.proposal_generator(images, features, gt_instances)
        
        # ROI heads (uses existing CubeRCNN losses)
        instances, detector_losses = self.roi_heads(
            images, features, proposals,
            Ks, im_scales_ratio,
            gt_instances
        )
        
        # Visualization
        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0 and storage.iter > 0:
                self.visualize_training(batched_inputs, proposals, instances)
        
        # Combine losses
        losses = {}
        losses.update(detector_losses)
        losses.update(rpn_losses)
        
        return losses
    
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """Inference mode forward pass.
        
        Args:
            batched_inputs: Input batch
            detected_instances: Optional pre-detected instances
            do_postprocess: Whether to resize outputs
            
        Returns:
            List of Instances with predictions
        """
        assert not self.training
        
        # Preprocess
        images, depths = self.preprocess_image(batched_inputs)
        
        # Scale ratios
        im_scales_ratio = [
            info['height'] / im.shape[1]
            for (info, im) in zip(batched_inputs, images)
        ]
        
        # Camera intrinsics
        Ks = [torch.FloatTensor(info['K']) for info in batched_inputs]
        
        # Extract features
        features = self.backbone(images.tensor, depth=depths)
        
        # Handle oracle 2D boxes
        if type(batched_inputs == list) and np.any(['oracle2D' in b for b in batched_inputs]):
            oracles = [b['oracle2D'] for b in batched_inputs]
            results, _ = self.roi_heads(
                images, features, oracles, Ks, im_scales_ratio, None
            )
        else:
            # Generate proposals
            proposals, _ = self.proposal_generator(images, features, None)
            # ROI heads inference
            results, _ = self.roi_heads(
                images, features, proposals, Ks, im_scales_ratio, None
            )
        
        if do_postprocess:
            assert not torch.jit.is_scripting()
            return GeneralizedRCNN._postprocess(
                results, batched_inputs, images.image_sizes
            )
        
        return results
    
    def visualize_training(self, batched_inputs, proposals, instances):
        """Visualize training samples."""
        # Reuse parent visualization if available
        pass
