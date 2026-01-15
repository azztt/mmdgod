# Copyright (c) Meta Platforms, Inc. and affiliates
# Modified for RGB-D support
"""
RCNN3D with RGB-D (Depth) support for Domain-Generalized 3D Detection.

This extends the base RCNN3D to:
1. Process both RGB and depth inputs
2. Use a dual-stream backbone for modality-specific feature extraction
3. Fuse RGB and depth features before passing to FPN/RPN/ROI heads
4. Support domain generalization via frozen/partially-frozen encoders
"""
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from detectron2.layers import ShapeSpec, batched_nms
from detectron2.utils.visualizer import Visualizer
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import Instances, ImageList
from detectron2.utils.events import get_event_storage
from detectron2.data import MetadataCatalog

from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.utils.logger import _log_api_usage
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY, GeneralizedRCNN

from cubercnn.modeling.roi_heads import build_roi_heads
from cubercnn.modeling.backbone import build_simple_dual_encoder_backbone
from cubercnn import util, vis

from pytorch3d.transforms import rotation_6d_to_matrix


@META_ARCH_REGISTRY.register()
class RCNN3D_RGBD(GeneralizedRCNN):
    """3D RCNN with RGB-D (depth) support for domain generalization.
    
    This model processes both RGB images and depth maps:
    - RGB: Domain-invariant features via frozen backbone
    - Depth: Geometry-aware features via partially-frozen backbone
    - Fusion: Combined features for 3D detection
    
    Architecture:
        RGB Image → RGB Encoder (frozen) → RGB Features ─┐
                                                         ├─→ Fusion → FPN → RPN → ROI Heads → 3D Boxes
        Depth Map → Depth Encoder (partial) → Depth Features ─┘
    
    The model maintains compatibility with the original RCNN3D interface
    while adding depth processing capabilities.
    """
    
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator,
        roi_heads,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: str = "BGR",
        vis_period: int = 0,
        depth_pixel_mean: float = 0.0,
        depth_pixel_std: float = 1.0,
    ):
        """Initialize RCNN3D_RGBD.
        
        Args:
            backbone: Feature extractor backbone (dual encoder or standard)
            proposal_generator: RPN for proposal generation
            roi_heads: ROI heads for 3D box prediction
            pixel_mean: Per-channel RGB mean for normalization
            pixel_std: Per-channel RGB std for normalization  
            input_format: Input image format (BGR or RGB)
            vis_period: Visualization period (0 = disabled)
            depth_pixel_mean: Mean for depth normalization
            depth_pixel_std: Std for depth normalization
        """
        super().__init__(
            backbone=backbone,
            proposal_generator=proposal_generator,
            roi_heads=roi_heads,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
            input_format=input_format,
            vis_period=vis_period,
        )
        
        # Depth normalization parameters
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
        """Build RCNN3D_RGBD from config.
        
        Args:
            cfg: Detectron2 config
            priors: Optional priors for CubeHead
            
        Returns:
            Dict of constructor arguments
        """
        backbone = build_backbone_rgbd(cfg, priors=priors)
        
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
    
    def preprocess_image(self, batched_inputs: List[Dict]) -> Tuple[ImageList, Optional[torch.Tensor]]:
        """Preprocess both RGB images and depth maps.
        
        Args:
            batched_inputs: List of input dicts with 'image' and optional 'depth'
            
        Returns:
            Tuple of (normalized RGB ImageList, depth tensor or None)
        """
        # Process RGB images (standard Detectron2 preprocessing)
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
            # Normalize depth
            depths = [(d - self.depth_pixel_mean) / self.depth_pixel_std for d in depths]
            # Pad to same size as images
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
                - instances (training only): Ground truth boxes
                
        Returns:
            Training: Dict of losses
            Inference: List of Instances with predictions
        """
        if not self.training:
            return self.inference(batched_inputs)
        
        # Preprocess inputs
        images, depths = self.preprocess_image(batched_inputs)
        
        # Compute scaling factors
        im_scales_ratio = [
            info['height'] / im.shape[1] 
            for (info, im) in zip(batched_inputs, images)
        ]
        
        # Get camera intrinsics
        Ks = [torch.FloatTensor(info['K']) for info in batched_inputs]
        
        # Get ground truth instances
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        # Extract features (backbone handles RGB-D fusion internally)
        if hasattr(self.backbone, 'forward') and depths is not None:
            # Check if backbone accepts depth
            import inspect
            sig = inspect.signature(self.backbone.forward)
            if 'depth' in sig.parameters:
                features = self.backbone(images.tensor, depth=depths)
            else:
                # Backbone doesn't support depth, concatenate to image channels
                # This is a fallback for standard backbones
                combined = torch.cat([images.tensor, depths], dim=1)
                features = self.backbone(combined)
        else:
            features = self.backbone(images.tensor)
        
        # Generate proposals
        proposals, proposal_losses = self.proposal_generator(
            images, features, gt_instances
        )
        
        # ROI heads for 3D detection
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
        losses.update(proposal_losses)
        
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
            do_postprocess: Whether to resize outputs to original size
            
        Returns:
            List of Instances with predictions
        """
        assert not self.training
        
        # Preprocess inputs
        images, depths = self.preprocess_image(batched_inputs)
        
        # Compute scaling factors
        im_scales_ratio = [
            info['height'] / im.shape[1] 
            for (info, im) in zip(batched_inputs, images)
        ]
        
        # Get camera intrinsics  
        Ks = [torch.FloatTensor(info['K']) for info in batched_inputs]
        
        # Extract features
        if hasattr(self.backbone, 'forward') and depths is not None:
            import inspect
            sig = inspect.signature(self.backbone.forward)
            if 'depth' in sig.parameters:
                features = self.backbone(images.tensor, depth=depths)
            else:
                combined = torch.cat([images.tensor, depths], dim=1)
                features = self.backbone(combined)
        else:
            features = self.backbone(images.tensor)
        
        # Pass oracle 2D boxes into the RoI heads if provided
        if type(batched_inputs == list) and np.any(['oracle2D' in b for b in batched_inputs]):
            oracles = [b['oracle2D'] for b in batched_inputs]
            results, _ = self.roi_heads(
                images, features, oracles, Ks, im_scales_ratio, None
            )
        else:
            # Normal inference: generate proposals then detect
            proposals, _ = self.proposal_generator(images, features, None)
            results, _ = self.roi_heads(
                images, features, proposals, Ks, im_scales_ratio, None
            )
        
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting not supported for postprocess"
            return GeneralizedRCNN._postprocess(
                results, batched_inputs, images.image_sizes
            )
        else:
            return results
    
    def visualize_training(self, batched_inputs, proposals, instances):
        """Visualize training samples with predictions.
        
        Extends base RCNN3D visualization with depth visualization.
        """
        storage = get_event_storage()
        max_vis_prop = 20
        
        if not hasattr(self, 'thing_classes'):
            self.thing_classes = MetadataCatalog.get('omni3d_model').thing_classes
            self.num_classes = len(self.thing_classes)
        
        for input, prop, instances_i in zip(batched_inputs, proposals, instances):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            img_3DGT = np.ascontiguousarray(img.copy()[:, :, [2, 1, 0]])
            img_3DPR = np.ascontiguousarray(img.copy()[:, :, [2, 1, 0]])
            
            # Visualize depth if available
            if "depth" in input:
                depth = input["depth"].cpu().numpy()
                if depth.ndim == 3:
                    depth = depth[0]
                # Normalize for visualization
                depth_vis = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
                depth_vis = (depth_vis * 255).astype(np.uint8)
                depth_vis = np.stack([depth_vis] * 3, axis=-1)
                storage.put_image("Depth input", depth_vis.transpose(2, 0, 1))
            
            # Visualize 2D GT and proposals
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            
            vis_img_rpn = np.concatenate((anno_img, prop_img), axis=1)
            vis_img_rpn = vis_img_rpn.transpose(2, 0, 1)
            storage.put_image(
                "Left: GT 2D boxes; Right: Predicted proposals", 
                vis_img_rpn
            )
            
            # Visualize 3D (same as base RCNN3D)
            K = torch.tensor(input['K'], device=self.device)
            scale = input['height'] / img.shape[0]
            fx, sx = (val.item() / scale for val in K[0, [0, 2]])
            fy, sy = (val.item() / scale for val in K[1, [1, 2]])
            
            K_scaled = torch.tensor(
                [[1/scale, 0, 0], [0, 1/scale, 0], [0, 0, 1.0]],
                dtype=torch.float32, device=self.device
            ) @ K
            
            gts_per_image = input["instances"]
            gt_classes = gts_per_image.gt_classes
            
            # Filter irrelevant groundtruth
            fg_selection_mask = (gt_classes != -1) & (gt_classes < self.num_classes)
            
            if fg_selection_mask.sum() > 0:
                gt_classes = gt_classes[fg_selection_mask]
                gt_class_names = [self.thing_classes[i] for i in gt_classes]
                gt_boxes = gts_per_image.gt_boxes.tensor[fg_selection_mask]
                gt_poses = gts_per_image.gt_poses[fg_selection_mask]
                gt_boxes3D = gts_per_image.gt_boxes3D[fg_selection_mask]
                
                gt_z = gt_boxes3D[:, 2]
                gt_x3D = gt_z * (gt_boxes3D[:, 0] - sx) / fx
                gt_y3D = gt_z * (gt_boxes3D[:, 1] - sy) / fy
                
                gt_center_3D = torch.stack((gt_x3D, gt_y3D, gt_z)).T
                gt_boxes3D_XYZ_WHL = torch.cat(
                    (gt_center_3D, gt_boxes3D[:, 3:6]), dim=1
                )
                
                gt_colors = torch.tensor(
                    [util.get_color(i) for i in range(len(gt_boxes3D_XYZ_WHL))],
                    device=self.device
                ) / 255.0
                
                gt_meshes = util.mesh_cuboid(gt_boxes3D_XYZ_WHL, gt_poses, gt_colors)
                gt_meshes = [gt_meshes.__getitem__(i) for i in range(len(gt_meshes))]
                
                img_3DGT = vis.draw_scene_view(
                    img_3DGT, K_scaled.cpu().numpy(), gt_meshes,
                    text=gt_class_names, mode='front',
                    blend_weight=0.0, blend_weight_overlay=0.85
                )
            
            # Predicted 3D boxes
            if len(instances_i) > 0:
                keep = batched_nms(
                    instances_i.pred_boxes.tensor,
                    instances_i.scores,
                    torch.zeros(len(instances_i.scores), dtype=torch.long, 
                               device=instances_i.scores.device),
                    self.roi_heads.box_predictor.test_nms_thresh
                )
                keep = keep[:max_vis_prop]
                
                pred_xyzwhl = torch.cat(
                    (instances_i.pred_center_cam[keep], 
                     instances_i.pred_dimensions[keep]), dim=1
                )
                pred_pose = instances_i.pred_pose[keep]
                
                pred_colors = torch.tensor(
                    [util.get_color(i) for i in range(len(keep))],
                    device=self.device
                ) / 255.0
                
                pred_classes = instances_i.pred_classes[keep]
                pred_scores = instances_i.scores[keep]
                pred_class_names = [
                    f'{self.thing_classes[i]} {s:.2f}' 
                    for i, s in zip(pred_classes, pred_scores)
                ]
                
                pred_meshes = util.mesh_cuboid(pred_xyzwhl, pred_pose, pred_colors)
                pred_meshes = [pred_meshes.__getitem__(i).detach() 
                              for i in range(len(pred_meshes))]
                
                img_3DPR = vis.draw_scene_view(
                    img_3DPR, K_scaled.cpu().numpy(), pred_meshes,
                    text=pred_class_names, mode='front',
                    blend_weight=0.0, blend_weight_overlay=0.85
                )
            
            vis_img_3d = np.concatenate((img_3DGT, img_3DPR), axis=1)
            vis_img_3d = vis_img_3d[:, :, [2, 1, 0]]  # RGB
            vis_img_3d = vis_img_3d.astype(np.uint8).transpose(2, 0, 1)
            
            storage.put_image(
                "Left: GT 3D cuboids; Right: Predicted 3D cuboids", 
                vis_img_3d
            )
            
            break  # Only visualize one image per batch


def build_model_rgbd(cfg, priors=None):
    """Build RCNN3D_RGBD model.
    
    Args:
        cfg: Detectron2 config
        priors: Optional priors for CubeHead
        
    Returns:
        RCNN3D_RGBD model instance
    """
    meta_arch = cfg.MODEL.META_ARCHITECTURE
    model = META_ARCH_REGISTRY.get(meta_arch)(cfg, priors=priors)
    model.to(torch.device(cfg.MODEL.DEVICE))
    _log_api_usage("modeling.meta_arch." + meta_arch)
    return model


def build_backbone_rgbd(cfg, input_shape=None, priors=None):
    """Build backbone for RGB-D input.
    
    If USE_DUAL_ENCODER is True in config, builds a dual encoder backbone.
    Otherwise builds a standard backbone.
    
    Args:
        cfg: Detectron2 config
        input_shape: Optional input shape specification
        priors: Optional priors
        
    Returns:
        Backbone instance
    """
    if input_shape is None:
        # Add depth channel if using dual encoder
        channels = len(cfg.MODEL.PIXEL_MEAN)
        if getattr(cfg.MODEL, 'USE_DUAL_ENCODER', False):
            # Dual encoder handles RGB and depth separately
            channels = 3
        input_shape = ShapeSpec(channels=channels)
    
    use_dual_encoder = getattr(cfg.MODEL, 'USE_DUAL_ENCODER', False)
    
    if use_dual_encoder:
        backbone = build_simple_dual_encoder_backbone(cfg, input_shape, priors)
    else:
        # Use standard backbone name from config
        backbone_name = cfg.MODEL.BACKBONE.NAME
        backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, input_shape, priors)
    
    assert isinstance(backbone, Backbone)
    return backbone
