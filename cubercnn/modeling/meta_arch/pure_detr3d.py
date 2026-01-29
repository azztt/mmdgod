# Copyright (c) Meta Platforms, Inc. and affiliates
# Pure DETR3D Architecture - No RPN, Learnable Queries with Multi-Scale Deformable Attention
"""
PureDETR3D: Pure DETR-style 3D Object Detection with Multi-Scale Deformable Attention.

Key differences from DETR3D_RGBD:
1. NO RPN - uses learnable object queries directly
2. Multi-scale deformable cross-attention between queries and feature maps (p2, p3, p4, p5)
3. Simpler architecture, fully end-to-end

Architecture:
    RGB Image → DINO Encoder (frozen) ─────────┐
                                                ├─→ Fusion → Multi-scale Features {p2, p3, p4, p5}
    Depth Map → DINO Encoder (partial freeze) ─┘                    ↓
                                                   100 Learnable Object Queries
                                                              ↓
                                                   Multi-Scale Deformable Decoder
                                                              ↓
                                                   3D Box + Class Predictions
"""

from typing import Dict, List, Optional, Tuple
import copy
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import sigmoid_focal_loss, box_iou, generalized_box_iou

logger = logging.getLogger(__name__)

from detectron2.layers import ShapeSpec
from detectron2.structures import Instances, ImageList, Boxes
from detectron2.utils.events import get_event_storage
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling.backbone import BACKBONE_REGISTRY

from pytorch3d.transforms import rotation_6d_to_matrix

# Import multi-scale deformable attention from detrex
from detrex.layers import MultiScaleDeformableAttention

from cubercnn.util import math_util as util


def rotation_matrix_to_yaw(R):
    """Extract yaw angle from a rotation matrix (gravity-aligned Y-axis rotation).
    
    Args:
        R: Rotation matrix of shape (..., 3, 3)
    
    Returns:
        yaw: Yaw angle in radians of shape (...,)
    """
    # For Y-axis rotation: R_y(θ) = [[cos(θ), 0, sin(θ)], [0, 1, 0], [-sin(θ), 0, cos(θ)]]
    # Extract yaw: θ = atan2(R[0,2], R[2,2])
    return torch.atan2(R[..., 0, 2], R[..., 2, 2])


def yaw_to_rotation_matrix(yaw):
    """Convert yaw angle to rotation matrix (Y-axis rotation).
    
    Args:
        yaw: Yaw angle in radians of shape (...,)
    
    Returns:
        R: Rotation matrix of shape (..., 3, 3)
    """
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)
    
    # R_y(θ) = [[cos(θ), 0, sin(θ)], [0, 1, 0], [-sin(θ), 0, cos(θ)]]
    R = torch.stack([
        torch.stack([cos_yaw, zeros, sin_yaw], dim=-1),
        torch.stack([zeros, ones, zeros], dim=-1),
        torch.stack([-sin_yaw, zeros, cos_yaw], dim=-1)
    ], dim=-2)
    
    return R


def inverse_sigmoid(x, eps=1e-5):
    """Inverse sigmoid function."""
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionEmbeddingSine(nn.Module):
    """
    Sinusoidal positional embedding for 2D feature maps.
    Standard DETR-style positional encoding.
    """
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (B, C, H, W)
        Returns:
            pos: Tensor of shape (B, num_pos_feats*2, H, W)
        """
        B, C, H, W = x.shape
        device = x.device
        dtype = x.dtype
        
        not_mask = torch.ones((B, H, W), device=device, dtype=torch.bool)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class DeformableTransformerDecoderLayer(nn.Module):
    """
    Deformable Transformer Decoder Layer with:
    - Self-attention among queries (standard attention)
    - Multi-scale deformable cross-attention to feature maps
    - FFN
    
    Uses the official detrex MultiScaleDeformableAttention implementation.
    """
    def __init__(
        self,
        d_model=256,
        n_heads=8,
        n_levels=4,
        n_points=4,
        d_ffn=1024,
        dropout=0.1,
    ):
        super().__init__()
        
        # Self-attention (standard attention among queries)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Multi-scale deformable cross-attention
        self.cross_attn = MultiScaleDeformableAttention(
            embed_dim=d_model,
            num_levels=n_levels,
            num_heads=n_heads,
            num_points=n_points,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        
        # FFN
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = nn.ReLU(inplace=True)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)
    
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt
    
    def forward(
        self,
        tgt,
        query_pos,
        reference_points,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_padding_mask=None,
    ):
        """
        Args:
            tgt: (B, num_queries, d_model) - query embeddings
            query_pos: (B, num_queries, d_model) - query positional embeddings  
            reference_points: (B, num_queries, n_levels, 2) - normalized reference points per level
            src: (B, sum(H_l * W_l), d_model) - flattened multi-scale features
            src_spatial_shapes: (n_levels, 2) - spatial shapes (H, W) of each level
            src_level_start_index: (n_levels,) - start index of each level in flattened src
            src_padding_mask: (B, sum(H_l * W_l)) - padding mask (optional)
        """
        # Self-attention among queries
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Multi-scale deformable cross-attention
        tgt2 = self.cross_attn(
            query=self.with_pos_embed(tgt, query_pos),
            reference_points=reference_points,
            value=src,
            spatial_shapes=src_spatial_shapes,
            level_start_index=src_level_start_index,
            key_padding_mask=src_padding_mask,
        )
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt = self.forward_ffn(tgt)
        
        return tgt


class DeformableTransformerDecoder(nn.Module):
    """
    Deformable Transformer Decoder.
    
    Takes learnable queries and multi-scale feature maps,
    outputs refined query embeddings at each layer.
    """
    def __init__(
        self,
        decoder_layer,
        num_layers,
        d_model=256,
        return_intermediate=True,
    ):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.d_model = d_model
        
        # Reference point head (predicts 2D reference points from query position)
        self.reference_points_head = nn.Linear(d_model, 2)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.reference_points_head.weight)
        nn.init.constant_(self.reference_points_head.bias, 0)
    
    def forward(
        self,
        tgt,
        query_pos,
        src,
        src_spatial_shapes,
        src_level_start_index,
        src_valid_ratios,
        src_padding_mask=None,
    ):
        """
        Args:
            tgt: (B, num_queries, d_model) - query embeddings
            query_pos: (B, num_queries, d_model) - query positional embeddings
            src: (B, sum(H_l * W_l), d_model) - flattened multi-scale features
            src_spatial_shapes: (n_levels, 2) - spatial shapes of each level
            src_level_start_index: (n_levels,) - start index of each level
            src_valid_ratios: (B, n_levels, 2) - valid ratio for each level (1.0 if no padding)
            src_padding_mask: (B, sum(H_l * W_l)) - padding mask
        """
        output = tgt
        
        # Compute initial reference points from query positional embeddings
        # Reference points are in [0, 1] normalized coordinates
        reference_points = self.reference_points_head(query_pos).sigmoid()  # (B, num_queries, 2)
        
        # Expand reference points for each level and scale by valid ratios
        # (B, num_queries, 2) -> (B, num_queries, n_levels, 2)
        n_levels = src_spatial_shapes.shape[0]
        reference_points = reference_points[:, :, None, :] * src_valid_ratios[:, None, :, :]
        
        intermediate = []
        intermediate_reference_points = []
        
        for layer in self.layers:
            output = layer(
                output,
                query_pos,
                reference_points,
                src,
                src_spatial_shapes,
                src_level_start_index,
                src_padding_mask,
            )
            
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        
        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)
        
        return output, reference_points


@META_ARCH_REGISTRY.register()
class PureDETR3D(nn.Module):
    """Pure DETR-style 3D Object Detection without RPN.
    
    Uses multi-scale deformable attention for efficient cross-attention
    between learnable queries and multi-scale feature maps from the backbone.
    
    Features:
    - Multi-scale features: p2, p3, p4, p5 from FPN
    - 100 learnable object queries
    - Multi-scale deformable cross-attention (4 levels, 4 points per level)
    - Hungarian matching for training
    - 3D box prediction: 2D center, depth, dimensions, 6D rotation
    """
    
    def __init__(self, cfg, priors=None):
        super().__init__()
        
        self.device = torch.device(cfg.MODEL.DEVICE)
        
        # Build backbone
        backbone_name = cfg.MODEL.BACKBONE.NAME
        self.backbone = BACKBONE_REGISTRY.get(backbone_name)(cfg, None)
        
        # Get feature dimensions from backbone
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = {k: v.stride for k, v in backbone_shape.items()}
        self.feature_channels = {k: v.channels for k, v in backbone_shape.items()}
        
        # Feature levels to use
        self.in_features = ['p2', 'p3', 'p4', 'p5']
        self.num_feature_levels = len(self.in_features)
        
        # DETR decoder settings
        d_model = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'FEATURE_DIM', 256)
        num_queries = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'NUM_QUERIES', 100)
        num_decoder_layers = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'NUM_DECODER_LAYERS', 6)
        num_heads = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'NUM_HEADS', 8)
        dim_feedforward = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'DIM_FEEDFORWARD', 1024)
        dropout = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'DROPOUT', 0.1)
        n_points = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'N_POINTS', 4)
        pose_type = cfg.MODEL.ROI_CUBE_HEAD.POSE_TYPE
        
        self.d_model = d_model
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.num_queries = num_queries
        self.pose_type = pose_type  # Store pose type for loss computation
        
        # Input projections for each feature level (project to d_model)
        self.input_proj = nn.ModuleList()
        for key in self.in_features:
            in_channels = self.feature_channels[key]
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, d_model, kernel_size=1),
                    nn.GroupNorm(32, d_model),
                )
            )
        
        # Positional encoding for feature maps
        self.pos_encoder = PositionEmbeddingSine(d_model // 2, normalize=True)
        
        # Level embedding (learned embedding for each feature level)
        self.level_embed = nn.Parameter(torch.Tensor(self.num_feature_levels, d_model))
        nn.init.normal_(self.level_embed)
        
        # Learnable query embeddings (content + positional)
        self.query_embed = nn.Embedding(num_queries, d_model * 2)
        
        # Decoder
        decoder_layer = DeformableTransformerDecoderLayer(
            d_model=d_model,
            n_heads=num_heads,
            n_levels=self.num_feature_levels,
            n_points=n_points,
            d_ffn=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = DeformableTransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            d_model=d_model,
            return_intermediate=True,
        )
        
        # Output heads
        self.class_head = nn.Linear(d_model, self.num_classes + 1)  # +1 for background
        self.center2d_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 2),
        )
        self.depth_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 1),
        )
        self.dims_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 3),
        )
        pose_dim = 1 if pose_type == 'yaw' else 6 if pose_type == '6d' else 4 if pose_type == 'quaternion' else 3
        self.pose_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, pose_dim),
        )
        
        # Loss weights
        self.loss_weights = {
            'cls': getattr(cfg.MODEL.ROI_CUBE_HEAD, 'LOSS_W_CLS', 2.0),
            'center2d': getattr(cfg.MODEL.ROI_CUBE_HEAD, 'LOSS_W_XY', 1.0),
            'depth': getattr(cfg.MODEL.ROI_CUBE_HEAD, 'LOSS_W_Z', 1.0),
            'dims': getattr(cfg.MODEL.ROI_CUBE_HEAD, 'LOSS_W_DIMS', 1.0),
            'pose': getattr(cfg.MODEL.ROI_CUBE_HEAD, 'LOSS_W_POSE', 1.0),
            'giou': getattr(cfg.MODEL.ROI_CUBE_HEAD, 'LOSS_W_GIOU', 2.0),
            'corners': getattr(cfg.MODEL.ROI_CUBE_HEAD, 'LOSS_W_JOINT', 0.25),
        }
        
        # Focal loss parameters (Deformable DETR defaults)
        self.focal_alpha = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'FOCAL_ALPHA', 0.25)
        self.focal_gamma = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'FOCAL_GAMMA', 2.0)
        
        # Note: Focal length is taken from per-image intrinsics (K matrix) in the data
        # No need for virtual_focal - we use the actual camera intrinsics
        
        # Normalization buffers
        self.register_buffer("pixel_mean", torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1), False)
        self.register_buffer("depth_pixel_mean", torch.tensor([getattr(cfg.MODEL, 'DEPTH_PIXEL_MEAN', 0.0)]).view(-1, 1, 1), False)
        self.register_buffer("depth_pixel_std", torch.tensor([getattr(cfg.MODEL, 'DEPTH_PIXEL_STD', 1.0)]).view(-1, 1, 1), False)
        
        # Test settings
        self.test_score_thresh = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'TEST_SCORE_THRESH', 0.05)
        self.test_nms_thresh = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'TEST_NMS_THRESH', 0.5)
        self.test_topk = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'TEST_TOPK_PER_IMAGE', 100)
        
        self._init_weights()
        self.to(self.device)
    
    def _init_weights(self):
        """Initialize weights."""
        # Initialize classification head to predict background initially
        nn.init.constant_(self.class_head.bias, 0)
        nn.init.constant_(self.class_head.weight, 0)
        nn.init.constant_(self.class_head.bias[-1], 2.0)  # Background bias
        
        # Initialize query embeddings
        nn.init.normal_(self.query_embed.weight, std=0.01)
        
        # Initialize input projections
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight)
            nn.init.constant_(proj[0].bias, 0)
        
        # Initialize output heads
        for head in [self.center2d_head, self.depth_head, self.dims_head, self.pose_head]:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, std=0.01)
                    nn.init.constant_(layer.bias, 0)
    
    @property
    def size_divisibility(self):
        return getattr(self.backbone, 'size_divisibility', 0)
    
    def preprocess_image(self, batched_inputs: List[Dict]) -> Tuple[ImageList, Optional[torch.Tensor]]:
        """Preprocess RGB and depth inputs."""
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        
        depths = None
        if "depth" in batched_inputs[0]:
            depths = [x["depth"].to(self.device) for x in batched_inputs]
            depths = [(d - self.depth_pixel_mean) / self.depth_pixel_std for d in depths]
            depths = ImageList.from_tensors(depths, self.size_divisibility).tensor
        
        return images, depths
    
    def prepare_multi_scale_features(self, features: Dict[str, torch.Tensor]):
        """
        Prepare multi-scale features for deformable attention.
        
        Args:
            features: Dict of feature maps {p2, p3, p4, p5}
            
        Returns:
            src_flatten: (B, sum(H_l * W_l), d_model) - flattened features
            pos_flatten: (B, sum(H_l * W_l), d_model) - flattened positional encodings
            spatial_shapes: (n_levels, 2) - (H, W) for each level
            level_start_index: (n_levels,) - start index of each level
            valid_ratios: (B, n_levels, 2) - valid ratio for each level
        """
        srcs = []
        pos_embeds = []
        spatial_shapes = []
        
        for lvl, key in enumerate(self.in_features):
            feat = features[key]
            src = self.input_proj[lvl](feat)  # Project to d_model
            pos = self.pos_encoder(src)  # Positional encoding
            
            B, C, H, W = src.shape
            spatial_shapes.append((H, W))
            
            # Flatten spatial dimensions and add level embedding
            src = src.flatten(2).transpose(1, 2)  # (B, HW, C)
            pos = pos.flatten(2).transpose(1, 2)  # (B, HW, C)
            src = src + self.level_embed[lvl].view(1, 1, -1)
            
            srcs.append(src)
            pos_embeds.append(pos)
        
        # Concatenate all levels
        src_flatten = torch.cat(srcs, dim=1)  # (B, sum(HW), C)
        pos_flatten = torch.cat(pos_embeds, dim=1)  # (B, sum(HW), C)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat([
            spatial_shapes.new_zeros((1,)),
            spatial_shapes.prod(1).cumsum(0)[:-1]
        ])
        
        # Valid ratios (all 1.0 since we don't use padding within feature maps)
        B = src_flatten.shape[0]
        valid_ratios = torch.ones((B, self.num_feature_levels, 2), device=src_flatten.device)
        
        return src_flatten, pos_flatten, spatial_shapes, level_start_index, valid_ratios
    
    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """Forward pass."""
        if not self.training:
            return self.inference(batched_inputs)
        
        # Preprocess
        images, depths = self.preprocess_image(batched_inputs)
        image_sizes = images.image_sizes
        Ks = [torch.FloatTensor(info['K']).to(self.device) for info in batched_inputs]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        depth_stats = [info.get('depth_stats', {}) for info in batched_inputs]
        
        # Extract features from backbone
        features = self.backbone(images.tensor, depth=depths)
        
        # Prepare multi-scale features for deformable attention
        src, pos, spatial_shapes, level_start_index, valid_ratios = self.prepare_multi_scale_features(features)
        
        B = src.shape[0]
        
        # Query embeddings (split into content and positional)
        query_embed = self.query_embed.weight  # (num_queries, d_model*2)
        query_embed = query_embed.unsqueeze(0).repeat(B, 1, 1)
        tgt, query_pos = query_embed.split(self.d_model, dim=-1)
        
        # Run decoder
        hs, reference_points = self.decoder(
            tgt, query_pos, src, spatial_shapes, level_start_index, valid_ratios
        )
        
        # Use last layer output
        hs = hs[-1]  # (B, num_queries, d_model)
        
        # Output predictions
        outputs = {
            'pred_logits': self.class_head(hs),
            'pred_center2d': self.center2d_head(hs).sigmoid(),
            'pred_depth': self.depth_head(hs),
            'pred_dims': self.dims_head(hs),
            'pred_pose': self.pose_head(hs),
        }
        
        # Compute losses
        losses = self.compute_losses(outputs, gt_instances, Ks, image_sizes, depth_stats)
        
        return losses
    
    def compute_losses(self, outputs, gt_instances, Ks, image_sizes, depth_stats):
        """Compute DETR-style losses with Hungarian matching."""
        from scipy.optimize import linear_sum_assignment
        
        B = len(gt_instances)
        device = outputs['pred_logits'].device
        
        losses = {
            'loss_cls': torch.tensor(0.0, device=device),
            'loss_center2d': torch.tensor(0.0, device=device),
            'loss_depth': torch.tensor(0.0, device=device),
            'loss_dims': torch.tensor(0.0, device=device),
            'loss_pose': torch.tensor(0.0, device=device),
            'loss_giou': torch.tensor(0.0, device=device),
            'loss_corners': torch.tensor(0.0, device=device),
        }
        
        total_matched = 0
        
        for b in range(B):
            gt = gt_instances[b]
            num_gt = len(gt)
            
            if num_gt == 0:
                # No GT: only classification loss (all should be background)
                pred_logits = outputs['pred_logits'][b]
                # Focal loss on all classes including background (num_classes + 1)
                target_classes_onehot = torch.zeros((self.num_queries, self.num_classes + 1), device=device)
                target_classes_onehot[:, self.num_classes] = 1.0  # All background
                # Apply focal loss to ALL dimensions (including background)
                losses['loss_cls'] += sigmoid_focal_loss(
                    pred_logits, target_classes_onehot,
                    alpha=self.focal_alpha, gamma=self.focal_gamma, reduction='sum'
                ) / self.num_queries
                continue
            
            # Get predictions
            pred_logits = outputs['pred_logits'][b]
            pred_center2d = outputs['pred_center2d'][b]
            pred_depth = outputs['pred_depth'][b]
            pred_dims = outputs['pred_dims'][b]
            pred_pose = outputs['pred_pose'][b]
            
            # Get GT
            gt_classes = gt.gt_classes
            gt_boxes3D = gt.gt_boxes3D  # [u, v, z, w, h, l, X, Y, Z]
            gt_poses = gt.gt_poses
            
            H, W = image_sizes[b]
            gt_center2d = gt_boxes3D[:, :2].clone()
            gt_center2d[:, 0] /= W
            gt_center2d[:, 1] /= H
            gt_z = gt_boxes3D[:, 2:3].clamp(min=1e-4)  # Camera-space depth in meters
            
            # Normalize GT depth to match input normalization (for domain generalization)
            stats = depth_stats[b]
            if stats.get('mode') == 'per_sample':
                # Apply same z-score normalization as input: (z - mean) / std / 3.0, clipped to [-1, 1]
                depth_mean = stats.get('mean', 5.0)
                depth_std = stats.get('std', 2.0)
                gt_z_normalized = ((gt_z - depth_mean) / depth_std) / 3.0
                gt_z_normalized = gt_z_normalized.clamp(-1.0, 1.0)
            elif stats.get('mode') == 'percentile':
                # Percentile normalization
                p5 = stats.get('p5', 0.5)
                p95 = stats.get('p95', 10.0)
                gt_z_normalized = 2.0 * (gt_z - p5) / (p95 - p5 + 1e-4) - 1.0
                gt_z_normalized = gt_z_normalized.clamp(-1.0, 1.0)
            else:
                # Fixed normalization: z / depth_max * 2 - 1
                depth_max = stats.get('max', 20.0)
                gt_z_normalized = (gt_z / depth_max) * 2.0 - 1.0
                gt_z_normalized = gt_z_normalized.clamp(-1.0, 1.0)
            
            gt_dims = gt_boxes3D[:, 3:6].clamp(min=1e-4)  # Clamp to avoid log(0)
            
            # Cost matrix for Hungarian matching
            pred_probs = pred_logits.softmax(-1)
            class_cost = -pred_probs[:, gt_classes]
            center2d_cost = torch.cdist(pred_center2d, gt_center2d, p=1)
            # Match in normalized depth space (both pred and GT are normalized)
            # No need for exp() or clamping - predictions are already in [-3, 3] normalized space
            depth_cost = torch.cdist(pred_depth, gt_z_normalized, p=1)
            
            cost_matrix = class_cost + 5.0 * center2d_cost + 2.0 * depth_cost
            cost_matrix = cost_matrix.detach().cpu().numpy()
            
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            row_ind = torch.tensor(row_ind, device=device, dtype=torch.long)
            col_ind = torch.tensor(col_ind, device=device, dtype=torch.long)
            
            total_matched += len(row_ind)
            
            # Classification loss with focal loss (Deformable DETR style)
            target_classes_onehot = torch.zeros((self.num_queries, self.num_classes + 1), device=device)
            target_classes_onehot[:, self.num_classes] = 1.0  # All background by default
            target_classes_onehot[row_ind, self.num_classes] = 0.0  # Remove background for matched
            target_classes_onehot[row_ind, gt_classes[col_ind]] = 1.0  # Set matched classes
            # Apply focal loss to ALL dimensions (including background)
            losses['loss_cls'] += sigmoid_focal_loss(
                pred_logits, target_classes_onehot,
                alpha=self.focal_alpha, gamma=self.focal_gamma, reduction='sum'
            ) / self.num_queries
            
            if len(row_ind) > 0:
                matched_pred_center2d = pred_center2d[row_ind]
                matched_pred_depth = pred_depth[row_ind]
                matched_pred_dims = pred_dims[row_ind]
                matched_pred_pose = pred_pose[row_ind]
                
                matched_gt_center2d = gt_center2d[col_ind]
                matched_gt_z_camera = gt_z[col_ind]  # Camera-space for 3D reconstruction
                matched_gt_z_normalized = gt_z_normalized[col_ind]  # Normalized for loss
                matched_gt_dims = gt_dims[col_ind]
                matched_gt_poses = gt_poses[col_ind]
                
                losses['loss_center2d'] += F.l1_loss(matched_pred_center2d, matched_gt_center2d)
                
                # GIoU loss: Project 3D boxes to 2D and compute GIoU
                # For predicted boxes: denormalize depth and project corners
                K = Ks[b]
                if stats.get('mode') == 'per_sample':
                    depth_mean = stats.get('mean', 5.0)
                    depth_std = stats.get('std', 2.0)
                    pred_z_linear = (matched_pred_depth.squeeze(-1).clamp(-1.0, 1.0) * 3.0 * depth_std + depth_mean).clamp(min=0.01)
                elif stats.get('mode') == 'percentile':
                    p5 = stats.get('p5', 0.5)
                    p95 = stats.get('p95', 10.0)
                    pred_z_linear = ((matched_pred_depth.squeeze(-1).clamp(-1.0, 1.0) + 1.0) / 2.0 * (p95 - p5) + p5).clamp(min=0.01)
                else:
                    depth_max = stats.get('max', 20.0)
                    pred_z_linear = ((matched_pred_depth.squeeze(-1).clamp(-1.0, 1.0) + 1.0) / 2.0 * depth_max).clamp(min=0.01)
                
                pred_center2d_pixel = matched_pred_center2d.clone()
                pred_center2d_pixel[:, 0] *= W
                pred_center2d_pixel[:, 1] *= H
                
                fx, fy = K[0, 0], K[1, 1]
                cx, cy = K[0, 2], K[1, 2]
                pred_x3d = pred_z_linear * (pred_center2d_pixel[:, 0] - cx) / fx
                pred_y3d = pred_z_linear * (pred_center2d_pixel[:, 1] - cy) / fy
                pred_3d = torch.stack([pred_x3d, pred_y3d, pred_z_linear], dim=1)
                
                pred_dims_linear = matched_pred_dims.exp()
                if self.pose_type == 'yaw':
                    pred_R = yaw_to_rotation_matrix(matched_pred_pose.squeeze(-1))
                else:
                    pred_R = rotation_6d_to_matrix(matched_pred_pose)
                pred_box3d = torch.cat([pred_3d, pred_dims_linear], dim=1)
                pred_corners_3d = util.get_cuboid_verts_faces(pred_box3d, pred_R)[0]  # (N, 8, 3)
                
                # Project predicted 3D corners to 2D
                pred_corners_2d = torch.stack([
                    fx * pred_corners_3d[:, :, 0] / pred_corners_3d[:, :, 2] + cx,
                    fy * pred_corners_3d[:, :, 1] / pred_corners_3d[:, :, 2] + cy,
                ], dim=-1)  # (N, 8, 2)
                
                # Get 2D bounding box from corners (min/max)
                pred_boxes_2d = torch.cat([
                    pred_corners_2d.min(dim=1)[0],  # x_min, y_min
                    pred_corners_2d.max(dim=1)[0],  # x_max, y_max
                ], dim=1)  # (N, 4)
                pred_boxes_norm = pred_boxes_2d / torch.tensor([W, H, W, H], device=device)
                pred_boxes_norm = pred_boxes_norm.clamp(0, 1)
                
                # For GT boxes: project corners to 2D
                gt_3d = gt_boxes3D[col_ind, 6:9]
                gt_box3d = torch.cat([gt_3d, matched_gt_dims], dim=1)
                gt_corners_3d = util.get_cuboid_verts_faces(gt_box3d, matched_gt_poses)[0]
                
                # Project GT 3D corners to 2D
                gt_corners_2d = torch.stack([
                    fx * gt_corners_3d[:, :, 0] / gt_corners_3d[:, :, 2] + cx,
                    fy * gt_corners_3d[:, :, 1] / gt_corners_3d[:, :, 2] + cy,
                ], dim=-1)  # (N, 8, 2)
                
                # Get 2D bounding box from corners
                gt_boxes_2d = torch.cat([
                    gt_corners_2d.min(dim=1)[0],  # x_min, y_min
                    gt_corners_2d.max(dim=1)[0],  # x_max, y_max
                ], dim=1)  # (N, 4)
                gt_boxes_norm = gt_boxes_2d / torch.tensor([W, H, W, H], device=device)
                gt_boxes_norm = gt_boxes_norm.clamp(0, 1)
                
                # Compute GIoU loss
                giou = generalized_box_iou(pred_boxes_norm, gt_boxes_norm)
                losses['loss_giou'] += (1 - torch.diag(giou)).mean()
                
                # Depth loss in NORMALIZED space (no log, no exp - direct L1)
                losses['loss_depth'] += F.l1_loss(matched_pred_depth, matched_gt_z_normalized)
                losses['loss_dims'] += F.l1_loss(matched_pred_dims, matched_gt_dims.log())
                
                # Pose loss: extract yaw from rotation matrix if using yaw type
                if self.pose_type == 'yaw':
                    gt_yaw = rotation_matrix_to_yaw(matched_gt_poses).unsqueeze(-1)  # (N, 1)
                    losses['loss_pose'] += F.l1_loss(matched_pred_pose, gt_yaw)
                else:
                    gt_pose_6d = matched_gt_poses[:, :, :2].reshape(-1, 6)
                    losses['loss_pose'] += F.l1_loss(matched_pred_pose, gt_pose_6d)
                
                # Corner loss: use already computed 3D corners from above
                losses['loss_corners'] += F.l1_loss(pred_corners_3d, gt_corners_3d)
        
        # Normalize by batch size
        for key in losses:
            losses[key] /= max(B, 1)
        
        # Apply loss weights
        losses['loss_cls'] *= self.loss_weights['cls']
        losses['loss_center2d'] *= self.loss_weights['center2d']
        losses['loss_depth'] *= self.loss_weights['depth']
        losses['loss_dims'] *= self.loss_weights['dims']
        losses['loss_pose'] *= self.loss_weights['pose']
        losses['loss_giou'] *= self.loss_weights['giou']
        losses['loss_corners'] *= self.loss_weights['corners']
        
        return losses
    
    def inference(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """Inference mode."""
        images, depths = self.preprocess_image(batched_inputs)
        image_sizes = images.image_sizes
        Ks = [torch.FloatTensor(info['K']).to(self.device) for info in batched_inputs]
        depth_stats = [info.get('depth_stats', {}) for info in batched_inputs]
        
        features = self.backbone(images.tensor, depth=depths)
        src, pos, spatial_shapes, level_start_index, valid_ratios = self.prepare_multi_scale_features(features)
        
        B = src.shape[0]
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        tgt, query_pos = query_embed.split(self.d_model, dim=-1)
        
        hs, _ = self.decoder(tgt, query_pos, src, spatial_shapes, level_start_index, valid_ratios)
        hs = hs[-1]
        
        outputs = {
            'pred_logits': self.class_head(hs),
            'pred_center2d': self.center2d_head(hs).sigmoid(),
            'pred_depth': self.depth_head(hs),
            'pred_dims': self.dims_head(hs),
            'pred_pose': self.pose_head(hs),
        }
        
        results = []
        for b in range(B):
            result = self.postprocess_single(outputs, b, Ks[b], image_sizes[b], depth_stats[b])
            results.append({'instances': result})
        
        return results
    
    def postprocess_single(self, outputs, batch_idx, K, image_size, depth_stats):
        """Post-process predictions for a single image."""
        H, W = image_size
        
        pred_logits = outputs['pred_logits'][batch_idx]
        pred_center2d = outputs['pred_center2d'][batch_idx]
        pred_depth = outputs['pred_depth'][batch_idx]
        pred_dims = outputs['pred_dims'][batch_idx]
        pred_pose = outputs['pred_pose'][batch_idx]
        
        # Get class predictions using softmax (standard DETR approach)
        # Even with focal loss training, we still use softmax at inference
        # because the model has num_classes+1 outputs (including background)
        pred_probs = pred_logits.softmax(-1)
        
        # Score = max probability over object classes (excluding background)
        # This implicitly uses objectness through softmax normalization
        class_probs = pred_probs[:, :-1]  # Remove background dimension
        scores, pred_classes = class_probs.max(-1)
        
        # DEBUG: Check what scores we're getting (first batch only)
        if batch_idx == 0:
            logger.info(f"[Inference] Scores - max: {scores.max().item():.4f}, min: {scores.min().item():.4f}, "
                       f"mean: {scores.mean().item():.4f}, median: {scores.median().item():.4f}")
            logger.info(f"[Inference] Num above threshold {self.test_score_thresh}: {(scores > self.test_score_thresh).sum().item()}/{len(scores)}")
        
        # Filter by score threshold
        keep = scores > self.test_score_thresh
        scores = scores[keep]
        pred_classes = pred_classes[keep]
        pred_center2d = pred_center2d[keep]
        pred_depth = pred_depth[keep]
        pred_dims = pred_dims[keep]
        pred_pose = pred_pose[keep]
        
        if len(scores) == 0:
            result = Instances(image_size)
            result.pred_classes = torch.zeros(0, dtype=torch.long, device=pred_logits.device)
            result.scores = torch.zeros(0, device=pred_logits.device)
            result.pred_boxes = Boxes(torch.zeros(0, 4, device=pred_logits.device))
            result.pred_bbox3D = torch.zeros(0, 8, 3, device=pred_logits.device)
            result.pred_center_cam = torch.zeros(0, 3, device=pred_logits.device)
            result.pred_dimensions = torch.zeros(0, 3, device=pred_logits.device)
            result.pred_pose = torch.zeros(0, 3, 3, device=pred_logits.device)
            return result
        
        # Convert to 3D
        pred_center2d_pixel = pred_center2d.clone()
        pred_center2d_pixel[:, 0] *= W
        pred_center2d_pixel[:, 1] *= H
        
        # Denormalize depth predictions from [-1, 1] normalized space back to camera-space meters
        # Handle missing depth_stats gracefully (fallback to reasonable defaults)
        mode = depth_stats.get('mode', 'fixed') if depth_stats else 'fixed'
        
        if mode == 'per_sample' and depth_stats:
            # Per-sample z-score denormalization
            depth_mean = depth_stats.get('mean', 5.0)
            depth_std = depth_stats.get('std', 2.0)
            pred_z = (pred_depth.squeeze(-1).clamp(-1.0, 1.0) * 3.0 * depth_std + depth_mean).clamp(min=0.01)
        elif mode == 'percentile' and depth_stats:
            # Percentile denormalization
            p5 = depth_stats.get('p5', 0.5)
            p95 = depth_stats.get('p95', 10.0)
            pred_z = ((pred_depth.squeeze(-1).clamp(-1.0, 1.0) + 1.0) / 2.0 * (p95 - p5) + p5).clamp(min=0.01)
        else:
            # Fixed denormalization (or fallback if stats missing)
            depth_max = depth_stats.get('max', 20.0) if depth_stats else 20.0
            pred_z = ((pred_depth.squeeze(-1).clamp(-1.0, 1.0) + 1.0) / 2.0 * depth_max).clamp(min=0.01)
        
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        pred_x = pred_z * (pred_center2d_pixel[:, 0] - cx) / fx
        pred_y = pred_z * (pred_center2d_pixel[:, 1] - cy) / fy
        
        pred_center_cam = torch.stack([pred_x, pred_y, pred_z], dim=1)
        pred_dims_linear = pred_dims.exp()
        
        # Convert pose to rotation matrix
        if self.pose_type == 'yaw':
            pred_R = yaw_to_rotation_matrix(pred_pose.squeeze(-1))
        else:
            pred_R = rotation_6d_to_matrix(pred_pose)
        
        pred_box3d = torch.cat([pred_center_cam, pred_dims_linear], dim=1)
        pred_bbox3D = util.get_cuboid_verts_faces(pred_box3d, pred_R)[0]
        
        # Project to 2D for NMS
        corners_2d = (K @ pred_bbox3D.permute(0, 2, 1)).permute(0, 2, 1)
        corners_2d = corners_2d[:, :, :2] / corners_2d[:, :, 2:3].clamp(min=1e-6)
        
        pred_boxes = torch.cat([
            corners_2d[:, :, 0].min(dim=1, keepdim=True)[0],
            corners_2d[:, :, 1].min(dim=1, keepdim=True)[0],
            corners_2d[:, :, 0].max(dim=1, keepdim=True)[0],
            corners_2d[:, :, 1].max(dim=1, keepdim=True)[0],
        ], dim=1)
        
        # Apply NMS
        from torchvision.ops import nms
        keep_nms = nms(pred_boxes, scores, self.test_nms_thresh)
        keep_nms = keep_nms[:self.test_topk]
        
        result = Instances(image_size)
        result.pred_classes = pred_classes[keep_nms]
        result.scores = scores[keep_nms]
        result.pred_boxes = Boxes(pred_boxes[keep_nms])
        result.pred_bbox3D = pred_bbox3D[keep_nms]
        result.pred_center_cam = pred_center_cam[keep_nms]
        result.pred_center_2D = pred_center2d_pixel[keep_nms]  # 2D center in pixels
        result.pred_dimensions = pred_dims_linear[keep_nms]
        result.pred_pose = pred_R[keep_nms]
        
        return result
