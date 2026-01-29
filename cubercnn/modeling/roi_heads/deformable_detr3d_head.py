# Copyright (c) Meta Platforms, Inc. and affiliates
# Deformable DETR-style Transformer Head for CubeRCNN
"""
Deformable Attention-based 3D Detection Head.

Key differences from standard DETR3DHead:
- Uses deformable attention: O(N×K) vs O(N²) where K is sampling points
- Multi-scale deformable attention for efficient feature aggregation
- Faster inference and training for high-resolution features

Based on:
- Deformable DETR (Zhu et al., ICLR 2021)
- 3DETR architecture
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes
import fvcore.nn.weight_init as weight_init

from pytorch3d.transforms.rotation_conversions import _copysign
from pytorch3d.transforms import (
    rotation_6d_to_matrix, 
    euler_angles_to_matrix, 
    quaternion_to_matrix
)

from .cube_head import ROI_CUBE_HEAD_REGISTRY

import logging
logger = logging.getLogger(__name__)


def get_reference_points(spatial_shapes: List[Tuple[int, int]], device: torch.device) -> torch.Tensor:
    """Generate reference points for all feature levels.
    
    Args:
        spatial_shapes: List of (H, W) for each feature level
        device: Target device
        
    Returns:
        reference_points: (1, sum(H*W), 2) normalized to [0, 1]
    """
    reference_points = []
    for H, W in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H - 0.5, H, device=device) / H,
            torch.linspace(0.5, W - 0.5, W, device=device) / W,
            indexing='ij'
        )
        ref_x = ref_x.flatten()
        ref_y = ref_y.flatten()
        ref = torch.stack([ref_x, ref_y], dim=-1)  # (H*W, 2)
        reference_points.append(ref)
    
    return torch.cat(reference_points, dim=0).unsqueeze(0)  # (1, sum(H*W), 2)


class MSDeformableAttention(nn.Module):
    """Multi-Scale Deformable Attention.
    
    Instead of attending to all positions (O(N²)), samples K positions 
    around a reference point (O(N×K)).
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_levels: Number of feature levels
        n_points: Number of sampling points per attention head per level
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_levels: int = 4,
        n_points: int = 4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        
        # Sampling offsets: each head samples n_points from each level
        # Output: n_heads × n_levels × n_points × 2 (x, y offsets)
        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        
        # Attention weights for each sampling point
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        
        # Value projection
        self.value_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with small offsets centered at reference point."""
        nn.init.constant_(self.sampling_offsets.weight, 0.0)
        nn.init.constant_(self.sampling_offsets.bias, 0.0)
        
        # Initialize offsets in a grid pattern around the reference point
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = grid_init / grid_init.abs().max(-1, keepdim=True)[0]  # Normalize
        
        # Apply to each level and point
        grid_init = grid_init.view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        
        # Scale for different points
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= (i + 1) * 0.1
        
        with torch.no_grad():
            self.sampling_offsets.bias.data = grid_init.view(-1)
        
        # Uniform attention weights
        nn.init.constant_(self.attention_weights.weight, 0.0)
        nn.init.constant_(self.attention_weights.bias, 0.0)
        
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.0)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.0)
    
    def forward(
        self,
        query: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of multi-scale deformable attention.
        
        Args:
            query: (B, N_q, d_model) - query features
            reference_points: (B, N_q, 2) - normalized reference points [0, 1]
            value: (B, sum(H*W), d_model) - flattened multi-scale features
            spatial_shapes: List of (H, W) for each level
            level_start_index: Starting index of each level in value
            
        Returns:
            output: (B, N_q, d_model)
        """
        B, N_q, _ = query.shape
        _, N_v, _ = value.shape
        
        # Project value
        value = self.value_proj(value)  # (B, N_v, d_model)
        value = value.view(B, N_v, self.n_heads, self.head_dim)  # (B, N_v, n_heads, head_dim)
        
        # Compute sampling offsets: (B, N_q, n_heads, n_levels, n_points, 2)
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(B, N_q, self.n_heads, self.n_levels, self.n_points, 2)
        
        # Compute attention weights: (B, N_q, n_heads, n_levels, n_points)
        attention_weights = self.attention_weights(query)
        attention_weights = attention_weights.view(B, N_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.view(B, N_q, self.n_heads, self.n_levels, self.n_points)
        
        # Compute sampling locations
        # reference_points: (B, N_q, 2) -> (B, N_q, 1, 1, 1, 2)
        reference_points = reference_points.view(B, N_q, 1, 1, 1, 2)
        
        # Offset scale for each level (smaller levels = smaller offsets)
        offset_normalizer = torch.tensor(spatial_shapes, device=query.device, dtype=torch.float32)
        offset_normalizer = offset_normalizer.flip(-1).view(1, 1, 1, self.n_levels, 1, 2)
        
        # Final sampling locations
        sampling_locations = reference_points + sampling_offsets / offset_normalizer
        sampling_locations = sampling_locations.clamp(0, 1)  # Keep in valid range
        
        # Sample values using bilinear interpolation
        # This is the key O(N×K) operation vs O(N²)
        output = self._sample_and_aggregate(
            value, sampling_locations, attention_weights, spatial_shapes, level_start_index
        )
        
        # Output projection
        output = self.output_proj(output)
        
        return output
    
    def _sample_and_aggregate(
        self,
        value: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """Sample from multi-scale features and aggregate with attention weights.
        
        Args:
            value: (B, N_v, n_heads, head_dim)
            sampling_locations: (B, N_q, n_heads, n_levels, n_points, 2)
            attention_weights: (B, N_q, n_heads, n_levels, n_points)
            spatial_shapes: List of (H, W)
            level_start_index: Starting indices
            
        Returns:
            output: (B, N_q, d_model)
        """
        B, N_q, n_heads, n_levels, n_points, _ = sampling_locations.shape
        
        # Aggregate across levels and points
        output = torch.zeros(B, N_q, n_heads, self.head_dim, device=value.device, dtype=value.dtype)
        
        for lvl, (H, W) in enumerate(spatial_shapes):
            start_idx = level_start_index[lvl].item()
            end_idx = start_idx + H * W
            
            # Get value for this level: (B, H*W, n_heads, head_dim)
            value_lvl = value[:, start_idx:end_idx, :, :].contiguous()
            
            # Reshape to spatial: (B, n_heads, head_dim, H, W)
            value_lvl = value_lvl.permute(0, 2, 3, 1).reshape(B, n_heads, self.head_dim, H, W)
            
            # Get sampling locations for this level: (B, N_q, n_heads, n_points, 2)
            sampling_locs_lvl = sampling_locations[:, :, :, lvl, :, :].contiguous()
            
            # Convert to grid_sample format: [-1, 1]
            sampling_locs_lvl = 2 * sampling_locs_lvl - 1  # [0,1] -> [-1,1]
            
            # Reshape for grid_sample: (B*n_heads, head_dim, H, W)
            value_lvl = value_lvl.reshape(B * n_heads, self.head_dim, H, W)
            
            # (B, N_q, n_heads, n_points, 2) -> (B*n_heads, N_q, n_points, 2)
            sampling_locs_lvl = sampling_locs_lvl.permute(0, 2, 1, 3, 4).reshape(
                B * n_heads, N_q, n_points, 2
            )
            
            # Sample: (B*n_heads, head_dim, N_q, n_points)
            sampled = F.grid_sample(
                value_lvl,
                sampling_locs_lvl,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False,
            )
            
            # Reshape back: (B, n_heads, head_dim, N_q, n_points)
            sampled = sampled.reshape(B, n_heads, self.head_dim, N_q, n_points)
            
            # Get attention weights for this level: (B, N_q, n_heads, n_points)
            attn_lvl = attention_weights[:, :, :, lvl, :]
            
            # Weight and sum: (B, n_heads, head_dim, N_q)
            sampled = (sampled * attn_lvl.permute(0, 2, 1, 3).unsqueeze(2)).sum(-1)
            
            # Accumulate: (B, N_q, n_heads, head_dim)
            output += sampled.permute(0, 3, 1, 2)
        
        # Flatten heads: (B, N_q, d_model)
        output = output.view(B, N_q, -1)
        
        return output


class DeformableCrossAttentionLayer(nn.Module):
    """Deformable cross-attention layer for 3D head."""
    
    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 8,
        n_levels: int = 4,
        n_points: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Deformable cross-attention
        self.cross_attn = MSDeformableAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_levels=n_levels,
            n_points=n_points,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Self-attention among queries
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        reference_points: torch.Tensor,
        value: torch.Tensor,
        spatial_shapes: List[Tuple[int, int]],
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            query: (B, N_q, d_model) query content
            query_pos: (B, N_q, d_model) query position embedding
            reference_points: (B, N_q, 2) normalized reference points
            value: (B, sum(H*W), d_model) flattened multi-scale features
            spatial_shapes: List of (H, W) for each level
            level_start_index: Starting index for each level
            
        Returns:
            Updated query (B, N_q, d_model)
        """
        # Self-attention among queries
        q = query + query_pos
        q2 = self.self_attn(q, q, query)[0]
        query = query + self.dropout2(q2)
        query = self.norm2(query)
        
        # Deformable cross-attention
        q2 = self.cross_attn(
            query=query + query_pos,
            reference_points=reference_points,
            value=value,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
        )
        query = query + self.dropout1(q2)
        query = self.norm1(query)
        
        # FFN
        query = query + self.ffn(query)
        query = self.norm3(query)
        
        return query


class BoxToQueryEmbedding(nn.Module):
    """Convert proposal boxes to query embeddings and reference points."""
    
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        
        # Project box coordinates to position embedding
        self.box_embed = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        
        # Project box size to content query
        self.size_embed = nn.Sequential(
            nn.Linear(4, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, d_model),
        )
        
        # Learned content query base
        self.content_base = nn.Parameter(torch.randn(1, d_model) * 0.1)
    
    def forward(
        self,
        boxes: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Convert boxes to queries and reference points.
        
        Args:
            boxes: (N, 4) proposal boxes [x1, y1, x2, y2]
            image_size: (H, W)
            
        Returns:
            query_pos: (N, d_model)
            query_content: (N, d_model)
            reference_points: (N, 2) normalized box centers
        """
        H, W = image_size
        N = boxes.shape[0]
        
        if N == 0:
            device = boxes.device
            return (
                torch.zeros(0, self.d_model, device=device),
                torch.zeros(0, self.d_model, device=device),
                torch.zeros(0, 2, device=device),
            )
        
        # Normalize boxes to [0, 1]
        boxes_norm = boxes.clone()
        boxes_norm[:, [0, 2]] = boxes_norm[:, [0, 2]] / W
        boxes_norm[:, [1, 3]] = boxes_norm[:, [1, 3]] / H
        boxes_norm = boxes_norm.clamp(0, 1)
        
        # Reference points = box centers
        cx = (boxes_norm[:, 0] + boxes_norm[:, 2]) / 2
        cy = (boxes_norm[:, 1] + boxes_norm[:, 3]) / 2
        reference_points = torch.stack([cx, cy], dim=-1)  # (N, 2)
        
        # Position embedding from box coordinates
        query_pos = self.box_embed(boxes_norm)
        
        # Content query from box size
        widths = boxes_norm[:, 2] - boxes_norm[:, 0]
        heights = boxes_norm[:, 3] - boxes_norm[:, 1]
        size_feats = torch.stack([widths, heights, cx, cy], dim=-1)
        query_content = self.content_base.expand(N, -1) + self.size_embed(size_feats)
        
        return query_pos, query_content, reference_points


@ROI_CUBE_HEAD_REGISTRY.register()
class DeformableDETR3DHead(nn.Module):
    """Deformable DETR-style 3D Detection Head.
    
    Uses deformable attention for O(N×K) complexity instead of O(N²).
    """
    
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()
        
        # Config
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        pooler_resolution = cfg.MODEL.ROI_CUBE_HEAD.get('POOLER_RESOLUTION', 7)
        in_channels = input_shape.channels
        
        # Get deformable attention params
        detr_cfg = cfg.MODEL.ROI_CUBE_HEAD.get('DETR3D', {})
        d_model = detr_cfg.get('D_MODEL', 256) if isinstance(detr_cfg, dict) else 256
        n_heads = detr_cfg.get('N_HEADS', 8) if isinstance(detr_cfg, dict) else 8
        n_layers = detr_cfg.get('NUM_CROSS_ATTENTION_LAYERS', 3) if isinstance(detr_cfg, dict) else 3
        n_points = detr_cfg.get('N_POINTS', 4) if isinstance(detr_cfg, dict) else 4
        dim_feedforward = detr_cfg.get('DIM_FEEDFORWARD', 1024) if isinstance(detr_cfg, dict) else 1024
        dropout = detr_cfg.get('DROPOUT', 0.1) if isinstance(detr_cfg, dict) else 0.1
        
        self.d_model = d_model
        self.n_levels = 4  # p2, p3, p4, p5
        self.pooler_resolution = pooler_resolution
        self.in_channels = in_channels
        
        # 3D Head params
        self.pose_type = cfg.MODEL.ROI_CUBE_HEAD.POSE_TYPE
        self.z_type = cfg.MODEL.ROI_CUBE_HEAD.Z_TYPE
        self.cluster_bins = cfg.MODEL.ROI_CUBE_HEAD.CLUSTER_BINS
        self.use_conf = cfg.MODEL.ROI_CUBE_HEAD.USE_CONFIDENCE
        
        # Input projection from flattened ROI features (C*H*W) to sequence (H*W, d_model)
        # ROI pooler outputs (N, C*H*W) = (N, 256*7*7) = (N, 12544)
        flattened_size = in_channels * pooler_resolution * pooler_resolution
        self.input_proj_flat = nn.Linear(in_channels, d_model)  # Per-token projection
        
        # Input projection for multi-scale features (for future full deformable attention)
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, d_model, 1),
                nn.GroupNorm(32, d_model),
            )
            for _ in range(self.n_levels)
        ])
        
        # Box to query embedding
        self.box_to_query = BoxToQueryEmbedding(d_model)
        
        # Deformable cross-attention layers
        self.layers = nn.ModuleList([
            DeformableCrossAttentionLayer(
                d_model=d_model,
                n_heads=n_heads,
                n_levels=self.n_levels,
                n_points=n_points,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
        # Output heads (same as CubeHead for compatibility)
        self.bbox_3D_center_deltas = nn.Linear(d_model, self.num_classes * 2)
        self.bbox_3D_dims = nn.Linear(d_model, self.num_classes * 3)
        
        # Z head
        if self.cluster_bins > 1:
            self.bbox_3D_center_depth = nn.Linear(d_model, self.cluster_bins * self.num_classes)
        else:
            self.bbox_3D_center_depth = nn.Linear(d_model, self.num_classes)
        
        # Pose head
        if self.pose_type == '6d':
            self.bbox_3D_pose = nn.Linear(d_model, self.num_classes * 6)
        elif self.pose_type == 'quaternion':
            self.bbox_3D_pose = nn.Linear(d_model, self.num_classes * 4)
        elif self.pose_type == 'euler':
            self.bbox_3D_pose = nn.Linear(d_model, self.num_classes * 3)
        
        # Uncertainty
        if self.use_conf:
            self.bbox_3D_uncertainty = nn.Linear(d_model, self.num_classes)
        
        # Initialize
        self._init_weights()
        
        logger.info(f"DeformableDETR3DHead: {n_layers} layers, {d_model}d, {n_heads} heads, {n_points} points")
    
    def _init_weights(self):
        """Initialize output heads."""
        for module in [self.bbox_3D_center_deltas, self.bbox_3D_dims, 
                       self.bbox_3D_center_depth, self.bbox_3D_pose]:
            nn.init.normal_(module.weight, std=0.001)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Forward pass - compatible with CubeHead interface.
        
        Uses self-attention over pooled ROI features instead of deformable
        cross-attention. This makes it work with the existing ROI heads pipeline.
        
        Args:
            x: Flattened ROI features (N, C*H*W) where C=256, H=W=7 typically
            
        Returns:
            Same format as CubeHead: (box_2d_deltas, box_z, box_dims, box_pose, box_uncert)
        """
        n = x.shape[0]
        if n == 0:
            return self._empty_output(x.device)
        
        # Input is (N, C*H*W) = (N, 12544) from ROI pooler
        # Reshape to (N, H*W, C) = (N, 49, 256) for transformer
        seq_len = self.pooler_resolution * self.pooler_resolution  # 49
        x = x.view(n, self.in_channels, seq_len).permute(0, 2, 1)  # (N, 49, 256)
        
        # Project to d_model if needed
        if self.in_channels != self.d_model:
            x = self.input_proj_flat(x)  # (N, 49, d_model)
        
        # Apply self-attention layers for feature enhancement
        query = x  # (N, seq_len, d_model)
        
        for layer in self.layers:
            # Self-attention only (skip deformable cross-attention)
            query = query + layer.dropout1(layer.self_attn(query, query, query)[0])
            query = layer.norm1(query)
            query = query + layer.ffn(query)
            query = layer.norm3(query)
        
        # Pool sequence to single vector
        features = query.mean(dim=1)  # (N, d_model)
        
        return self._predict(features, n)
    
    def _predict(self, features: torch.Tensor, n: int):
        """Make predictions from query features."""
        # 2D deltas
        box_2d_deltas = self.bbox_3D_center_deltas(features)
        
        # Dimensions
        box_dims = self.bbox_3D_dims(features)
        
        # Pose
        box_pose = self.bbox_3D_pose(features)
        
        # Z depth
        box_z = self.bbox_3D_center_depth(features)
        
        # Uncertainty
        box_uncert = None
        if self.use_conf:
            box_uncert = self.bbox_3D_uncertainty(features).clip(0.01)
        
        # Convert pose to rotation matrices
        if self.pose_type == '6d':
            box_pose = rotation_6d_to_matrix(box_pose.view(-1, 6))
        elif self.pose_type == 'quaternion':
            quats = box_pose.view(-1, 4)
            quats_scales = (quats * quats).sum(1)
            quats = quats / _copysign(torch.sqrt(quats_scales), quats[:, 0])[:, None]
            box_pose = quaternion_to_matrix(quats)
        elif self.pose_type == 'euler':
            box_pose = euler_angles_to_matrix(box_pose.view(-1, 3), 'XYZ')
        
        # Reshape to CubeHead format
        box_2d_deltas = box_2d_deltas.view(n, self.num_classes, 2)
        box_dims = box_dims.view(n, self.num_classes, 3)
        box_pose = box_pose.view(n, self.num_classes, 3, 3)
        
        if self.cluster_bins > 1:
            box_z = box_z.view(n, self.cluster_bins, self.num_classes, -1)
        else:
            box_z = box_z.view(n, self.num_classes, -1)
        
        return box_2d_deltas, box_z, box_dims, box_pose, box_uncert
    
    def _empty_output(self, device):
        """Return empty outputs for zero proposals."""
        box_2d_deltas = torch.zeros(0, self.num_classes, 2, device=device)
        box_z = torch.zeros(0, self.num_classes, 1, device=device)
        box_dims = torch.zeros(0, self.num_classes, 3, device=device)
        box_pose = torch.zeros(0, self.num_classes, 3, 3, device=device)
        box_uncert = torch.zeros(0, self.num_classes, device=device) if self.use_conf else None
        return box_2d_deltas, box_z, box_dims, box_pose, box_uncert
