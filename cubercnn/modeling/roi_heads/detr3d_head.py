# Copyright (c) Meta Platforms, Inc. and affiliates
# 3DETR-style Transformer Decoder Head for CubeRCNN
"""
DETR-style Transformer 3D Head that replaces ROI pooling with cross-attention.

Instead of ROI pooling + MLP (CubeHead), this uses:
1. RPN proposals → Object queries (position + content)
2. Cross-attention with multi-scale fused features
3. Same output format as CubeHead for compatibility with existing losses

Key differences from standard CubeHead:
- No ROI pooling - uses cross-attention to aggregate features
- Query positions initialized from proposal boxes
- Multi-scale feature aggregation via cross-attention
- Same output interface: (box_2d_deltas, box_z, box_dims, box_pose, box_uncert)

This allows using 3DETR-style architecture while keeping CubeRCNN's 
matching, loss computation, and output format.
"""

import math
import copy
import numpy as np
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


def get_clones(module: nn.Module, N: int) -> nn.ModuleList:
    """Create N identical copies of a module."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding2D(nn.Module):
    """2D sinusoidal positional encoding for feature maps."""
    
    def __init__(self, d_model: int, temperature: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.temperature = temperature
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Generate positional encoding.
        
        Args:
            x: Feature map (B, C, H, W)
            
        Returns:
            Positional encoding (B, H*W, d_model)
        """
        B, C, H, W = x.shape
        device = x.device
        
        # Create coordinate grids
        y_embed = torch.arange(H, device=device, dtype=torch.float32)
        x_embed = torch.arange(W, device=device, dtype=torch.float32)
        
        # Normalize to [0, 1]  
        y_embed = y_embed / (H + 1e-6)
        x_embed = x_embed / (W + 1e-6)
        
        # Meshgrid
        y_embed, x_embed = torch.meshgrid(y_embed, x_embed, indexing='ij')
        
        # Flatten
        y_embed = y_embed.flatten()  # (H*W,)
        x_embed = x_embed.flatten()  # (H*W,)
        
        # Sinusoidal encoding
        dim_t = torch.arange(self.d_model // 4, device=device, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / (self.d_model // 4))
        
        pos_x = x_embed.unsqueeze(-1) / dim_t
        pos_y = y_embed.unsqueeze(-1) / dim_t
        
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
        
        pos = torch.cat([pos_x, pos_y], dim=-1)  # (H*W, d_model)
        
        return pos.unsqueeze(0).expand(B, -1, -1)


class BoxToQueryEmbedding(nn.Module):
    """Convert proposal boxes to query embeddings.
    
    Takes 2D bounding boxes and creates position + content queries
    for the transformer decoder.
    """
    
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        
        # Project normalized box coordinates to position embedding
        # Box: [x1, y1, x2, y2] normalized to [0, 1]
        self.box_embed = nn.Sequential(
            nn.Linear(4, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        
        # Project box dimensions/aspect to content query
        self.size_embed = nn.Sequential(
            nn.Linear(4, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, d_model),
        )
        
        # Learned content query base
        self.content_base = nn.Parameter(torch.randn(1, d_model))
    
    def forward(
        self,
        boxes: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert boxes to query embeddings.
        
        Args:
            boxes: Proposal boxes (N, 4) in [x1, y1, x2, y2] format
            image_size: (H, W) for normalization
            
        Returns:
            query_pos: Position embeddings (N, d_model)
            query_content: Content queries (N, d_model)
        """
        H, W = image_size
        N = boxes.shape[0]
        
        if N == 0:
            return (
                torch.zeros(0, self.d_model, device=boxes.device),
                torch.zeros(0, self.d_model, device=boxes.device),
            )
        
        # Normalize boxes to [0, 1]
        boxes_norm = boxes.clone()
        boxes_norm[:, [0, 2]] = boxes_norm[:, [0, 2]] / W
        boxes_norm[:, [1, 3]] = boxes_norm[:, [1, 3]] / H
        boxes_norm = boxes_norm.clamp(0, 1)
        
        # Position embedding from box coordinates
        query_pos = self.box_embed(boxes_norm)
        
        # Content query from box size/aspect
        widths = boxes_norm[:, 2] - boxes_norm[:, 0]
        heights = boxes_norm[:, 3] - boxes_norm[:, 1]
        cx = (boxes_norm[:, 0] + boxes_norm[:, 2]) / 2
        cy = (boxes_norm[:, 1] + boxes_norm[:, 3]) / 2
        size_feats = torch.stack([widths, heights, cx, cy], dim=-1)
        
        query_content = self.content_base.expand(N, -1) + self.size_embed(size_feats)
        
        return query_pos, query_content


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for querying multi-scale features."""
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Cross-attention: queries attend to memory (features)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Self-attention among queries (optional but helps)
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        query_pos: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            query: Query content (N, d_model)
            query_pos: Query position encoding (N, d_model)
            memory: Memory features (M, d_model) - flattened multi-scale
            memory_pos: Memory position encoding (M, d_model)
            
        Returns:
            Updated query (N, d_model)
        """
        # Add position to query and memory for attention
        q = query + query_pos
        k = memory + memory_pos
        
        # Cross-attention with features
        cross_out, _ = self.cross_attn(
            q.unsqueeze(0), k.unsqueeze(0), memory.unsqueeze(0)
        )
        cross_out = cross_out.squeeze(0)
        query = query + self.dropout1(cross_out)
        query = self.norm1(query)
        
        # Self-attention among queries
        q = query + query_pos
        self_out, _ = self.self_attn(
            q.unsqueeze(0), q.unsqueeze(0), query.unsqueeze(0)
        )
        self_out = self_out.squeeze(0)
        query = query + self.dropout2(self_out)
        query = self.norm2(query)
        
        # FFN
        query = query + self.ffn(query)
        query = self.norm3(query)
        
        return query


@ROI_CUBE_HEAD_REGISTRY.register()
class DETR3DHead(nn.Module):
    """3DETR-style head with cross-attention, CubeRCNN output format.
    
    Replaces ROI pooling with cross-attention to aggregate features
    from multi-scale feature maps. Outputs same format as CubeHead
    for compatibility with existing losses.
    
    Output format (same as CubeHead):
        box_2d_deltas: (N, num_classes, 2) - 2D center offsets (NOT USED for 3D-only)
        box_z: (N, num_classes, 1) or (N, cluster_bins, num_classes, 1) - depth
        box_dims: (N, num_classes, 3) - dimensions (w, h, l)
        box_pose: (N, num_classes, 3, 3) - rotation matrices
        box_uncert: (N, num_classes) - confidence
    """
    
    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()
        
        # Settings from config (same as CubeHead)
        self.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.use_conf = cfg.MODEL.ROI_CUBE_HEAD.USE_CONFIDENCE
        self.z_type = cfg.MODEL.ROI_CUBE_HEAD.Z_TYPE
        self.pose_type = cfg.MODEL.ROI_CUBE_HEAD.POSE_TYPE
        self.cluster_bins = cfg.MODEL.ROI_CUBE_HEAD.CLUSTER_BINS
        
        # Transformer settings
        self.d_model = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'FEATURE_DIM', 256)
        self.nhead = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'NUM_HEADS', 8)
        self.num_layers = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'NUM_DECODER_LAYERS', 4)
        self.dim_feedforward = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'DIM_FEEDFORWARD', 1024)
        self.dropout = getattr(cfg.MODEL.ROI_CUBE_HEAD, 'DROPOUT', 0.1)
        
        # Input channel (from ROI pooled features shape)
        input_dim = int(np.prod((input_shape.channels, input_shape.height, input_shape.width)))
        
        # Query embedding from proposal boxes
        self.box_to_query = BoxToQueryEmbedding(self.d_model)
        
        # Positional encoding for features
        self.pos_encoding = PositionalEncoding2D(self.d_model)
        
        # Project ROI features to d_model (fallback for standard usage)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, self.d_model),
            nn.LayerNorm(self.d_model),
            nn.ReLU(inplace=True),
        )
        
        # Feature projection for cross-attention (when using raw FPN features)
        self.feat_proj = nn.Conv2d(input_shape.channels, self.d_model, 1)
        
        # Cross-attention layers
        self.layers = nn.ModuleList([
            CrossAttentionLayer(
                d_model=self.d_model,
                nhead=self.nhead,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
            )
            for _ in range(self.num_layers)
        ])
        
        # Output heads (same outputs as CubeHead)
        cluster_bins = self.cluster_bins if self.cluster_bins > 1 else 1
        
        # XY deltas (for 2D, can be zeroed for 3D-only)
        self.bbox_3D_center_deltas = nn.Linear(self.d_model, self.num_classes * 2)
        
        # Dimensions
        self.bbox_3D_dims = nn.Linear(self.d_model, self.num_classes * 3)
        
        # Pose
        if self.pose_type == '6d':
            self.bbox_3D_pose = nn.Linear(self.d_model, self.num_classes * 6)
        elif self.pose_type == 'quaternion':
            self.bbox_3D_pose = nn.Linear(self.d_model, self.num_classes * 4)
        elif self.pose_type == 'euler':
            self.bbox_3D_pose = nn.Linear(self.d_model, self.num_classes * 3)
        
        # Z depth
        self.bbox_3D_center_depth = nn.Linear(self.d_model, self.num_classes * cluster_bins)
        
        # Confidence
        if self.use_conf:
            self.bbox_3D_uncertainty = nn.Linear(self.d_model, self.num_classes * 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize prediction head weights."""
        for head in [self.bbox_3D_center_deltas, self.bbox_3D_dims, 
                     self.bbox_3D_pose, self.bbox_3D_center_depth]:
            nn.init.normal_(head.weight, std=0.001)
            nn.init.constant_(head.bias, 0)
        
        if self.use_conf:
            nn.init.normal_(self.bbox_3D_uncertainty.weight, std=0.001)
            nn.init.constant_(self.bbox_3D_uncertainty.bias, 5)
    
    def forward(
        self,
        x: torch.Tensor,
        proposal_boxes: Optional[List[Boxes]] = None,
        features: Optional[Dict[str, torch.Tensor]] = None,
        image_sizes: Optional[List[Tuple[int, int]]] = None,
    ):
        """Forward pass with CubeHead-compatible interface.
        
        When called with just x (pooled features), behaves like CubeHead.
        When called with proposal_boxes and features, uses cross-attention.
        
        Args:
            x: Flattened ROI pooled features (N, C*H*W) - standard CubeHead input
            proposal_boxes: List of Boxes for cross-attention (optional)
            features: Dict of FPN features for cross-attention (optional)
            image_sizes: List of (H, W) tuples (optional)
            
        Returns:
            Same as CubeHead:
            - box_2d_deltas: (N, num_classes, 2)
            - box_z: (N, num_classes, 1) or (N, bins, num_classes, 1)
            - box_dims: (N, num_classes, 3)
            - box_pose: (N, num_classes, 3, 3)
            - box_uncert: (N, num_classes) or None
        """
        n = x.shape[0]
        
        if n == 0:
            return self._empty_output(x.device)
        
        # Standard mode: use pooled features directly (like CubeHead)
        if proposal_boxes is None or features is None:
            query = self.input_proj(x)  # (N, d_model)
        else:
            # Cross-attention mode: query features using proposal boxes
            query = self._forward_cross_attention(
                x, proposal_boxes, features, image_sizes
            )
        
        # Predict outputs
        return self._predict(query, n)
    
    def _forward_cross_attention(
        self,
        pooled_features: torch.Tensor,
        proposal_boxes: List[Boxes],
        features: Dict[str, torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ) -> torch.Tensor:
        """Use cross-attention to aggregate features.
        
        Args:
            pooled_features: ROI pooled features (N, C*H*W) - used as initial query
            proposal_boxes: List of Boxes per image
            features: FPN features dict
            image_sizes: List of (H, W) per image
            
        Returns:
            Refined query features (N, d_model)
        """
        device = pooled_features.device
        
        # Initial query from pooled features
        query = self.input_proj(pooled_features)  # (N, d_model)
        
        # Flatten and concat all boxes
        all_boxes = torch.cat([b.tensor for b in proposal_boxes], dim=0)
        num_boxes_per_image = [len(b) for b in proposal_boxes]
        
        # Get query position from boxes
        # Use average image size for normalization (approximate)
        avg_H = sum(s[0] for s in image_sizes) / len(image_sizes)
        avg_W = sum(s[1] for s in image_sizes) / len(image_sizes)
        query_pos, _ = self.box_to_query(all_boxes, (avg_H, avg_W))
        
        # Prepare memory from features (use one scale for simplicity)
        # Could extend to multi-scale
        feat_key = list(features.keys())[0]
        feat = features[feat_key]  # (B, C, H, W)
        
        B, C, H, W = feat.shape
        
        # Project and flatten features
        feat_proj = self.feat_proj(feat)  # (B, d_model, H, W)
        memory = feat_proj.flatten(2).transpose(1, 2)  # (B, H*W, d_model)
        
        # Positional encoding for features
        memory_pos = self.pos_encoding(feat)  # (B, H*W, d_model)
        
        # Process each image's boxes with its features
        outputs = []
        offset = 0
        for img_idx, n_boxes in enumerate(num_boxes_per_image):
            if n_boxes == 0:
                continue
            
            # Get queries for this image
            q = query[offset:offset + n_boxes]  # (n_boxes, d_model)
            q_pos = query_pos[offset:offset + n_boxes]
            
            # Get memory for this image
            mem = memory[img_idx]  # (H*W, d_model)
            mem_pos = memory_pos[img_idx]  # (H*W, d_model)
            
            # Apply cross-attention layers
            for layer in self.layers:
                q = layer(q, q_pos, mem, mem_pos)
            
            outputs.append(q)
            offset += n_boxes
        
        if len(outputs) == 0:
            return query
        
        return torch.cat(outputs, dim=0)
    
    def _predict(self, features: torch.Tensor, n: int):
        """Make predictions from query features.
        
        Args:
            features: Query features (N, d_model)
            n: Number of queries
            
        Returns:
            CubeHead-format outputs
        """
        # 2D deltas (kept for interface compatibility, can be zeros for 3D-only)
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
