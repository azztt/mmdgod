# Copyright (c) Meta Platforms, Inc. and affiliates
# Modified for transformer-based 3D detection head
"""
Transformer Decoder 3D Head

Replaces the original CubeHead with a transformer encoder architecture:
1. Pooled ROI features are projected to tokens
2. Self-attention refines the features
3. MLP regressor predicts 3D box parameters (same outputs as CubeHead)

Uses Flash Attention (PyTorch 2.0+) for efficiency.
Same interface as CubeHead: input flattened features, output per-class predictions.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from detectron2.layers import ShapeSpec
import fvcore.nn.weight_init as weight_init

from pytorch3d.transforms.rotation_conversions import _copysign
from pytorch3d.transforms import (
    rotation_6d_to_matrix, 
    euler_angles_to_matrix, 
    quaternion_to_matrix
)

from .cube_head import ROI_CUBE_HEAD_REGISTRY

# Try to import flash attention utilities
try:
    from torch.nn.functional import scaled_dot_product_attention
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


class FlashMultiheadAttention(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention using PyTorch 2.0's
    scaled_dot_product_attention (Flash Attention backend when available).
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, batch_first: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Separate Q, K, V projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.use_flash = HAS_FLASH_ATTN
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: (B, N_q, D) query tensor
            key: (B, N_k, D) key tensor
            value: (B, N_k, D) value tensor
            
        Returns:
            output: (B, N_q, D) attention output
        """
        B, N_q, D = query.shape
        N_k = key.shape[1]
        
        # Project Q, K, V
        q = self.q_proj(query).reshape(B, N_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(B, N_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Use scaled_dot_product_attention (Flash when available)
        if self.use_flash and attn_mask is None and key_padding_mask is None:
            attn_output = scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
            )
        else:
            # Fallback to standard attention
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            
            if attn_mask is not None:
                attn = attn + attn_mask
            if key_padding_mask is not None:
                attn = attn.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf')
                )
            
            attn = F.softmax(attn, dim=-1)
            if self.training and self.dropout > 0:
                attn = F.dropout(attn, p=self.dropout)
            attn_output = attn @ v
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(B, N_q, D)
        output = self.out_proj(attn_output)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """
    A single transformer encoder layer with self-attention.
    Uses Flash Attention when available.
    """
    
    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_flash_attention: bool = True,
    ):
        super().__init__()
        
        # Self-attention
        if use_flash_attention:
            self.self_attn = FlashMultiheadAttention(d_model, nhead, dropout=dropout)
        else:
            self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.use_flash = use_flash_attention
    
    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: (B, N, D) input tensor
            
        Returns:
            output: (B, N, D) processed tensor
        """
        # Self-attention
        if self.use_flash:
            attn_out = self.self_attn(src, src, src)
        else:
            attn_out, _ = self.self_attn(src, src, src)
        src = src + self.dropout(attn_out)
        src = self.norm1(src)
        
        # FFN
        src = src + self.ffn(src)
        src = self.norm2(src)
        
        return src


@ROI_CUBE_HEAD_REGISTRY.register()
class TransformerDecoder3DHead(nn.Module):
    """
    Transformer-based 3D detection head that replaces CubeHead.
    
    Same interface as CubeHead:
    - Input: Flattened ROI pooled features (N, C*H*W) 
    - Output: (box_2d_deltas, box_z, box_dims, box_pose, box_uncert)
      All per-class: box_2d_deltas=(N, num_classes, 2), etc.
    
    Architecture:
    1. Project pooled features to tokens
    2. Self-attention to refine features
    3. Per-class MLP heads for 3D predictions
    
    Uses Flash Attention (PyTorch 2.0+) for efficiency.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()

        #-------------------------------------------
        # Settings (same as CubeHead)
        #-------------------------------------------
        self.num_classes        = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.use_conf           = cfg.MODEL.ROI_CUBE_HEAD.USE_CONFIDENCE
        self.z_type             = cfg.MODEL.ROI_CUBE_HEAD.Z_TYPE
        self.pose_type          = cfg.MODEL.ROI_CUBE_HEAD.POSE_TYPE
        self.cluster_bins       = cfg.MODEL.ROI_CUBE_HEAD.CLUSTER_BINS
        
        # Transformer settings
        self.feature_dim        = getattr(cfg.MODEL.ROI_CUBE_HEAD, "FEATURE_DIM", 256)
        self.num_decoder_layers = getattr(cfg.MODEL.ROI_CUBE_HEAD, "NUM_DECODER_LAYERS", 6)
        self.num_heads          = getattr(cfg.MODEL.ROI_CUBE_HEAD, "NUM_HEADS", 8)
        self.dim_feedforward    = getattr(cfg.MODEL.ROI_CUBE_HEAD, "DIM_FEEDFORWARD", 2048)
        self.dropout            = getattr(cfg.MODEL.ROI_CUBE_HEAD, "DROPOUT", 0.1)
        self.use_flash          = getattr(cfg.MODEL.ROI_CUBE_HEAD, "USE_FLASH_ATTENTION", True) and HAS_FLASH_ATTN

        #-------------------------------------------
        # Input projection
        #-------------------------------------------
        input_dim = int(np.prod((input_shape.channels, input_shape.height, input_shape.width)))
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
        )
        
        #-------------------------------------------
        # Transformer encoder layers (self-attention)
        #-------------------------------------------
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.feature_dim,
                nhead=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                use_flash_attention=self.use_flash,
            )
            for _ in range(self.num_decoder_layers)
        ])

        #-------------------------------------------
        # 3D output heads (same outputs as CubeHead)
        #-------------------------------------------
        cluster_bins = self.cluster_bins if self.cluster_bins > 1 else 1

        # XY (2D center deltas)
        self.bbox_3D_center_deltas = nn.Linear(self.feature_dim, self.num_classes * 2)
        nn.init.normal_(self.bbox_3D_center_deltas.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_center_deltas.bias, 0)

        # Dimensions in meters (width, height, length)
        self.bbox_3D_dims = nn.Linear(self.feature_dim, self.num_classes * 3)
        nn.init.normal_(self.bbox_3D_dims.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_dims.bias, 0)

        # Pose
        if self.pose_type == '6d':
            self.bbox_3D_pose = nn.Linear(self.feature_dim, self.num_classes * 6)
        elif self.pose_type == 'quaternion':
            self.bbox_3D_pose = nn.Linear(self.feature_dim, self.num_classes * 4)
        elif self.pose_type == 'euler':
            self.bbox_3D_pose = nn.Linear(self.feature_dim, self.num_classes * 3)
        else:
            raise ValueError(f'Cuboid pose type {self.pose_type} is not recognized')
        
        nn.init.normal_(self.bbox_3D_pose.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_pose.bias, 0)

        # Z depth
        self.bbox_3D_center_depth = nn.Linear(self.feature_dim, self.num_classes * cluster_bins)
        nn.init.normal_(self.bbox_3D_center_depth.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_center_depth.bias, 0)

        # Confidence
        if self.use_conf:
            self.bbox_3D_uncertainty = nn.Linear(self.feature_dim, self.num_classes * 1)
            nn.init.normal_(self.bbox_3D_uncertainty.weight, std=0.001)
            nn.init.constant_(self.bbox_3D_uncertainty.bias, 5)

        # Class prediction head (replaces BoxHead classification)
        self.class_predictor = nn.Linear(self.feature_dim, self.num_classes + 1)  # +1 for background
        nn.init.normal_(self.class_predictor.weight, std=0.01)
        nn.init.constant_(self.class_predictor.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        Forward pass with same interface as CubeHead.
        
        Args:
            x: (N, C*H*W) flattened ROI pooled features
            
        Returns:
            box_2d_deltas: (N, num_classes, 2) 2D center offsets
            box_z: (N, num_classes, 1) or (N, cluster_bins, num_classes, 1) depth
            box_dims: (N, num_classes, 3) dimensions
            box_pose: (N, num_classes, 3, 3) rotation matrices
            box_uncert: (N, num_classes) confidence or None
        """
        n = x.shape[0]
        
        if n == 0:
            # Handle empty input
            box_2d_deltas = x.new_zeros(0, self.num_classes, 2)
            box_z = x.new_zeros(0, self.num_classes, 1)
            box_dims = x.new_zeros(0, self.num_classes, 3)
            box_pose = x.new_zeros(0, self.num_classes, 3, 3)
            box_uncert = x.new_zeros(0, self.num_classes) if self.use_conf else None
            class_logits = x.new_zeros(0, self.num_classes + 1)
            return box_2d_deltas, box_z, box_dims, box_pose, box_uncert, class_logits
        
        # Project input features
        features = self.input_projection(x)  # (N, feature_dim)
        
        # Add batch dimension for transformer: (N, feature_dim) -> (1, N, feature_dim)
        features = features.unsqueeze(0)
        
        # Pass through transformer encoder layers
        for layer in self.transformer_layers:
            features = layer(features)
        
        # Remove batch dimension: (1, N, feature_dim) -> (N, feature_dim)
        features = features.squeeze(0)
        
        # Predict 3D outputs (same as CubeHead)
        box_2d_deltas = self.bbox_3D_center_deltas(features)
        box_dims = self.bbox_3D_dims(features)
        box_pose = self.bbox_3D_pose(features)
        box_z = self.bbox_3D_center_depth(features)
        
        box_uncert = None
        if self.use_conf:
            box_uncert = self.bbox_3D_uncertainty(features).clip(0.01)

        # Process pose to rotation matrices
        if self.pose_type == '6d':
            box_pose = rotation_6d_to_matrix(box_pose.view(-1, 6))
        elif self.pose_type == 'quaternion':
            quats = box_pose.view(-1, 4)
            quats_scales = (quats * quats).sum(1)
            quats = quats / _copysign(torch.sqrt(quats_scales), quats[:, 0])[:, None]
            box_pose = quaternion_to_matrix(quats)
        elif self.pose_type == 'euler':
            box_pose = euler_angles_to_matrix(box_pose.view(-1, 3), 'XYZ')

        # Reshape outputs to match CubeHead format
        box_2d_deltas = box_2d_deltas.view(n, self.num_classes, 2)
        box_dims = box_dims.view(n, self.num_classes, 3)
        box_pose = box_pose.view(n, self.num_classes, 3, 3)

        if self.cluster_bins > 1:
            box_z = box_z.view(n, self.cluster_bins, self.num_classes, -1)
        else:
            box_z = box_z.view(n, self.num_classes, -1)
        
        # Predict class logits
        class_logits = self.class_predictor(features)  # (N, num_classes + 1)
            
        return box_2d_deltas, box_z, box_dims, box_pose, box_uncert, class_logits

class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer with self-attention and cross-attention.
    Used by HybridTransformer3DHead for learned queries.
    """
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, 
                 dropout: float = 0.1, use_flash_attention: bool = True):
        super().__init__()
        
        # Self-attention
        self.self_attn = FlashMultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Cross-attention (queries attend to image features)
        self.cross_attn = FlashMultiheadAttention(d_model, nhead, dropout=dropout)
        
        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, tgt: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tgt: (B, N_q, D) query features
            memory: (B, N_m, D) image features (from backbone)
        Returns:
            (B, N_q, D) refined query features
        """
        # Self-attention
        tgt2 = self.self_attn(tgt, tgt, tgt)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross-attention with image features
        tgt2 = self.cross_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # FFN
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


@ROI_CUBE_HEAD_REGISTRY.register()
class HybridTransformer3DHead(nn.Module):
    """
    Hybrid 3D detection head combining:
    1. RPN proposals (ROI-pooled features) - strong localization priors
    2. Learned object queries (like DETR) - can discover missed objects
    
    Architecture:
    - RPN queries: ROI pooled features → projection → tokens
    - Learned queries: learnable embeddings → cross-attention with image features
    - Both go through shared transformer decoder
    - Deduplication via NMS at inference
    
    The learned queries don't have 2D box priors, so they predict absolute
    2D centers instead of deltas. This is handled in roi_heads.py.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()

        #-------------------------------------------
        # Settings
        #-------------------------------------------
        self.num_classes        = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.use_conf           = cfg.MODEL.ROI_CUBE_HEAD.USE_CONFIDENCE
        self.z_type             = cfg.MODEL.ROI_CUBE_HEAD.Z_TYPE
        self.pose_type          = cfg.MODEL.ROI_CUBE_HEAD.POSE_TYPE
        self.cluster_bins       = cfg.MODEL.ROI_CUBE_HEAD.CLUSTER_BINS
        
        # Transformer settings
        self.feature_dim        = getattr(cfg.MODEL.ROI_CUBE_HEAD, "FEATURE_DIM", 256)
        self.num_decoder_layers = getattr(cfg.MODEL.ROI_CUBE_HEAD, "NUM_DECODER_LAYERS", 6)
        self.num_heads          = getattr(cfg.MODEL.ROI_CUBE_HEAD, "NUM_HEADS", 8)
        self.dim_feedforward    = getattr(cfg.MODEL.ROI_CUBE_HEAD, "DIM_FEEDFORWARD", 2048)
        self.dropout            = getattr(cfg.MODEL.ROI_CUBE_HEAD, "DROPOUT", 0.1)
        self.use_flash          = getattr(cfg.MODEL.ROI_CUBE_HEAD, "USE_FLASH_ATTENTION", True) and HAS_FLASH_ATTN
        
        # Hybrid settings
        self.num_learned_queries = getattr(cfg.MODEL.ROI_CUBE_HEAD, "NUM_LEARNED_QUERIES", 100)

        #-------------------------------------------
        # Input projection for RPN features
        #-------------------------------------------
        input_dim = int(np.prod((input_shape.channels, input_shape.height, input_shape.width)))
        
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
        )
        
        #-------------------------------------------
        # Learned object queries (like DETR)
        #-------------------------------------------
        self.learned_queries = nn.Embedding(self.num_learned_queries, self.feature_dim)
        self.query_pos_embed = nn.Embedding(self.num_learned_queries, self.feature_dim)
        nn.init.normal_(self.learned_queries.weight, std=0.02)
        nn.init.normal_(self.query_pos_embed.weight, std=0.02)
        
        #-------------------------------------------
        # Transformer decoder layers (self + cross attention)
        #-------------------------------------------
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=self.feature_dim,
                nhead=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                use_flash_attention=self.use_flash,
            )
            for _ in range(self.num_decoder_layers)
        ])
        
        # Self-attention layers for RPN queries (no cross-attn needed, they have ROI features)
        self.rpn_encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=self.feature_dim,
                nhead=self.num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=self.dropout,
                use_flash_attention=self.use_flash,
            )
            for _ in range(self.num_decoder_layers)
        ])

        #-------------------------------------------
        # 3D output heads
        #-------------------------------------------
        cluster_bins = self.cluster_bins if self.cluster_bins > 1 else 1

        # For RPN queries: predict deltas from 2D box
        # For learned queries: predict absolute 2D center (normalized 0-1)
        self.bbox_3D_center_deltas = nn.Linear(self.feature_dim, self.num_classes * 2)
        self.bbox_3D_center_absolute = nn.Linear(self.feature_dim, self.num_classes * 2)  # For learned queries
        nn.init.normal_(self.bbox_3D_center_deltas.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_center_deltas.bias, 0)
        nn.init.normal_(self.bbox_3D_center_absolute.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_center_absolute.bias, 0.5)  # Center of image

        # Dimensions
        self.bbox_3D_dims = nn.Linear(self.feature_dim, self.num_classes * 3)
        nn.init.normal_(self.bbox_3D_dims.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_dims.bias, 0)

        # Pose
        if self.pose_type == '6d':
            self.bbox_3D_pose = nn.Linear(self.feature_dim, self.num_classes * 6)
        elif self.pose_type == 'quaternion':
            self.bbox_3D_pose = nn.Linear(self.feature_dim, self.num_classes * 4)
        elif self.pose_type == 'euler':
            self.bbox_3D_pose = nn.Linear(self.feature_dim, self.num_classes * 3)
        else:
            raise ValueError(f'Cuboid pose type {self.pose_type} is not recognized')
        nn.init.normal_(self.bbox_3D_pose.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_pose.bias, 0)

        # Z depth
        self.bbox_3D_center_depth = nn.Linear(self.feature_dim, self.num_classes * cluster_bins)
        nn.init.normal_(self.bbox_3D_center_depth.weight, std=0.001)
        nn.init.constant_(self.bbox_3D_center_depth.bias, 0)

        # Confidence
        if self.use_conf:
            self.bbox_3D_uncertainty = nn.Linear(self.feature_dim, self.num_classes * 1)
            nn.init.normal_(self.bbox_3D_uncertainty.weight, std=0.001)
            nn.init.constant_(self.bbox_3D_uncertainty.bias, 5)

        # Class prediction
        self.class_predictor = nn.Linear(self.feature_dim, self.num_classes + 1)
        nn.init.normal_(self.class_predictor.weight, std=0.01)
        nn.init.constant_(self.class_predictor.bias, 0)
        
        # 2D box prediction for learned queries (to generate proposal boxes)
        self.bbox_2d_predictor = nn.Linear(self.feature_dim, 4)  # x1, y1, x2, y2 normalized
        nn.init.normal_(self.bbox_2d_predictor.weight, std=0.001)
        nn.init.constant_(self.bbox_2d_predictor.bias, 0)

    def forward(self, x: torch.Tensor):
        """
        Forward pass for hybrid RPN + learned queries.
        
        Learned queries cross-attend to RPN features (no need for image features).
        This way learned queries "see" the scene through RPN's ROI-pooled features.
        
        Args:
            x: (N_rpn, C*H*W) flattened ROI pooled features from RPN proposals
            
        Returns:
            box_2d_deltas: (N_total, num_classes, 2) - deltas for RPN, zeros for learned
            box_z: depth predictions
            box_dims: dimension predictions  
            box_pose: pose predictions
            box_uncert: confidence predictions
            class_logits: class predictions
            query_type: (N_total,) tensor - 0 for RPN queries, 1 for learned queries
            learned_boxes_2d: (N_learned, 4) predicted 2D boxes for learned queries (normalized)
        """
        n_rpn = x.shape[0]
        device = x.device
        n_learned = self.num_learned_queries
        
        # Handle empty RPN input
        if n_rpn == 0:
            # Still process learned queries with dummy memory
            rpn_features = x.new_zeros(1, self.feature_dim)  # Dummy token
            n_rpn_effective = 0
        else:
            # Project RPN features
            rpn_features = self.input_projection(x)  # (N_rpn, feature_dim)
            n_rpn_effective = n_rpn
        
        # Self-attention for RPN queries
        rpn_for_memory = rpn_features.unsqueeze(0)  # (1, N_rpn, feature_dim)
        for layer in self.rpn_encoder_layers:
            rpn_for_memory = layer(rpn_for_memory)
        rpn_refined = rpn_for_memory.squeeze(0)  # (N_rpn, feature_dim)
        
        # Get learned queries
        queries = self.learned_queries.weight  # (N_learned, feature_dim)
        query_pos = self.query_pos_embed.weight
        tgt = (queries + query_pos).unsqueeze(0)  # (1, N_learned, feature_dim)
        
        # Cross-attention: learned queries attend to RPN features
        memory = rpn_refined.unsqueeze(0)  # (1, N_rpn, feature_dim)
        for layer in self.decoder_layers:
            tgt = layer(tgt, memory)
        
        learned_features = tgt.squeeze(0)  # (N_learned, feature_dim)
        
        # Combine features: RPN first, then learned
        if n_rpn_effective > 0:
            all_features = torch.cat([rpn_refined, learned_features], dim=0)
            query_type = torch.cat([
                torch.zeros(n_rpn_effective, device=device, dtype=torch.long),
                torch.ones(n_learned, device=device, dtype=torch.long)
            ])
        else:
            all_features = learned_features
            query_type = torch.ones(n_learned, device=device, dtype=torch.long)
        
        n_total = all_features.shape[0]
        
        # Predict outputs (shared heads for both query types)
        box_2d_deltas = self.bbox_3D_center_deltas(all_features)
        box_dims = self.bbox_3D_dims(all_features)
        box_pose = self.bbox_3D_pose(all_features)
        box_z = self.bbox_3D_center_depth(all_features)
        class_logits = self.class_predictor(all_features)
        
        box_uncert = None
        if self.use_conf:
            box_uncert = self.bbox_3D_uncertainty(all_features).clip(0.01)
        
        # For learned queries, predict 2D boxes (normalized 0-1)
        learned_boxes_2d = torch.sigmoid(self.bbox_2d_predictor(learned_features))  # (N_learned, 4)

        # Process pose to rotation matrices
        if self.pose_type == '6d':
            box_pose = rotation_6d_to_matrix(box_pose.view(-1, 6))
        elif self.pose_type == 'quaternion':
            quats = box_pose.view(-1, 4)
            quats_scales = (quats * quats).sum(1)
            quats = quats / _copysign(torch.sqrt(quats_scales), quats[:, 0])[:, None]
            box_pose = quaternion_to_matrix(quats)
        elif self.pose_type == 'euler':
            box_pose = euler_angles_to_matrix(box_pose.view(-1, 3), 'XYZ')

        # Reshape outputs
        box_2d_deltas = box_2d_deltas.view(n_total, self.num_classes, 2)
        box_dims = box_dims.view(n_total, self.num_classes, 3)
        box_pose = box_pose.view(n_total, self.num_classes, 3, 3)

        if self.cluster_bins > 1:
            box_z = box_z.view(n_total, self.cluster_bins, self.num_classes, -1)
        else:
            box_z = box_z.view(n_total, self.num_classes, -1)
            
        return (box_2d_deltas, box_z, box_dims, box_pose, box_uncert, 
                class_logits, query_type, learned_boxes_2d)