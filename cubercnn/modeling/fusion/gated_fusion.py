# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Gated Fusion Module for RGB-D feature fusion.

This module implements a gating mechanism that learns to combine RGB and
depth features using cross-attention followed by a learned gating network.
"""
import torch
import torch.nn as nn
from typing import Dict, Optional


class GatedFusion(nn.Module):
    """Gated fusion for RGB and Depth features using cross-attention and gating.

    This module:
    1. Performs bidirectional cross-attention (RGB queries Depth, Depth queries RGB)
    2. Uses a learned gating network to combine the attended features
    3. Outputs a single fused feature representation

    Args:
        feature_dim (int): The dimensionality of input features. Default: 256.
        num_heads (int): Number of attention heads. Default: 8.
        dim_feedforward (int): Hidden dimension in the gating network. Default: 1024.
        dropout (float): Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # Cross-attention: RGB queries Depth
        self.cross_attn_rgb_q = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention: Depth queries RGB
        self.cross_attn_depth_q = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Gating mechanism using 1D convolutions
        self.gating_net = nn.Sequential(
            nn.Conv1d(
                in_channels=feature_dim * 2,
                out_channels=dim_feedforward,
                kernel_size=1,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(
                in_channels=dim_feedforward,
                out_channels=feature_dim,
                kernel_size=1,
            ),
            nn.Sigmoid(),
        )

        # Layer normalization for stability
        self.norm_rgb = nn.LayerNorm(feature_dim)
        self.norm_depth = nn.LayerNorm(feature_dim)

    def forward(
        self,
        rgb_features: torch.Tensor,
        depth_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for gated fusion.

        Args:
            rgb_features (torch.Tensor): RGB features of shape (B, N, C) or (B, C, H, W).
            depth_features (torch.Tensor): Depth features of shape (B, N, C) or (B, C, H, W).

        Returns:
            torch.Tensor: Fused features of same shape as input.
        """
        # Handle 4D input (feature maps)
        is_4d = rgb_features.dim() == 4
        if is_4d:
            B, C, H, W = rgb_features.shape
            # Reshape to (B, N, C) where N = H*W
            rgb_features = rgb_features.flatten(2).transpose(1, 2)  # (B, H*W, C)
            depth_features = depth_features.flatten(2).transpose(1, 2)  # (B, H*W, C)
        
        # 1. Bidirectional cross-attention
        # RGB queries Depth
        rgb_attended, _ = self.cross_attn_rgb_q(
            query=rgb_features,
            key=depth_features,
            value=depth_features,
        )
        # Residual connection and normalization
        rgb_updated = self.norm_rgb(rgb_features + rgb_attended)

        # Depth queries RGB
        depth_attended, _ = self.cross_attn_depth_q(
            query=depth_features,
            key=rgb_features,
            value=rgb_features,
        )
        # Residual connection and normalization
        depth_updated = self.norm_depth(depth_features + depth_attended)

        # 2. Concatenate for gating: (B, N, C*2)
        concatenated = torch.cat([rgb_updated, depth_updated], dim=2)

        # 3. Gating network expects (B, C*2, N), so permute
        concatenated_permuted = concatenated.permute(0, 2, 1)

        # 4. Compute gate values
        gate = self.gating_net(concatenated_permuted)  # (B, C, N)
        gate = gate.permute(0, 2, 1)  # (B, N, C)

        # 5. Apply gating to produce fused features
        fused = gate * rgb_updated + (1 - gate) * depth_updated
        
        # Reshape back to 4D if needed
        if is_4d:
            fused = fused.transpose(1, 2).reshape(B, C, H, W)

        return fused


class GatedFusionFPN(nn.Module):
    """Gated fusion applied to FPN feature maps.
    
    Applies gated fusion at each FPN level.
    
    Args:
        in_channels: Number of input channels (same for all levels)
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # One gated fusion module per FPN level
        self.fusion_p2 = GatedFusion(in_channels, num_heads, dropout=dropout)
        self.fusion_p3 = GatedFusion(in_channels, num_heads, dropout=dropout)
        self.fusion_p4 = GatedFusion(in_channels, num_heads, dropout=dropout)
        self.fusion_p5 = GatedFusion(in_channels, num_heads, dropout=dropout)
    
    def forward(
        self,
        rgb_features: Dict[str, torch.Tensor],
        depth_features: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Fuse RGB and depth features at each FPN level.
        
        Args:
            rgb_features: Dict of RGB features {p2, p3, p4, p5}
            depth_features: Dict of depth features {p2, p3, p4, p5}
            
        Returns:
            Dict of fused features {p2, p3, p4, p5}
        """
        return {
            'p2': self.fusion_p2(rgb_features['p2'], depth_features['p2']),
            'p3': self.fusion_p3(rgb_features['p3'], depth_features['p3']),
            'p4': self.fusion_p4(rgb_features['p4'], depth_features['p4']),
            'p5': self.fusion_p5(rgb_features['p5'], depth_features['p5']),
        }
