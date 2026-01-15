# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Depth-Aware Attention Fusion (DAAF) module.

This module fuses RGB and Depth features using geometry-aware attention masking.
Key innovation: Attention is masked based on 3D geometric proximity using depth
values. Patches that are far apart in 3D space cannot attend to each other,
preventing spurious feature mixing across depth discontinuities.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class DepthAwareAttentionFusion(nn.Module):
    """Depth-Aware Attention Fusion (DAAF) for RGB-D features.

    This module uses depth information to create geometry-aware attention masks,
    preventing attention across depth discontinuities. It implements:
    1. Depth-masked self-attention for each modality
    2. Depth-masked cross-attention for fusion
    3. Final MLP fusion layer

    Args:
        feature_dim (int): Input feature dimension. Default: 256.
        num_heads (int): Number of attention heads. Default: 8.
        depth_threshold (float): Maximum depth difference (in meters) for
            patches to attend to each other. Default: 0.5.
        ffn_ratio (float): Ratio for FFN hidden dimension. Default: 2.0.
        dropout (float): Dropout probability. Default: 0.1.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        depth_threshold: float = 0.5,
        ffn_ratio: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.depth_threshold = depth_threshold

        # Self-attention for RGB
        self.rgb_self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Self-attention for Depth
        self.depth_self_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention: RGB queries Depth
        self.rgb_cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Cross-attention: Depth queries RGB
        self.depth_cross_attn = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer normalization
        self.norm_rgb_sa = nn.LayerNorm(feature_dim)
        self.norm_depth_sa = nn.LayerNorm(feature_dim)
        self.norm_rgb_ca = nn.LayerNorm(feature_dim)
        self.norm_depth_ca = nn.LayerNorm(feature_dim)

        # Feedforward networks
        ffn_hidden = int(feature_dim * ffn_ratio)
        self.ffn_rgb = nn.Sequential(
            nn.Linear(feature_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, feature_dim),
            nn.Dropout(dropout),
        )
        self.ffn_depth = nn.Sequential(
            nn.Linear(feature_dim, ffn_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, feature_dim),
            nn.Dropout(dropout),
        )

        # Final fusion MLP
        self.final_fusion = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
        )

    def _compute_depth_mask(
        self,
        depth_map: torch.Tensor,
        spatial_shape: tuple,
    ) -> torch.Tensor:
        """Compute depth-aware attention mask.

        Args:
            depth_map (torch.Tensor): Depth map (B, 1, H_orig, W_orig) in meters.
                Already normalized to [-1, 1], so un-normalize first.
            spatial_shape (tuple): (H, W) of the feature map.

        Returns:
            torch.Tensor: Attention mask (B, num_heads, N, N) where -inf = mask.
        """
        B = depth_map.shape[0]
        H, W = spatial_shape
        N = H * W

        # Pool depth to feature map size
        # Un-normalize depth from [-1, 1] to [0, max_depth] (assume 8m max)
        depth_unnorm = (depth_map + 1) / 2 * 8.0  # Approximately
        
        patch_depths = F.adaptive_avg_pool2d(
            depth_unnorm, (H, W)
        )  # (B, 1, H, W)
        patch_depths = patch_depths.flatten(2).transpose(1, 2)  # (B, N, 1)

        # Compute pairwise depth differences
        depth_diff = torch.abs(
            patch_depths.unsqueeze(2) - patch_depths.unsqueeze(1)
        ).squeeze(-1)  # (B, N, N)

        # Create mask: 0 for close patches (attend), -inf for distant patches
        attn_mask = torch.where(
            depth_diff < self.depth_threshold,
            torch.zeros_like(depth_diff),
            torch.full_like(depth_diff, float('-inf')),
        )
        
        # Expand for all heads: (B, num_heads, N, N)
        attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        return attn_mask

    def forward(
        self,
        rgb_features: torch.Tensor,
        depth_features: torch.Tensor,
        depth_map: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with depth-aware attention masking.

        Args:
            rgb_features (torch.Tensor): RGB features (B, C, H, W) or (B, N, C).
            depth_features (torch.Tensor): Depth features (B, C, H, W) or (B, N, C).
            depth_map (torch.Tensor, optional): Full-resolution depth map
                (B, 1, H, W). If None, uses standard attention without masking.

        Returns:
            torch.Tensor: Fused features of same shape as input.
        """
        # Handle 4D input (feature maps)
        is_4d = rgb_features.dim() == 4
        if is_4d:
            B, C, H, W = rgb_features.shape
            spatial_shape = (H, W)
            rgb_features = rgb_features.flatten(2).transpose(1, 2)  # (B, N, C)
            depth_features = depth_features.flatten(2).transpose(1, 2)  # (B, N, C)
        else:
            B, N, C = rgb_features.shape
            # Estimate spatial shape as square
            H = W = int(math.sqrt(N))
            spatial_shape = (H, W)

        # Compute depth-aware attention mask if depth map is provided
        if depth_map is not None:
            if depth_map.dim() == 3:
                depth_map = depth_map.unsqueeze(1)
            
            # For MultiheadAttention with batch_first=True, 
            # attn_mask shape should be (B*num_heads, N, N) or (N, N)
            attn_mask = self._compute_depth_mask(depth_map, spatial_shape)
            # Reshape for PyTorch MHA: (B*num_heads, N, N)
            attn_mask = attn_mask.reshape(B * self.num_heads, N, N)
        else:
            attn_mask = None

        # 1. Depth-masked self-attention for RGB
        rgb_sa, _ = self.rgb_self_attn(
            query=rgb_features,
            key=rgb_features,
            value=rgb_features,
            attn_mask=attn_mask,
        )
        rgb_features = self.norm_rgb_sa(rgb_features + rgb_sa)

        # 2. Depth-masked self-attention for Depth
        depth_sa, _ = self.depth_self_attn(
            query=depth_features,
            key=depth_features,
            value=depth_features,
            attn_mask=attn_mask,
        )
        depth_features = self.norm_depth_sa(depth_features + depth_sa)

        # 3. Depth-masked cross-attention: RGB queries Depth
        rgb_ca, _ = self.rgb_cross_attn(
            query=rgb_features,
            key=depth_features,
            value=depth_features,
            attn_mask=attn_mask,
        )
        rgb_features = self.norm_rgb_ca(rgb_features + rgb_ca)
        rgb_features = rgb_features + self.ffn_rgb(rgb_features)

        # 4. Depth-masked cross-attention: Depth queries RGB
        depth_ca, _ = self.depth_cross_attn(
            query=depth_features,
            key=rgb_features,
            value=rgb_features,
            attn_mask=attn_mask,
        )
        depth_features = self.norm_depth_ca(depth_features + depth_ca)
        depth_features = depth_features + self.ffn_depth(depth_features)

        # 5. Final fusion
        concatenated = torch.cat([rgb_features, depth_features], dim=2)  # (B, N, 2C)
        fused = self.final_fusion(concatenated)  # (B, N, C)

        # Reshape back to 4D if needed
        if is_4d:
            fused = fused.transpose(1, 2).reshape(B, C, H, W)

        return fused


class DepthAwareAttentionFusionFPN(nn.Module):
    """DAAF applied to FPN feature maps.
    
    Applies depth-aware attention fusion at each FPN level.
    
    Args:
        in_channels: Number of input channels
        num_heads: Number of attention heads
        depth_threshold: Max depth difference for attention
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_heads: int = 8,
        depth_threshold: float = 0.5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # One DAAF module per FPN level
        self.fusion_p2 = DepthAwareAttentionFusion(
            in_channels, num_heads, depth_threshold, dropout=dropout
        )
        self.fusion_p3 = DepthAwareAttentionFusion(
            in_channels, num_heads, depth_threshold, dropout=dropout
        )
        self.fusion_p4 = DepthAwareAttentionFusion(
            in_channels, num_heads, depth_threshold, dropout=dropout
        )
        self.fusion_p5 = DepthAwareAttentionFusion(
            in_channels, num_heads, depth_threshold, dropout=dropout
        )
    
    def forward(
        self,
        rgb_features: Dict[str, torch.Tensor],
        depth_features: Dict[str, torch.Tensor],
        depth_map: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Fuse RGB and depth features at each FPN level.
        
        Args:
            rgb_features: Dict of RGB features {p2, p3, p4, p5}
            depth_features: Dict of depth features {p2, p3, p4, p5}
            depth_map: Optional depth map for geometry-aware masking
            
        Returns:
            Dict of fused features {p2, p3, p4, p5}
        """
        return {
            'p2': self.fusion_p2(rgb_features['p2'], depth_features['p2'], depth_map),
            'p3': self.fusion_p3(rgb_features['p3'], depth_features['p3'], depth_map),
            'p4': self.fusion_p4(rgb_features['p4'], depth_features['p4'], depth_map),
            'p5': self.fusion_p5(rgb_features['p5'], depth_features['p5'], depth_map),
        }
