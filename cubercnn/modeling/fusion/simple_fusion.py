# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Simple fusion modules for RGB-D features.

These provide basic fusion strategies that can be used as baselines
or when computational efficiency is important.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ConcatFusion(nn.Module):
    """Simple concatenation fusion with projection.
    
    Concatenates RGB and depth features along channel dimension,
    then projects back to original dimension.
    
    Args:
        in_channels: Number of input channels per modality
        out_channels: Number of output channels (default: same as in_channels)
        reduction: Channel reduction factor for intermediate layer
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = None,
        reduction: int = 2,
    ):
        super().__init__()
        
        out_channels = out_channels or in_channels
        mid_channels = in_channels * 2 // reduction
        
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(
        self,
        rgb_features: torch.Tensor,
        depth_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse RGB and depth features via concatenation.
        
        Args:
            rgb_features: RGB features (B, C, H, W)
            depth_features: Depth features (B, C, H, W)
            
        Returns:
            Fused features (B, C, H, W)
        """
        concat = torch.cat([rgb_features, depth_features], dim=1)
        return self.fusion(concat)


class AddFusion(nn.Module):
    """Element-wise addition fusion with optional projection.
    
    Args:
        in_channels: Number of input channels
        use_projection: Whether to project before adding
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        use_projection: bool = True,
    ):
        super().__init__()
        
        self.use_projection = use_projection
        
        if use_projection:
            self.rgb_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
            self.depth_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(
        self,
        rgb_features: torch.Tensor,
        depth_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse RGB and depth features via addition.
        
        Args:
            rgb_features: RGB features (B, C, H, W)
            depth_features: Depth features (B, C, H, W)
            
        Returns:
            Fused features (B, C, H, W)
        """
        if self.use_projection:
            rgb_features = self.rgb_proj(rgb_features)
            depth_features = self.depth_proj(depth_features)
        
        return rgb_features + depth_features


class AttentionFusion(nn.Module):
    """Simple channel attention fusion.
    
    Uses channel attention to compute fusion weights for RGB and depth.
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio for attention
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        reduction: int = 16,
    ):
        super().__init__()
        
        mid_channels = max(in_channels // reduction, 32)
        
        # Shared MLP for channel attention
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels * 2, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, 2),  # 2 weights for RGB and depth
            nn.Softmax(dim=1),
        )
    
    def forward(
        self,
        rgb_features: torch.Tensor,
        depth_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse RGB and depth features via channel attention.
        
        Args:
            rgb_features: RGB features (B, C, H, W)
            depth_features: Depth features (B, C, H, W)
            
        Returns:
            Fused features (B, C, H, W)
        """
        B, C, H, W = rgb_features.shape
        
        # Compute attention weights
        concat = torch.cat([rgb_features, depth_features], dim=1)  # (B, 2C, H, W)
        weights = self.attention(concat)  # (B, 2)
        
        # Apply weights
        rgb_weight = weights[:, 0:1, None, None]  # (B, 1, 1, 1)
        depth_weight = weights[:, 1:2, None, None]  # (B, 1, 1, 1)
        
        return rgb_weight * rgb_features + depth_weight * depth_features


class MultiscaleFusion(nn.Module):
    """Apply fusion at multiple FPN levels.
    
    Wraps a single-scale fusion module and applies it at each FPN level.
    
    Args:
        fusion_type: Type of fusion ('concat', 'add', 'attention')
        in_channels: Number of input channels
        **fusion_kwargs: Additional arguments for fusion module
    """
    
    def __init__(
        self,
        fusion_type: str = 'concat',
        in_channels: int = 256,
        **fusion_kwargs,
    ):
        super().__init__()
        
        fusion_cls = {
            'concat': ConcatFusion,
            'add': AddFusion,
            'attention': AttentionFusion,
        }[fusion_type]
        
        # Create one fusion module per FPN level
        self.fusion_p2 = fusion_cls(in_channels, **fusion_kwargs)
        self.fusion_p3 = fusion_cls(in_channels, **fusion_kwargs)
        self.fusion_p4 = fusion_cls(in_channels, **fusion_kwargs)
        self.fusion_p5 = fusion_cls(in_channels, **fusion_kwargs)
    
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
