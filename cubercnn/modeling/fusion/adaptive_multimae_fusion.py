# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Adaptive Windowed Patch Fusion Module for RGB-D feature fusion.

Inspired by CuTR/MultiMAE: Uses windowed attention for cross-modal fusion
with adaptive window sizes to handle resolution asymmetry between modalities.

Key idea: Different window sizes for RGB and Depth ensure same number of 
patches per window for joint attention, even when modalities have different
spatial resolutions.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition features into non-overlapping windows with padding if needed.
    
    Args:
        x (Tensor): Input features with shape [B, H, W, C].
        window_size (int): Window size.

    Returns:
        windows: Windows after partition [B * num_windows, window_size, window_size, C].
        (Hp, Wp): Padded height and width before partition.
    """
    B, H, W, C = x.shape

    # Pad to multiple of window_size
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    
    Hp, Wp = H + pad_h, W + pad_w

    # Reshape into windows
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    windows = windows.view(-1, window_size, window_size, C)
    
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, 
    window_size: int, 
    pad_hw: Tuple[int, int], 
    hw: Tuple[int, int]
) -> torch.Tensor:
    """Reverse window partition and remove padding.
    
    Args:
        windows (Tensor): Windows [B * num_windows, window_size, window_size, C].
        window_size (int): Window size.
        pad_hw (Tuple): Padded height and width (Hp, Wp).
        hw (Tuple): Original height and width (H, W) before padding.

    Returns:
        x: Unpartitioned features [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    num_windows_total = Hp * Wp // (window_size * window_size)
    if num_windows_total == 0:
        num_windows_total = 1  # Avoid division by zero
    B = windows.shape[0] // num_windows_total
    
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, Hp, Wp, -1)
    
    # Remove padding
    x = x[:, :H, :W, :].contiguous()
    
    return x


class CrossModalWindowAttention(nn.Module):
    """Cross-modal attention within windows.
    
    Given RGB and Depth patches from corresponding windows, performs
    bidirectional cross-attention for multi-modal fusion.
    
    Args:
        dim (int): Feature dimension.
        num_heads (int): Number of attention heads.
        qkv_bias (bool): Add bias to QKV projections. Default: True.
        attn_drop (float): Attention dropout rate. Default: 0.0.
        proj_drop (float): Projection dropout rate. Default: 0.0.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Separate QKV projections for each modality
        self.qkv_rgb = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.qkv_depth = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # Output projections
        self.proj_rgb = nn.Linear(dim, dim)
        self.proj_depth = nn.Linear(dim, dim)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(
        self,
        rgb_patches: torch.Tensor,
        depth_patches: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cross-modal attention within windows.
        
        Args:
            rgb_patches (Tensor): RGB patches [B*num_windows, N_rgb, C].
            depth_patches (Tensor): Depth patches [B*num_windows, N_depth, C].
            
        Returns:
            Tuple of:
                - Updated RGB patches [B*num_windows, N_rgb, C]
                - Updated Depth patches [B*num_windows, N_depth, C]
        """
        BW, N_rgb, C = rgb_patches.shape
        N_depth = depth_patches.shape[1]
        
        # Compute Q, K, V for RGB
        qkv_rgb = self.qkv_rgb(rgb_patches)
        qkv_rgb = qkv_rgb.reshape(BW, N_rgb, 3, self.num_heads, self.head_dim)
        qkv_rgb = qkv_rgb.permute(2, 0, 3, 1, 4)  # (3, BW, heads, N, head_dim)
        q_rgb, k_rgb, v_rgb = qkv_rgb.unbind(0)
        
        # Compute Q, K, V for Depth  
        qkv_depth = self.qkv_depth(depth_patches)
        qkv_depth = qkv_depth.reshape(BW, N_depth, 3, self.num_heads, self.head_dim)
        qkv_depth = qkv_depth.permute(2, 0, 3, 1, 4)
        q_depth, k_depth, v_depth = qkv_depth.unbind(0)
        
        # Concatenate K, V from both modalities for joint attention
        k_joint = torch.cat([k_rgb, k_depth], dim=2)  # (BW, heads, N_rgb+N_depth, head_dim)
        v_joint = torch.cat([v_rgb, v_depth], dim=2)
        
        # RGB queries attend to joint K, V
        attn_rgb = (q_rgb @ k_joint.transpose(-2, -1)) * self.scale
        attn_rgb = attn_rgb.softmax(dim=-1)
        attn_rgb = self.attn_drop(attn_rgb)
        out_rgb = (attn_rgb @ v_joint).transpose(1, 2).reshape(BW, N_rgb, C)
        out_rgb = self.proj_drop(self.proj_rgb(out_rgb))
        
        # Depth queries attend to joint K, V
        attn_depth = (q_depth @ k_joint.transpose(-2, -1)) * self.scale
        attn_depth = attn_depth.softmax(dim=-1)
        attn_depth = self.attn_drop(attn_depth)
        out_depth = (attn_depth @ v_joint).transpose(1, 2).reshape(BW, N_depth, C)
        out_depth = self.proj_drop(self.proj_depth(out_depth))
        
        return out_rgb, out_depth


class WindowedCrossModalBlock(nn.Module):
    """Transformer block with windowed cross-modal attention.
    
    Applies:
    1. Window partition (different sizes per modality for resolution matching)
    2. Cross-modal attention within windows
    3. Window unpartition
    4. Residual connection + MLP
    
    Args:
        dim (int): Feature dimension.
        num_heads (int): Number of attention heads.
        rgb_window_size (int): Window size for RGB features.
        depth_window_size (int): Window size for Depth features.
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.0.
        qkv_bias (bool): Add bias to QKV. Default: True.
        drop (float): Dropout rate. Default: 0.0.
        attn_drop (float): Attention dropout. Default: 0.0.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        rgb_window_size: int = 7,
        depth_window_size: int = 7,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
    ):
        super().__init__()
        
        self.dim = dim
        self.rgb_window_size = rgb_window_size
        self.depth_window_size = depth_window_size
        
        # Layer norms
        self.norm1_rgb = nn.LayerNorm(dim)
        self.norm1_depth = nn.LayerNorm(dim)
        
        # Cross-modal window attention
        self.attn = CrossModalWindowAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        
        # MLP
        self.norm2_rgb = nn.LayerNorm(dim)
        self.norm2_depth = nn.LayerNorm(dim)
        
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp_rgb = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )
        self.mlp_depth = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )
        
    def forward(
        self,
        rgb_features: torch.Tensor,
        depth_features: torch.Tensor,
        rgb_spatial: Tuple[int, int],
        depth_spatial: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with windowed cross-modal attention.
        
        Args:
            rgb_features (Tensor): RGB features [B, N_rgb, C].
            depth_features (Tensor): Depth features [B, N_depth, C].
            rgb_spatial (Tuple): RGB spatial shape (H, W).
            depth_spatial (Tuple): Depth spatial shape (H, W).
            
        Returns:
            Tuple of updated (rgb_features, depth_features).
        """
        B = rgb_features.shape[0]
        H_rgb, W_rgb = rgb_spatial
        H_depth, W_depth = depth_spatial
        
        # Reshape to spatial layout: (B, H, W, C)
        rgb_2d = rgb_features.view(B, H_rgb, W_rgb, -1)
        depth_2d = depth_features.view(B, H_depth, W_depth, -1)
        
        # Store shortcuts
        shortcut_rgb = rgb_2d
        shortcut_depth = depth_2d
        
        # Layer norm
        rgb_2d = self.norm1_rgb(rgb_2d)
        depth_2d = self.norm1_depth(depth_2d)
        
        # Window partition
        rgb_windows, rgb_pad_hw = window_partition(rgb_2d, self.rgb_window_size)
        depth_windows, depth_pad_hw = window_partition(depth_2d, self.depth_window_size)
        
        # Flatten window spatial dims: (B*nW, ws*ws, C)
        rgb_windows = rgb_windows.view(-1, self.rgb_window_size * self.rgb_window_size, self.dim)
        depth_windows = depth_windows.view(-1, self.depth_window_size * self.depth_window_size, self.dim)
        
        # Check window count matches
        num_windows_rgb = rgb_windows.shape[0] // B
        num_windows_depth = depth_windows.shape[0] // B
        
        if num_windows_rgb != num_windows_depth:
            # Fallback: if window counts don't match, use global cross-attention
            rgb_windows = rgb_features
            depth_windows = depth_features
            rgb_out, depth_out = self.attn(rgb_windows, depth_windows)
            rgb_out = rgb_out.view(B, H_rgb, W_rgb, -1)
            depth_out = depth_out.view(B, H_depth, W_depth, -1)
        else:
            # Cross-modal attention within windows
            rgb_out, depth_out = self.attn(rgb_windows, depth_windows)
            
            # Reshape back to window format
            rgb_out = rgb_out.view(-1, self.rgb_window_size, self.rgb_window_size, self.dim)
            depth_out = depth_out.view(-1, self.depth_window_size, self.depth_window_size, self.dim)
            
            # Window unpartition
            rgb_out = window_unpartition(rgb_out, self.rgb_window_size, rgb_pad_hw, (H_rgb, W_rgb))
            depth_out = window_unpartition(depth_out, self.depth_window_size, depth_pad_hw, (H_depth, W_depth))
        
        # Residual + MLP
        rgb_out = shortcut_rgb + rgb_out
        depth_out = shortcut_depth + depth_out
        
        rgb_out = rgb_out + self.mlp_rgb(self.norm2_rgb(rgb_out))
        depth_out = depth_out + self.mlp_depth(self.norm2_depth(depth_out))
        
        # Flatten back to sequence: (B, N, C)
        rgb_out = rgb_out.view(B, -1, self.dim)
        depth_out = depth_out.view(B, -1, self.dim)
        
        return rgb_out, depth_out


class AdaptiveMultiMAEFusion(nn.Module):
    """Adaptive MultiMAE-style Windowed Patch Fusion for RGB-D features.
    
    Inspired by CuTR/MultiMAE: Uses adaptive window sizes to handle
    resolution asymmetry between RGB and Depth modalities. Windows are
    sized such that both modalities produce the same number of patches
    per window, enabling direct cross-modal attention.
    
    This is particularly useful for FPN features where RGB and depth
    may have different spatial resolutions at each level.
    
    Architecture:
        1. Compute optimal window sizes based on modality resolutions
        2. Apply N windowed cross-modal attention blocks
        3. Final projection and fusion
    
    Args:
        feature_dim (int): Feature dimension. Default: 256.
        num_heads (int): Number of attention heads. Default: 8.
        num_blocks (int): Number of cross-modal blocks. Default: 2.
        base_window_size (int): Base window size. Default: 7.
        mlp_ratio (float): MLP hidden dim ratio. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.1.
        fusion_mode (str): How to combine modalities at the end.
            'concat_proj': Concatenate and project (default).
            'add': Simple addition.
            'gated': Learned gating.
    """
    
    VALID_WINDOW_SIZES = [4, 6, 7, 8, 12, 14, 16]
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        num_blocks: int = 2,
        base_window_size: int = 7,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        fusion_mode: str = 'concat_proj',
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.base_window_size = base_window_size
        self.fusion_mode = fusion_mode
        
        # Cross-modal attention blocks
        self.blocks = nn.ModuleList([
            WindowedCrossModalBlock(
                dim=feature_dim,
                num_heads=num_heads,
                rgb_window_size=base_window_size,
                depth_window_size=base_window_size,
                mlp_ratio=mlp_ratio,
                drop=dropout,
                attn_drop=dropout,
            )
            for _ in range(num_blocks)
        ])
        
        # Final fusion layer
        if fusion_mode == 'concat_proj':
            self.fusion_proj = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim * 2, feature_dim),
            )
        elif fusion_mode == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(feature_dim * 2, feature_dim),
                nn.Sigmoid(),
            )
        elif fusion_mode == 'add':
            self.fusion_proj = nn.Identity()
        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
        
        self.final_norm = nn.LayerNorm(feature_dim)
    
    def _find_best_window_sizes(
        self,
        rgb_spatial: Tuple[int, int],
        depth_spatial: Tuple[int, int],
    ) -> Tuple[int, int]:
        """Find window sizes that produce matching window counts."""
        H_rgb, W_rgb = rgb_spatial
        H_depth, W_depth = depth_spatial
        
        def num_windows(H, W, ws):
            Hp = H + (ws - H % ws) % ws
            Wp = W + (ws - W % ws) % ws
            return (Hp // ws) * (Wp // ws)
        
        best_rgb_ws = self.base_window_size
        best_depth_ws = self.base_window_size
        min_diff = float('inf')
        
        for rgb_ws in self.VALID_WINDOW_SIZES:
            nw_rgb = num_windows(H_rgb, W_rgb, rgb_ws)
            for depth_ws in self.VALID_WINDOW_SIZES:
                nw_depth = num_windows(H_depth, W_depth, depth_ws)
                diff = abs(nw_rgb - nw_depth)
                if diff < min_diff:
                    min_diff = diff
                    best_rgb_ws = rgb_ws
                    best_depth_ws = depth_ws
                if diff == 0:
                    return rgb_ws, depth_ws
        
        return best_rgb_ws, best_depth_ws
    
    def forward(
        self,
        rgb_features: torch.Tensor,
        depth_features: torch.Tensor,
        rgb_spatial: Optional[Tuple[int, int]] = None,
        depth_spatial: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Forward pass for adaptive windowed patch fusion.
        
        Args:
            rgb_features (Tensor): RGB features [B, C, H, W] or [B, N_rgb, C].
            depth_features (Tensor): Depth features [B, C, H, W] or [B, N_depth, C].
            rgb_spatial (Tuple, optional): RGB spatial shape (H, W).
            depth_spatial (Tuple, optional): Depth spatial shape (H, W).
                
        Returns:
            Tensor: Fused features [B, C, H, W] (matches RGB shape).
        """
        # Handle 4D input (feature maps from FPN)
        is_4d = rgb_features.dim() == 4
        if is_4d:
            B, C, H_rgb, W_rgb = rgb_features.shape
            _, _, H_depth, W_depth = depth_features.shape
            rgb_spatial = (H_rgb, W_rgb)
            depth_spatial = (H_depth, W_depth)
            # Reshape to (B, N, C)
            rgb_features = rgb_features.flatten(2).transpose(1, 2)
            depth_features = depth_features.flatten(2).transpose(1, 2)
        
        B, N_rgb, C = rgb_features.shape
        N_depth = depth_features.shape[1]
        
        # Infer spatial shapes if not provided
        if rgb_spatial is None:
            H_rgb = W_rgb = int(math.sqrt(N_rgb))
            rgb_spatial = (H_rgb, W_rgb)
        if depth_spatial is None:
            H_depth = W_depth = int(math.sqrt(N_depth))
            depth_spatial = (H_depth, W_depth)
        
        # Find optimal window sizes
        rgb_ws, depth_ws = self._find_best_window_sizes(rgb_spatial, depth_spatial)
        
        # Update block window sizes dynamically
        for block in self.blocks:
            block.rgb_window_size = rgb_ws
            block.depth_window_size = depth_ws
        
        # Apply cross-modal blocks
        rgb_out = rgb_features
        depth_out = depth_features
        
        for block in self.blocks:
            rgb_out, depth_out = block(rgb_out, depth_out, rgb_spatial, depth_spatial)
        
        # Align depth to RGB resolution if different
        H_rgb, W_rgb = rgb_spatial
        H_depth, W_depth = depth_spatial
        
        if (H_rgb, W_rgb) != (H_depth, W_depth):
            depth_2d = depth_out.view(B, H_depth, W_depth, C).permute(0, 3, 1, 2)
            depth_2d = F.interpolate(depth_2d, size=(H_rgb, W_rgb), mode='bilinear', align_corners=False)
            depth_out = depth_2d.permute(0, 2, 3, 1).reshape(B, -1, C)
        
        # Final fusion
        if self.fusion_mode == 'concat_proj':
            fused = torch.cat([rgb_out, depth_out], dim=-1)
            fused = self.fusion_proj(fused)
        elif self.fusion_mode == 'gated':
            concat = torch.cat([rgb_out, depth_out], dim=-1)
            gate = self.gate(concat)
            fused = gate * rgb_out + (1 - gate) * depth_out
        elif self.fusion_mode == 'add':
            fused = rgb_out + depth_out
        
        fused = self.final_norm(fused)
        
        # Reshape back to 4D if needed
        if is_4d:
            fused = fused.transpose(1, 2).reshape(B, C, H_rgb, W_rgb)
        
        return fused


class AdaptiveMultiMAEFusionFPN(nn.Module):
    """AdaptiveMultiMAE Fusion applied at each FPN level.
    
    Uses adaptive windowed attention fusion optimized for each FPN scale.
    
    Args:
        in_channels: Number of input channels (FPN output channels)
        num_heads: Number of attention heads
        num_blocks: Number of cross-modal blocks per level
        base_window_size: Base window size
        dropout: Dropout rate
        fusion_mode: Fusion strategy
    """
    
    def __init__(
        self,
        in_channels: int = 256,
        num_heads: int = 8,
        num_blocks: int = 2,
        base_window_size: int = 7,
        dropout: float = 0.1,
        fusion_mode: str = 'concat_proj',
    ):
        super().__init__()
        
        # One fusion module per FPN level
        self.fusion_p2 = AdaptiveMultiMAEFusion(
            in_channels, num_heads, num_blocks, base_window_size, dropout=dropout, fusion_mode=fusion_mode
        )
        self.fusion_p3 = AdaptiveMultiMAEFusion(
            in_channels, num_heads, num_blocks, base_window_size, dropout=dropout, fusion_mode=fusion_mode
        )
        self.fusion_p4 = AdaptiveMultiMAEFusion(
            in_channels, num_heads, num_blocks, base_window_size, dropout=dropout, fusion_mode=fusion_mode
        )
        self.fusion_p5 = AdaptiveMultiMAEFusion(
            in_channels, num_heads, num_blocks, base_window_size, dropout=dropout, fusion_mode=fusion_mode
        )
    
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
