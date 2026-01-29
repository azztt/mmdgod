# Copyright (c) Meta Platforms, Inc. and affiliates
# Auxiliary heads for self-supervised learning
"""
Auxiliary Heads for Domain Generalization

1. DepthCompletionHead: Self-supervised depth densification
   - Input: Sparse depth (randomly masked original depth)
   - Output: Dense depth reconstruction
   - Loss: L1/L2 against original dense depth at masked locations

2. Future: Cross-modal consistency, etc.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from detectron2.config import configurable


class DepthSparsifier(nn.Module):
    """
    On-the-fly depth sparsification module.
    
    Randomly masks out depth pixels to create sparse depth input.
    Used during training only.
    """
    
    def __init__(self, density: float = 0.1, min_density: float = 0.05, max_density: float = 0.2):
        """
        Args:
            density: Base probability of keeping a pixel (0.1 = 10% of pixels kept)
            min_density: Minimum density for random variation
            max_density: Maximum density for random variation
        """
        super().__init__()
        self.density = density
        self.min_density = min_density
        self.max_density = max_density
    
    def forward(self, depth: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sparsify depth map by random masking.
        
        Args:
            depth: (B, 1, H, W) dense depth map
            
        Returns:
            sparse_depth: (B, 1, H, W) sparsified depth
            valid_mask: (B, 1, H, W) mask of valid (non-zero) original pixels
        """
        if not self.training:
            # During eval, return original depth
            valid_mask = (depth > 0).float()
            return depth, valid_mask
        
        # Random density per sample for augmentation
        B = depth.shape[0]
        densities = torch.rand(B, 1, 1, 1, device=depth.device)
        densities = self.min_density + densities * (self.max_density - self.min_density)
        
        # Create random mask
        mask = (torch.rand_like(depth) < densities).float()
        
        # Also respect original valid pixels
        valid_mask = (depth > 0).float()
        
        # Sparse depth = original * random_mask * valid_mask
        sparse_depth = depth * mask * valid_mask
        
        return sparse_depth, valid_mask


class DepthCompletionHead(nn.Module):
    """
    Self-supervised depth completion head.
    
    Takes sparse depth features and predicts dense depth reconstruction.
    Trained with L1 loss against original dense depth at masked locations.
    
    Architecture:
    1. Encoder: Process sparse depth features from backbone
    2. Decoder: Upsample to original resolution
    """
    
    @configurable
    def __init__(
        self,
        *,
        in_channels: int = 256,
        feature_dim: int = 256,
        num_decoder_layers: int = 4,
        use_skip_connections: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels from FPN
            feature_dim: Hidden dimension
            num_decoder_layers: Number of upsampling layers
            use_skip_connections: Whether to use skip connections from encoder
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.use_skip_connections = use_skip_connections
        
        # Encoder: 1x1 conv to project FPN features
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, 1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        # Decoder: Progressive upsampling
        # Each layer: ConvTranspose2d (2x upsample) + Conv + BN + ReLU
        decoder_layers = []
        current_dim = feature_dim
        
        for i in range(num_decoder_layers):
            out_dim = max(current_dim // 2, 32)
            decoder_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(current_dim, out_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_dim),
                    nn.ReLU(inplace=True),
                )
            )
            current_dim = out_dim
        
        self.decoder = nn.ModuleList(decoder_layers)
        
        # Output head: predict depth
        self.output_head = nn.Sequential(
            nn.Conv2d(current_dim, current_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(current_dim // 2, 1, kernel_size=1),
            nn.ReLU(inplace=True),  # Depth is positive
        )
        
        self._init_weights()
    
    @classmethod
    def from_config(cls, cfg):
        return {
            "in_channels": cfg.MODEL.FPN.OUT_CHANNELS,
            "feature_dim": cfg.MODEL.DEPTH_COMPLETION.get("FEATURE_DIM", 256),
            "num_decoder_layers": cfg.MODEL.DEPTH_COMPLETION.get("NUM_DECODER_LAYERS", 4),
            "use_skip_connections": cfg.MODEL.DEPTH_COMPLETION.get("USE_SKIP_CONNECTIONS", True),
        }
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        features: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        """
        Forward pass for depth completion.
        
        Args:
            features: (B, C, H, W) FPN features (typically from p2 or fused)
            target_size: (H, W) target output size
            
        Returns:
            pred_depth: (B, 1, H, W) predicted dense depth
        """
        # Encode
        x = self.encoder(features)
        
        # Decode with progressive upsampling
        for layer in self.decoder:
            x = layer(x)
        
        # Output
        pred_depth = self.output_head(x)
        
        # Resize to target size if needed
        if pred_depth.shape[-2:] != target_size:
            pred_depth = F.interpolate(
                pred_depth,
                size=target_size,
                mode='bilinear',
                align_corners=False,
            )
        
        return pred_depth


class DepthCompletionLoss(nn.Module):
    """
    Self-supervised loss for depth completion.
    
    Computes loss only at locations where:
    1. Original depth was valid (non-zero)
    2. The pixel was masked out during sparsification
    """
    
    def __init__(self, loss_type: str = "l1", loss_weight: float = 1.0):
        """
        Args:
            loss_type: "l1" or "l2" or "berhu"
            loss_weight: Weight for the loss
        """
        super().__init__()
        self.loss_type = loss_type
        self.loss_weight = loss_weight
    
    def forward(
        self,
        pred_depth: torch.Tensor,
        target_depth: torch.Tensor,
        sparse_depth: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute depth completion loss.
        
        Args:
            pred_depth: (B, 1, H, W) predicted depth
            target_depth: (B, 1, H, W) original dense depth (ground truth)
            sparse_depth: (B, 1, H, W) sparse input depth
            
        Returns:
            loss: Scalar loss value
        """
        # Valid mask: where original depth is valid
        valid_mask = (target_depth > 0).float()
        
        # Masked mask: where we masked out the depth (sparse == 0 but target > 0)
        masked_locations = (sparse_depth == 0) & (target_depth > 0)
        supervision_mask = masked_locations.float()
        
        # If no supervision locations, return zero loss
        if supervision_mask.sum() == 0:
            return torch.tensor(0.0, device=pred_depth.device)
        
        # Compute error
        error = pred_depth - target_depth
        
        if self.loss_type == "l1":
            loss = torch.abs(error)
        elif self.loss_type == "l2":
            loss = error ** 2
        elif self.loss_type == "berhu":
            # Reverse Huber loss (better for depth)
            abs_error = torch.abs(error)
            c = 0.2 * abs_error.max()
            loss = torch.where(
                abs_error <= c,
                abs_error,
                (abs_error ** 2 + c ** 2) / (2 * c)
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply supervision mask and average
        loss = (loss * supervision_mask).sum() / supervision_mask.sum().clamp(min=1)
        
        return self.loss_weight * loss


class AuxiliaryBranch(nn.Module):
    """
    Combined auxiliary branch containing depth completion and other tasks.
    
    This module wraps all auxiliary tasks and handles:
    1. Depth sparsification
    2. Depth completion prediction
    3. Loss computation
    """
    
    @configurable
    def __init__(
        self,
        *,
        depth_completion_enabled: bool = True,
        depth_sparsity: float = 0.1,
        sparsity_min: float = 0.05,
        sparsity_max: float = 0.2,
        in_channels: int = 256,
        feature_dim: int = 256,
        loss_type: str = "l1",
        loss_weight: float = 1.0,
    ):
        super().__init__()
        
        self.depth_completion_enabled = depth_completion_enabled
        
        if depth_completion_enabled:
            self.sparsifier = DepthSparsifier(
                density=depth_sparsity,
                min_density=sparsity_min,
                max_density=sparsity_max,
            )
            
            self.depth_completion = DepthCompletionHead(
                in_channels=in_channels,
                feature_dim=feature_dim,
            )
            
            self.depth_loss = DepthCompletionLoss(
                loss_type=loss_type,
                loss_weight=loss_weight,
            )
    
    @classmethod
    def from_config(cls, cfg):
        dc_cfg = cfg.MODEL.get("DEPTH_COMPLETION", {})
        return {
            "depth_completion_enabled": dc_cfg.get("ENABLED", False),
            "depth_sparsity": dc_cfg.get("SPARSITY", 0.1),
            "sparsity_min": dc_cfg.get("SPARSITY_MIN", 0.05),
            "sparsity_max": dc_cfg.get("SPARSITY_MAX", 0.2),
            "in_channels": cfg.MODEL.FPN.OUT_CHANNELS,
            "feature_dim": dc_cfg.get("FEATURE_DIM", 256),
            "loss_type": dc_cfg.get("LOSS_TYPE", "l1"),
            "loss_weight": dc_cfg.get("LOSS_WEIGHT", 1.0),
        }
    
    def forward(
        self,
        depth_images: torch.Tensor,
        fpn_features: Dict[str, torch.Tensor],
        feature_key: str = "p2",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for auxiliary tasks.
        
        Args:
            depth_images: (B, 1, H, W) original dense depth
            fpn_features: Dict of FPN features
            feature_key: Which FPN level to use for depth completion
            
        Returns:
            Dict containing:
            - sparse_depth: (B, 1, H, W) sparsified depth input
            - pred_depth: (B, 1, H, W) predicted dense depth
            - loss_depth_completion: Scalar loss (if training)
        """
        outputs = {}
        
        if not self.depth_completion_enabled:
            return outputs
        
        # Get target size from input depth
        target_size = depth_images.shape[-2:]
        
        # Sparsify depth
        sparse_depth, valid_mask = self.sparsifier(depth_images)
        outputs["sparse_depth"] = sparse_depth
        outputs["valid_mask"] = valid_mask
        
        # Get features for depth completion
        features = fpn_features[feature_key]
        
        # Predict dense depth
        pred_depth = self.depth_completion(features, target_size)
        outputs["pred_depth"] = pred_depth
        
        # Compute loss (only during training)
        if self.training:
            loss = self.depth_loss(pred_depth, depth_images, sparse_depth)
            outputs["loss_depth_completion"] = loss
        
        return outputs
