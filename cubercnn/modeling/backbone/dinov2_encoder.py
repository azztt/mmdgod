# Copyright (c) Meta Platforms, Inc. and affiliates
# DINOv2 Backbone for RGB-D 3D Object Detection
"""
DINOv2 Backbone Encoder for RGB-D processing.

Uses the DINOv2 Vision Transformer as backbone with:
- Complete freezing for RGB (domain-invariant features)
- Partial freezing for depth (geometry adaptation)
- Multi-scale feature extraction from intermediate layers

DINOv2 outputs a single-scale feature, so we extract features from
multiple intermediate layers and project them to create FPN-like outputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone


class DINOv2Encoder(nn.Module):
    """DINOv2 encoder for a single modality (RGB or Depth).
    
    Wraps a DINOv2 ViT and provides multi-scale features by extracting
    from intermediate transformer blocks.
    
    Args:
        model_name: DINOv2 model variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        in_channels: Number of input channels (3 for RGB, 1 for depth)
        freeze_all: If True, freeze entire model
        num_frozen_blocks: Number of transformer blocks to freeze (from start)
        output_layers: Which transformer layers to extract features from
        pretrained: Whether to load pretrained weights
    """
    
    # Layer indices for multi-scale features (DINOv2-L has 24 blocks)
    DEFAULT_OUTPUT_LAYERS = {
        'dinov2_vits14': [2, 5, 8, 11],      # ViT-S: 12 blocks
        'dinov2_vitb14': [2, 5, 8, 11],      # ViT-B: 12 blocks  
        'dinov2_vitl14': [4, 11, 17, 23],    # ViT-L: 24 blocks
        'dinov2_vitg14': [9, 19, 29, 39],    # ViT-G: 40 blocks
    }
    
    EMBED_DIMS = {
        'dinov2_vits14': 384,
        'dinov2_vitb14': 768,
        'dinov2_vitl14': 1024,
        'dinov2_vitg14': 1536,
    }
    
    def __init__(
        self, 
        model_name: str = 'dinov2_vitl14',
        in_channels: int = 3,
        freeze_all: bool = False,
        num_frozen_blocks: int = 0,
        output_layers: Optional[List[int]] = None,
        pretrained: bool = True,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.in_channels = in_channels
        self.freeze_all = freeze_all
        self.num_frozen_blocks = num_frozen_blocks
        
        # Load DINOv2 model
        if pretrained:
            self.dino = torch.hub.load('facebookresearch/dinov2', model_name)
        else:
            # Load architecture without pretrained weights
            self.dino = torch.hub.load('facebookresearch/dinov2', model_name, pretrained=False)
        
        self.embed_dim = self.EMBED_DIMS[model_name]
        self.patch_size = 14  # DINOv2 uses 14x14 patches
        
        # Determine output layers
        if output_layers is None:
            self.output_layers = self.DEFAULT_OUTPUT_LAYERS[model_name]
        else:
            self.output_layers = output_layers
        
        # Modify patch embedding if input channels != 3
        if in_channels != 3:
            old_patch_embed = self.dino.patch_embed.proj
            new_patch_embed = nn.Conv2d(
                in_channels, 
                self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            )
            # Initialize with averaged pretrained weights
            with torch.no_grad():
                if pretrained:
                    new_patch_embed.weight.data = old_patch_embed.weight.data.mean(dim=1, keepdim=True)
                    if in_channels > 1:
                        new_patch_embed.weight.data = new_patch_embed.weight.data.expand(-1, in_channels, -1, -1).clone()
                    new_patch_embed.bias.data = old_patch_embed.bias.data.clone()
            self.dino.patch_embed.proj = new_patch_embed
        
        # Apply freezing
        self._apply_freezing()
        
        # Output feature info (strides relative to 14x14 patches)
        # We'll create "virtual" strides for FPN compatibility
        self._out_feature_strides = {
            'layer0': 4,   # Upsampled 
            'layer1': 8,   
            'layer2': 16,  
            'layer3': 32,  # Original patch stride (14) rounded to 32
        }
        self._out_feature_channels = {
            'layer0': self.embed_dim,
            'layer1': self.embed_dim,
            'layer2': self.embed_dim,
            'layer3': self.embed_dim,
        }
    
    def _apply_freezing(self):
        """Apply freezing based on settings."""
        if self.freeze_all:
            for param in self.dino.parameters():
                param.requires_grad = False
            return
        
        # Freeze patch embedding
        if self.num_frozen_blocks > 0:
            for param in self.dino.patch_embed.parameters():
                param.requires_grad = False
            for param in self.dino.cls_token.parameters() if hasattr(self.dino, 'cls_token') else []:
                param.requires_grad = False
        
        # Freeze transformer blocks
        for i, block in enumerate(self.dino.blocks):
            if i < self.num_frozen_blocks:
                for param in block.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from DINOv2.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Dict of features at different "scales" {layer0, layer1, layer2, layer3}
        """
        B, C, H, W = x.shape
        
        # Patch embed
        x = self.dino.patch_embed(x)
        
        # Get grid size
        h = H // self.patch_size
        w = W // self.patch_size
        
        # Prepend CLS token if present
        if hasattr(self.dino, 'cls_token') and self.dino.cls_token is not None:
            cls_token = self.dino.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            has_cls = True
        else:
            has_cls = False
        
        # Add position embedding
        x = x + self.dino.pos_embed
        
        # Extract features from specified layers
        features = {}
        layer_names = ['layer0', 'layer1', 'layer2', 'layer3']
        
        for i, block in enumerate(self.dino.blocks):
            x = block(x)
            
            if i in self.output_layers:
                layer_idx = self.output_layers.index(i)
                layer_name = layer_names[layer_idx]
                
                # Remove CLS token and reshape to spatial
                if has_cls:
                    feat = x[:, 1:, :]  # (B, h*w, D)
                else:
                    feat = x  # (B, h*w, D)
                
                # Reshape to spatial format: (B, D, h, w)
                feat = feat.transpose(1, 2).reshape(B, self.embed_dim, h, w)
                features[layer_name] = feat
        
        return features
    
    @property
    def out_channels(self) -> Dict[str, int]:
        return self._out_feature_channels
    
    @property
    def out_feature_strides(self) -> Dict[str, int]:
        return self._out_feature_strides


class DINOv2DualEncoder(Backbone):
    """Dual-stream DINOv2 backbone for RGB-D feature extraction.
    
    Uses separate DINOv2 encoders for RGB and depth, then fuses
    their features for FPN input.
    
    Args:
        cfg: Detectron2 config
        model_name: DINOv2 model variant
        rgb_freeze_all: Freeze entire RGB encoder
        depth_num_frozen_blocks: Number of blocks to freeze in depth encoder
        fusion_type: How to fuse features ('concat', 'add', 'gated')
        out_channels: Output channel dimension after fusion
    """
    
    def __init__(
        self,
        cfg=None,
        model_name: str = 'dinov2_vitl14',
        rgb_freeze_all: bool = True,
        depth_num_frozen_blocks: int = 12,
        fusion_type: str = 'concat',
        out_channels: int = 256,
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.out_channels = out_channels
        
        # RGB encoder (completely frozen for domain invariance)
        self.rgb_encoder = DINOv2Encoder(
            model_name=model_name,
            in_channels=3,
            freeze_all=rgb_freeze_all,
            pretrained=True,
        )
        
        # Depth encoder (partially frozen for geometry adaptation)
        self.depth_encoder = DINOv2Encoder(
            model_name=model_name,
            in_channels=1,
            freeze_all=False,
            num_frozen_blocks=depth_num_frozen_blocks,
            pretrained=True,
        )
        
        self.embed_dim = self.rgb_encoder.embed_dim
        
        # Feature dimension reduction and fusion
        layer_names = ['layer0', 'layer1', 'layer2', 'layer3']
        
        if fusion_type == 'concat':
            # Concatenate then project
            self.fusion_projections = nn.ModuleDict()
            for layer in layer_names:
                in_ch = self.embed_dim * 2
                self.fusion_projections[layer] = nn.Sequential(
                    nn.Conv2d(in_ch, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
        elif fusion_type == 'add':
            # Add features (same dimension)
            self.fusion_projections = nn.ModuleDict()
            for layer in layer_names:
                self.fusion_projections[layer] = nn.Sequential(
                    nn.Conv2d(self.embed_dim, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
        elif fusion_type == 'gated':
            # Gated fusion with learned weights
            self.fusion_gates = nn.ModuleDict()
            self.fusion_projections = nn.ModuleDict()
            for layer in layer_names:
                # Gate network
                self.fusion_gates[layer] = nn.Sequential(
                    nn.Conv2d(self.embed_dim * 2, self.embed_dim, kernel_size=1),
                    nn.Sigmoid(),
                )
                # Output projection
                self.fusion_projections[layer] = nn.Sequential(
                    nn.Conv2d(self.embed_dim, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Multi-scale feature adaptation (upsample/downsample to match FPN strides)
        self.scale_adapters = nn.ModuleDict()
        target_strides = [4, 8, 16, 32]
        
        for i, (layer, stride) in enumerate(zip(layer_names, target_strides)):
            # All DINOv2 features are at patch_size=14 stride
            # We need to resize to match target strides
            self.scale_adapters[layer] = nn.Identity()  # Will resize in forward
        
        # Output feature info for FPN
        self._out_features = ['p2', 'p3', 'p4', 'p5']
        self._out_feature_strides = {'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32}
        self._out_feature_channels = {k: out_channels for k in self._out_features}
    
    @property
    def size_divisibility(self) -> int:
        """Input size should be divisible by patch size * some factor."""
        return 14 * 4  # 56
    
    def forward(self, x: torch.Tensor, depth: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass with RGB and depth inputs.
        
        Args:
            x: RGB tensor (B, 3, H, W)
            depth: Depth tensor (B, 1, H, W). If None, uses zero depth.
            
        Returns:
            Dict of fused features {p2, p3, p4, p5} ready for RPN
        """
        B, _, H, W = x.shape
        
        # Handle missing depth
        if depth is None:
            depth = torch.zeros(B, 1, H, W, device=x.device, dtype=x.dtype)
        
        # Extract features from both encoders
        rgb_feats = self.rgb_encoder(x)
        depth_feats = self.depth_encoder(depth)
        
        # Fuse features
        fused_feats = {}
        layer_names = ['layer0', 'layer1', 'layer2', 'layer3']
        output_names = ['p2', 'p3', 'p4', 'p5']
        target_sizes = [
            (H // 4, W // 4),
            (H // 8, W // 8),
            (H // 16, W // 16),
            (H // 32, W // 32),
        ]
        
        for layer, out_name, target_size in zip(layer_names, output_names, target_sizes):
            r_feat = rgb_feats[layer]
            d_feat = depth_feats[layer]
            
            if self.fusion_type == 'concat':
                concat = torch.cat([r_feat, d_feat], dim=1)
                fused = self.fusion_projections[layer](concat)
            elif self.fusion_type == 'add':
                added = r_feat + d_feat
                fused = self.fusion_projections[layer](added)
            elif self.fusion_type == 'gated':
                gate_input = torch.cat([r_feat, d_feat], dim=1)
                gate = self.fusion_gates[layer](gate_input)
                gated = gate * r_feat + (1 - gate) * d_feat
                fused = self.fusion_projections[layer](gated)
            
            # Resize to target stride
            if fused.shape[2:] != target_size:
                fused = F.interpolate(fused, size=target_size, mode='bilinear', align_corners=False)
            
            fused_feats[out_name] = fused
        
        return fused_feats
    
    def output_shape(self) -> Dict[str, ShapeSpec]:
        """Return output shape specifications for FPN/RPN."""
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


def build_dinov2_dual_encoder(cfg, priors=None) -> DINOv2DualEncoder:
    """Build DINOv2 dual encoder backbone from config.
    
    Args:
        cfg: Detectron2 config with MODEL.BACKBONE section
        priors: Optional priors (unused for backbone)
        
    Returns:
        DINOv2DualEncoder backbone
    """
    # Get config values with defaults
    backbone_cfg = getattr(cfg.MODEL, 'BACKBONE', {})
    
    model_name = getattr(backbone_cfg, 'DINOV2_MODEL', 'dinov2_vitl14')
    rgb_freeze_all = getattr(backbone_cfg, 'RGB_FREEZE_ALL', True)
    depth_num_frozen_blocks = getattr(backbone_cfg, 'DEPTH_NUM_FROZEN_BLOCKS', 12)
    fusion_type = getattr(backbone_cfg, 'FUSION_TYPE', 'concat')
    out_channels = getattr(backbone_cfg, 'OUT_CHANNELS', 256)
    
    return DINOv2DualEncoder(
        cfg=cfg,
        model_name=model_name,
        rgb_freeze_all=rgb_freeze_all,
        depth_num_frozen_blocks=depth_num_frozen_blocks,
        fusion_type=fusion_type,
        out_channels=out_channels,
    )
