# Copyright (c) Meta Platforms, Inc. and affiliates
# Modified for RGB-D dual encoder support
"""
Dual Encoder Backbone for RGB-D 3D Object Detection.

This module provides a dual-stream encoder that processes RGB and depth
modalities separately, then fuses them for domain-generalized detection.

Architecture:
    RGB Image → Frozen/Partially Frozen ResNet → RGB Features
    Depth Map → Partially Frozen ResNet → Depth Features
    RGB + Depth Features → Fusion Module → FPN → RPN → ROI Heads

Key Features:
- Separate backbones for RGB and depth (can use different freeze strategies)
- Supports multiple fusion strategies (concat, gated, attention)
- Compatible with Cube R-CNN's FPN and CubeHead
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torchvision import models
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.backbone.build import BACKBONE_REGISTRY
from detectron2.modeling.backbone.fpn import FPN, LastLevelMaxPool


class ResNetEncoder(nn.Module):
    """ResNet encoder for a single modality (RGB or Depth).
    
    Wraps a torchvision ResNet and provides multi-scale features
    suitable for FPN.
    
    Args:
        depth: ResNet depth (18, 34, 50, 101)
        pretrained: Whether to load ImageNet pretrained weights
        in_channels: Number of input channels (3 for RGB, 1 for depth)
        freeze_at: Freeze stages up to this index (0=none, 1=stem, 2=res2, etc.)
    """
    
    def __init__(
        self, 
        depth: int = 50,
        pretrained: bool = True,
        in_channels: int = 3,
        freeze_at: int = 0,
    ):
        super().__init__()
        
        # Load pretrained ResNet
        if depth == 18:
            base = models.resnet18(pretrained=pretrained)
            self._out_channels = {'res2': 64, 'res3': 128, 'res4': 256, 'res5': 512}
        elif depth == 34:
            base = models.resnet34(pretrained=pretrained)
            self._out_channels = {'res2': 64, 'res3': 128, 'res4': 256, 'res5': 512}
        elif depth == 50:
            base = models.resnet50(pretrained=pretrained)
            self._out_channels = {'res2': 256, 'res3': 512, 'res4': 1024, 'res5': 2048}
        elif depth == 101:
            base = models.resnet101(pretrained=pretrained)
            self._out_channels = {'res2': 256, 'res3': 512, 'res4': 1024, 'res5': 2048}
        else:
            raise ValueError(f"Unsupported ResNet depth: {depth}")
        
        # Modify first conv if input channels != 3
        if in_channels != 3:
            # Initialize new conv1 with averaged pretrained weights
            old_conv1 = base.conv1
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Initialize with mean of pretrained RGB weights
            with torch.no_grad():
                if pretrained:
                    self.conv1.weight.data = old_conv1.weight.data.mean(dim=1, keepdim=True)
                    if in_channels > 1:
                        self.conv1.weight.data = self.conv1.weight.data.expand(-1, in_channels, -1, -1)
        else:
            self.conv1 = base.conv1
        
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1  # res2
        self.layer2 = base.layer2  # res3
        self.layer3 = base.layer3  # res4
        self.layer4 = base.layer4  # res5
        
        self._out_feature_strides = {'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32}
        self.freeze_at = freeze_at
        
        # Apply freezing
        self._freeze_stages()
    
    def _freeze_stages(self):
        """Freeze stages based on freeze_at setting."""
        if self.freeze_at >= 1:
            # Freeze stem (conv1, bn1)
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False
        
        freeze_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
        for i, layer in enumerate(freeze_layers):
            if self.freeze_at >= i + 2:  # freeze_at=2 freezes layer1, etc.
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Dict of features at different scales {res2, res3, res4, res5}
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        res2 = self.layer1(x)
        res3 = self.layer2(res2)
        res4 = self.layer3(res3)
        res5 = self.layer4(res4)
        
        return {
            'res2': res2,
            'res3': res3,
            'res4': res4,
            'res5': res5,
        }
    
    @property
    def out_channels(self) -> Dict[str, int]:
        return self._out_channels
    
    @property
    def out_feature_strides(self) -> Dict[str, int]:
        return self._out_feature_strides


class DualEncoderBackbone(Backbone):
    """Dual-stream backbone for RGB-D feature extraction.
    
    Uses separate ResNet encoders for RGB and depth, then fuses
    their features before passing to FPN.
    
    Args:
        cfg: Detectron2 config
        input_shape: Input shape specification
        rgb_depth: ResNet depth for RGB encoder (default: 50)
        depth_depth: ResNet depth for depth encoder (default: 50)
        rgb_freeze_at: Freeze RGB encoder stages (default: 4 = freeze all)
        depth_freeze_at: Freeze depth encoder stages (default: 2)
        fusion_type: How to fuse features ('concat', 'add', 'gated')
        pretrained: Whether to use pretrained weights
    """
    
    def __init__(
        self,
        cfg,
        input_shape: ShapeSpec,
        rgb_depth: int = 50,
        depth_depth: int = 50,
        rgb_freeze_at: int = 4,
        depth_freeze_at: int = 2,
        fusion_type: str = 'concat',
        pretrained: bool = True,
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # RGB encoder (more frozen for domain-invariant features)
        self.rgb_encoder = ResNetEncoder(
            depth=rgb_depth,
            pretrained=pretrained,
            in_channels=3,
            freeze_at=rgb_freeze_at,
        )
        
        # Depth encoder (less frozen to adapt to depth modality)
        self.depth_encoder = ResNetEncoder(
            depth=depth_depth,
            pretrained=pretrained,
            in_channels=1,
            freeze_at=depth_freeze_at,
        )
        
        # Compute output channels based on fusion type
        rgb_channels = self.rgb_encoder.out_channels
        depth_channels = self.depth_encoder.out_channels
        
        if fusion_type == 'concat':
            # Concatenate features, then project back
            self._out_feature_channels = {}
            self.fusion_projections = nn.ModuleDict()
            
            for key in ['res2', 'res3', 'res4', 'res5']:
                in_ch = rgb_channels[key] + depth_channels[key]
                out_ch = rgb_channels[key]  # Match original dimension
                self.fusion_projections[key] = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
                self._out_feature_channels[key] = out_ch
        
        elif fusion_type == 'add':
            # Add features (requires same dimensions)
            self._out_feature_channels = rgb_channels
            self.fusion_projections = None
            
            # Project depth features to match RGB if dimensions differ
            if rgb_channels != depth_channels:
                self.depth_projections = nn.ModuleDict()
                for key in ['res2', 'res3', 'res4', 'res5']:
                    self.depth_projections[key] = nn.Conv2d(
                        depth_channels[key], rgb_channels[key], kernel_size=1
                    )
            else:
                self.depth_projections = None
        
        elif fusion_type == 'gated':
            # Gated fusion with learned weights
            self._out_feature_channels = {}
            self.fusion_gates = nn.ModuleDict()
            
            for key in ['res2', 'res3', 'res4', 'res5']:
                rgb_ch = rgb_channels[key]
                depth_ch = depth_channels[key]
                out_ch = rgb_ch
                
                # Gate network predicts per-pixel fusion weights
                self.fusion_gates[key] = nn.Sequential(
                    nn.Conv2d(rgb_ch + depth_ch, out_ch, kernel_size=1),
                    nn.Sigmoid(),
                )
                
                # Project depth to match RGB dimension if needed
                if rgb_ch != depth_ch:
                    self.fusion_gates[f'{key}_depth_proj'] = nn.Conv2d(
                        depth_ch, rgb_ch, kernel_size=1
                    )
                
                self._out_feature_channels[key] = out_ch
        
        elif fusion_type == 'windowed':
            # Adaptive MultiMAE-style windowed attention fusion
            from ..fusion import AdaptiveMultiMAEFusion
            
            self._out_feature_channels = {}
            self.windowed_fusion = nn.ModuleDict()
            
            # Get windowed fusion config
            windowed_cfg = getattr(cfg.MODEL, 'FUSION', {})
            windowed_params = getattr(windowed_cfg, 'WINDOWED', {})
            
            num_heads = getattr(windowed_params, 'NUM_HEADS', 8)
            num_blocks = getattr(windowed_params, 'NUM_BLOCKS', 2)
            base_window_size = getattr(windowed_params, 'BASE_WINDOW_SIZE', 7)
            dropout = getattr(windowed_params, 'DROPOUT', 0.1)
            fusion_mode = getattr(windowed_params, 'FUSION_MODE', 'concat_proj')
            
            for key in ['res2', 'res3', 'res4', 'res5']:
                rgb_ch = rgb_channels[key]
                depth_ch = depth_channels[key]
                
                # Project depth to same dimension as RGB first
                if rgb_ch != depth_ch:
                    self.windowed_fusion[f'{key}_depth_proj'] = nn.Conv2d(
                        depth_ch, rgb_ch, kernel_size=1
                    )
                
                # Windowed fusion operates on same-dimension features
                self.windowed_fusion[key] = AdaptiveMultiMAEFusion(
                    feature_dim=rgb_ch,
                    num_heads=min(num_heads, rgb_ch // 32),  # Ensure heads divide dim
                    num_blocks=num_blocks,
                    base_window_size=base_window_size,
                    dropout=dropout,
                    fusion_mode=fusion_mode,
                )
                self._out_feature_channels[key] = rgb_ch
        
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Output feature strides (same as ResNet)
        self._out_feature_strides = {'res2': 4, 'res3': 8, 'res4': 16, 'res5': 32}
        self._out_features = ['res2', 'res3', 'res4', 'res5']
    
    def forward(self, x: torch.Tensor, depth: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass with RGB and depth inputs.
        
        Args:
            x: RGB tensor (B, 3, H, W)
            depth: Depth tensor (B, 1, H, W). If None, uses zero depth.
            
        Returns:
            Dict of fused features {res2, res3, res4, res5}
        """
        # Handle case where depth is not provided
        if depth is None:
            depth = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], 
                              device=x.device, dtype=x.dtype)
        
        # Extract features from both encoders
        rgb_feats = self.rgb_encoder(x)
        depth_feats = self.depth_encoder(depth)
        
        # Fuse features
        fused_feats = {}
        
        if self.fusion_type == 'concat':
            for key in self._out_features:
                concat = torch.cat([rgb_feats[key], depth_feats[key]], dim=1)
                fused_feats[key] = self.fusion_projections[key](concat)
        
        elif self.fusion_type == 'add':
            for key in self._out_features:
                d_feat = depth_feats[key]
                if self.depth_projections is not None:
                    d_feat = self.depth_projections[key](d_feat)
                fused_feats[key] = rgb_feats[key] + d_feat
        
        elif self.fusion_type == 'gated':
            for key in self._out_features:
                r_feat = rgb_feats[key]
                d_feat = depth_feats[key]
                
                # Project depth if needed
                if f'{key}_depth_proj' in self.fusion_gates:
                    d_feat = self.fusion_gates[f'{key}_depth_proj'](d_feat)
                
                # Compute gate
                gate_input = torch.cat([r_feat, d_feat], dim=1)
                gate = self.fusion_gates[key](gate_input)
                
                # Apply gated fusion
                fused_feats[key] = gate * r_feat + (1 - gate) * d_feat
        
        elif self.fusion_type == 'windowed':
            for key in self._out_features:
                r_feat = rgb_feats[key]
                d_feat = depth_feats[key]
                
                # Project depth to same dimension if needed
                if f'{key}_depth_proj' in self.windowed_fusion:
                    d_feat = self.windowed_fusion[f'{key}_depth_proj'](d_feat)
                
                # Apply windowed attention fusion
                fused_feats[key] = self.windowed_fusion[key](r_feat, d_feat)
        
        return fused_feats
    
    def output_shape(self):
        """Return output shape specification for each feature map."""
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


class DualEncoderFPN(Backbone):
    """Complete dual encoder with FPN for Cube R-CNN.
    
    This combines the DualEncoderBackbone with an FPN to produce
    multi-scale features ready for the RPN and ROI heads.
    
    Args:
        cfg: Detectron2 config
        input_shape: Input shape specification
    """
    
    def __init__(self, cfg, input_shape: ShapeSpec, priors=None):
        super().__init__()
        
        # Get config values with defaults
        rgb_depth = getattr(cfg.MODEL, 'RGB_RESNET_DEPTH', 50)
        depth_resnet_depth = getattr(cfg.MODEL, 'DEPTH_RESNET_DEPTH', 50)
        rgb_freeze = getattr(cfg.MODEL, 'RGB_FREEZE_AT', 4)
        depth_freeze = getattr(cfg.MODEL, 'DEPTH_FREEZE_AT', 2)
        fusion_type = getattr(cfg.MODEL, 'FUSION_TYPE', 'concat')
        imagenet_pretrain = cfg.MODEL.WEIGHTS_PRETRAIN + cfg.MODEL.WEIGHTS == ''
        
        # Build dual encoder backbone
        self.bottom_up = DualEncoderBackbone(
            cfg=cfg,
            input_shape=input_shape,
            rgb_depth=rgb_depth,
            depth_depth=depth_resnet_depth,
            rgb_freeze_at=rgb_freeze,
            depth_freeze_at=depth_freeze,
            fusion_type=fusion_type,
            pretrained=imagenet_pretrain,
        )
        
        # Build FPN on top
        in_features = cfg.MODEL.FPN.IN_FEATURES
        out_channels = cfg.MODEL.FPN.OUT_CHANNELS
        
        self.fpn = FPN(
            bottom_up=self.bottom_up,
            in_features=in_features,
            out_channels=out_channels,
            norm=cfg.MODEL.FPN.NORM,
            top_block=LastLevelMaxPool(),
            fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
        )
        
        # Copy output specifications from FPN
        self._out_features = self.fpn._out_features
        self._out_feature_channels = {
            k: out_channels for k in self._out_features
        }
        self._out_feature_strides = self.fpn._out_feature_strides
    
    def forward(self, images: torch.Tensor, depth: torch.Tensor = None):
        """Forward pass.
        
        Note: For compatibility with Cube R-CNN, this expects the depth
        to be passed separately or extracted from the input if concatenated.
        
        Args:
            images: Either RGB only (B, 3, H, W) or RGB+D concatenated (B, 4, H, W)
            depth: Optional depth tensor (B, 1, H, W)
            
        Returns:
            Dict of FPN features {p2, p3, p4, p5, p6}
        """
        # Handle concatenated input
        if images.shape[1] == 4 and depth is None:
            depth = images[:, 3:4, :, :]
            images = images[:, :3, :, :]
        
        # Get bottom-up features with fusion
        bottom_up_features = self.bottom_up(images, depth)
        
        # Pass through FPN
        # Note: FPN expects the bottom_up module's forward to be called internally
        # We need to override this behavior
        
        # Build FPN features manually
        results = []
        in_features = self.fpn.in_features
        
        prev_features = None
        for i, f in enumerate(reversed(in_features)):
            lateral_conv = self.fpn.lateral_convs[len(in_features) - 1 - i]
            output_conv = self.fpn.output_convs[len(in_features) - 1 - i]
            
            features = bottom_up_features[f]
            lateral_features = lateral_conv(features)
            
            if prev_features is not None:
                top_down = F.interpolate(prev_features, size=lateral_features.shape[-2:], mode="nearest")
                prev_features = lateral_features + top_down
            else:
                prev_features = lateral_features
            
            results.insert(0, output_conv(prev_features))
        
        # Add top block (max pool) features
        last_feature = results[-1]
        for block in self.fpn.top_block:
            last_feature = block(last_feature)
            results.append(last_feature)
        
        # Build output dict
        out = {}
        for i, f in enumerate(self.fpn._out_features):
            out[f] = results[i]
        
        return out
    
    def output_shape(self):
        """Return output shape specification."""
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels.get(name, self.fpn._out_channels),
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }


@BACKBONE_REGISTRY.register()
def build_dual_encoder_fpn_backbone(cfg, input_shape: ShapeSpec, priors=None):
    """Build dual encoder FPN backbone for RGB-D input.
    
    Args:
        cfg: Detectron2 config
        input_shape: Input shape specification
        priors: Optional priors (not used but kept for API compatibility)
        
    Returns:
        DualEncoderFPN backbone
    """
    return DualEncoderFPN(cfg, input_shape, priors)


# Also register a simpler version that just uses concat without FPN integration issues
@BACKBONE_REGISTRY.register()
def build_simple_dual_encoder_backbone(cfg, input_shape: ShapeSpec, priors=None):
    """Build simple dual encoder backbone (without FPN complications).
    
    This version is simpler and uses the original FPN class properly.
    
    Args:
        cfg: Detectron2 config  
        input_shape: Input shape specification
        priors: Optional priors
        
    Returns:
        Backbone with dual encoder and FPN
    """
    # Get config values (support both old FREEZE_AT and new FROZEN_STAGES keys)
    rgb_depth = getattr(cfg.MODEL, 'RGB_RESNET_DEPTH', cfg.MODEL.RESNETS.DEPTH)
    depth_resnet_depth = getattr(cfg.MODEL, 'DEPTH_RESNET_DEPTH', cfg.MODEL.RESNETS.DEPTH)
    
    # New config keys (dgmmod style): RGB_FROZEN_STAGES, DEPTH_FROZEN_STAGES
    # Old config keys: RGB_FREEZE_AT, DEPTH_FREEZE_AT
    rgb_frozen_stages = getattr(cfg.MODEL, 'RGB_FROZEN_STAGES', 
                                getattr(cfg.MODEL, 'RGB_FREEZE_AT', 4))
    depth_frozen_stages = getattr(cfg.MODEL, 'DEPTH_FROZEN_STAGES',
                                  getattr(cfg.MODEL, 'DEPTH_FREEZE_AT', 2))
    
    fusion_type = getattr(cfg.MODEL, 'FUSION_TYPE', 'concat')
    imagenet_pretrain = cfg.MODEL.WEIGHTS_PRETRAIN + cfg.MODEL.WEIGHTS == ''
    
    # Build dual encoder backbone
    bottom_up = DualEncoderBackbone(
        cfg=cfg,
        input_shape=input_shape,
        rgb_depth=rgb_depth,
        depth_depth=depth_resnet_depth,
        rgb_freeze_at=rgb_frozen_stages,
        depth_freeze_at=depth_frozen_stages,
        fusion_type=fusion_type,
        pretrained=imagenet_pretrain,
    )
    
    # Build FPN
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    
    return backbone
