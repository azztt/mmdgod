# Copyright (c) Meta Platforms, Inc. and affiliates
# Vision Transformer Backbone for RGB-D 3D Object Detection
"""
ViT Backbone Encoder for RGB-D processing (DINOv2-compatible).

This is a standalone implementation that doesn't require torch.hub,
suitable for offline training environments. 

Uses Vision Transformer as backbone with:
- Complete freezing for RGB (domain-invariant features)
- Partial freezing for depth (geometry adaptation)
- Multi-scale feature extraction from intermediate layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from functools import partial
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=14, in_chans=3, embed_dim=1024):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/patch, W/patch)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class Mlp(nn.Module):
    """MLP as used in Vision Transformer."""
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    def __init__(self, dim, num_heads=8, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Use scaled_dot_product_attention if available (PyTorch 2.0+)
        if hasattr(F, 'scaled_dot_product_attention'):
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block with LayerNorm, Attention, and MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer backbone.
    
    Compatible with DINOv2 architecture but doesn't require torch.hub.
    """
    def __init__(
        self,
        img_size: int = 518,  # 37 * 14
        patch_size: int = 14,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_blocks = depth
        
        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, 
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize position embeddings with truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize linear layers
        self.apply(self._init_module)
    
    def _init_module(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def interpolate_pos_embed(self, x, h, w):
        """Interpolate position embeddings for different image sizes."""
        npatch = x.shape[1] - 1  # Exclude CLS token
        N = self.pos_embed.shape[1] - 1
        
        if npatch == N and h == w:
            return self.pos_embed
        
        # Separate CLS token and patch embeddings
        cls_embed = self.pos_embed[:, 0:1, :]
        patch_embed = self.pos_embed[:, 1:, :]
        
        # Reshape and interpolate
        dim = self.embed_dim
        orig_size = int(N ** 0.5)
        patch_embed = patch_embed.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
        patch_embed = F.interpolate(patch_embed, size=(h, w), mode='bicubic', align_corners=False)
        patch_embed = patch_embed.permute(0, 2, 3, 1).reshape(1, h*w, dim)
        
        return torch.cat([cls_embed, patch_embed], dim=1)
    
    def forward_features(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        """Forward pass returning features."""
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1 + num_patches, embed_dim)
        
        # Add position embeddings
        h = H // self.patch_size
        w = W // self.patch_size
        pos_embed = self.interpolate_pos_embed(x, h, w)
        x = x + pos_embed
        
        # Apply transformer blocks
        for blk in self.blocks:
            x = blk(x)
        
        x = self.norm(x)
        
        if return_all_tokens:
            return x  # (B, 1 + num_patches, embed_dim)
        else:
            return x[:, 0]  # Just CLS token: (B, embed_dim)
    
    def get_intermediate_layers(self, x: torch.Tensor, layer_indices: List[int]) -> List[torch.Tensor]:
        """Get features from intermediate transformer layers.
        
        Args:
            x: Input tensor (B, C, H, W)
            layer_indices: Which layers to extract (0-indexed)
            
        Returns:
            List of features from specified layers, each (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        h = H // self.patch_size
        w = W // self.patch_size
        pos_embed = self.interpolate_pos_embed(x, h, w)
        x = x + pos_embed
        
        # Run through blocks and collect intermediate outputs
        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in layer_indices:
                # Apply norm and exclude CLS token
                normed = self.norm(x)
                outputs.append(normed[:, 1:, :])  # Exclude CLS token
        
        return outputs


class ViTEncoder(nn.Module):
    """ViT encoder for a single modality (RGB or Depth).
    
    Provides multi-scale features by extracting from intermediate transformer blocks.
    
    Args:
        model_name: ViT model variant ('vit_small', 'vit_base', 'vit_large', 'vit_giant')
        in_channels: Number of input channels (3 for RGB, 1 for depth)
        freeze_all: If True, freeze entire model
        num_frozen_blocks: Number of transformer blocks to freeze (from start)
        output_layers: Which transformer layers to extract features from
        weights_path: Optional path to pretrained weights
    """
    
    # Layer indices for multi-scale features
    DEFAULT_OUTPUT_LAYERS = {
        'vit_small': [2, 5, 8, 11],      # 12 blocks
        'vit_base': [2, 5, 8, 11],       # 12 blocks  
        'vit_large': [4, 11, 17, 23],    # 24 blocks
        'vit_giant': [9, 19, 29, 39],    # 40 blocks
    }
    
    VIT_CONFIGS = {
        'vit_small': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'vit_base': {'embed_dim': 768, 'depth': 12, 'num_heads': 12},
        'vit_large': {'embed_dim': 1024, 'depth': 24, 'num_heads': 16},
        'vit_giant': {'embed_dim': 1536, 'depth': 40, 'num_heads': 24},
    }
    
    def __init__(
        self, 
        model_name: str = 'vit_large',
        in_channels: int = 3,
        freeze_all: bool = False,
        num_frozen_blocks: int = 0,
        output_layers: Optional[List[int]] = None,
        weights_path: Optional[str] = None,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.in_channels = in_channels
        self.freeze_all = freeze_all
        self.num_frozen_blocks = num_frozen_blocks
        
        # Get config
        config = self.VIT_CONFIGS[model_name]
        self.embed_dim = config['embed_dim']
        self.patch_size = 14  # Compatible with DINOv2
        
        # Create ViT
        self.vit = VisionTransformer(
            patch_size=self.patch_size,
            in_chans=in_channels,
            embed_dim=config['embed_dim'],
            depth=config['depth'],
            num_heads=config['num_heads'],
        )
        
        # Load weights if provided
        if weights_path is not None:
            self._load_weights(weights_path)
        
        # Determine output layers
        if output_layers is None:
            self.output_layers = self.DEFAULT_OUTPUT_LAYERS[model_name]
        else:
            self.output_layers = output_layers
        
        # Apply freezing
        self._apply_freezing()
        
        # Output feature info
        self._out_feature_strides = {
            'layer0': 4,
            'layer1': 8,
            'layer2': 16,
            'layer3': 32,
        }
        self._out_feature_channels = {
            'layer0': self.embed_dim,
            'layer1': self.embed_dim,
            'layer2': self.embed_dim,
            'layer3': self.embed_dim,
        }
    
    def _load_weights(self, path: str):
        """Load weights from a checkpoint file."""
        state_dict = torch.load(path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Load with non-strict to handle mismatches
        missing, unexpected = self.vit.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys when loading ViT: {missing[:5]}...")
        if unexpected:
            print(f"Unexpected keys when loading ViT: {unexpected[:5]}...")
    
    def _apply_freezing(self):
        """Apply freezing based on settings."""
        if self.freeze_all:
            for param in self.vit.parameters():
                param.requires_grad = False
            return
        
        # Freeze patch embedding
        if self.num_frozen_blocks > 0:
            for param in self.vit.patch_embed.parameters():
                param.requires_grad = False
            self.vit.cls_token.requires_grad = False
        
        # Freeze transformer blocks
        for i, block in enumerate(self.vit.blocks):
            if i < self.num_frozen_blocks:
                for param in block.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from ViT.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Dict of features at different "scales" {layer0, layer1, layer2, layer3}
        """
        B, C, H, W = x.shape
        
        # Get intermediate features
        features = self.vit.get_intermediate_layers(x, self.output_layers)
        
        # Reshape from (B, N, C) to (B, C, h, w) for each layer
        h = H // self.patch_size
        w = W // self.patch_size
        
        out = {}
        for i, feat in enumerate(features):
            # Reshape to spatial format: (B, N, C) -> (B, C, h, w)
            spatial = feat.permute(0, 2, 1).reshape(B, self.embed_dim, h, w)
            out[f'layer{i}'] = spatial
        
        return out


class ViTDualEncoder(Backbone):
    """Dual ViT encoder for RGB-D processing.
    
    Uses separate ViT encoders for RGB and depth, then fuses features
    to create FPN-compatible multi-scale outputs.
    
    Args:
        cfg: Detectron2 config
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        # Get config values
        model_name = cfg.MODEL.BACKBONE.get('VIT_MODEL', 'vit_large')
        rgb_freeze = cfg.MODEL.BACKBONE.get('RGB_FREEZE_ALL', True)
        depth_freeze_blocks = cfg.MODEL.BACKBONE.get('DEPTH_NUM_FROZEN_BLOCKS', 12)
        fusion_type = cfg.MODEL.BACKBONE.get('FUSION_TYPE', 'concat')
        out_channels = cfg.MODEL.BACKBONE.get('OUT_CHANNELS', 256)
        rgb_weights = cfg.MODEL.BACKBONE.get('RGB_WEIGHTS', None)
        depth_weights = cfg.MODEL.BACKBONE.get('DEPTH_WEIGHTS', None)
        
        # Create encoders
        self.rgb_encoder = ViTEncoder(
            model_name=model_name,
            in_channels=3,
            freeze_all=rgb_freeze,
            weights_path=rgb_weights,
        )
        
        self.depth_encoder = ViTEncoder(
            model_name=model_name,
            in_channels=1,
            num_frozen_blocks=depth_freeze_blocks,
            weights_path=depth_weights,
        )
        
        embed_dim = self.rgb_encoder.embed_dim
        self.fusion_type = fusion_type
        
        # Fusion layers
        if fusion_type == 'concat':
            fusion_in = embed_dim * 2
        else:
            fusion_in = embed_dim
        
        # Project to FPN channels
        self.proj_layers = nn.ModuleDict()
        strides = [4, 8, 16, 32]
        for i, stride in enumerate(strides):
            self.proj_layers[f'p{i+2}'] = nn.Sequential(
                nn.Conv2d(fusion_in, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        
        # Gated fusion if needed
        if fusion_type == 'gated':
            self.gates = nn.ModuleDict()
            for i in range(4):
                self.gates[f'layer{i}'] = nn.Sequential(
                    nn.Conv2d(embed_dim * 2, embed_dim, 1),
                    nn.Sigmoid(),
                )
        
        # Output specs
        self._out_feature_strides = {'p2': 4, 'p3': 8, 'p4': 16, 'p5': 32}
        self._out_feature_channels = {f'p{i}': out_channels for i in range(2, 6)}
        self._out_features = ['p2', 'p3', 'p4', 'p5']
    
    @property
    def size_divisibility(self) -> int:
        return 32
    
    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }
    
    def _resize_features(self, features: Dict[str, torch.Tensor], target_strides: Dict[str, int], input_size: Tuple[int, int]) -> Dict[str, torch.Tensor]:
        """Resize features to match FPN stride expectations."""
        H, W = input_size
        resized = {}
        for name, feat in features.items():
            if name not in target_strides:
                continue
            target_stride = target_strides[name]
            target_h = H // target_stride
            target_w = W // target_stride
            if feat.shape[2] != target_h or feat.shape[3] != target_w:
                feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
            resized[name] = feat
        return resized
    
    def forward(self, images: torch.Tensor, depth: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for RGB-D inputs.
        
        Args:
            images: RGB images (B, 3, H, W)
            depth: Depth maps (B, 1, H, W)
            
        Returns:
            FPN-style features {p2, p3, p4, p5}
        """
        B, _, H, W = images.shape
        
        # Get features from each encoder
        rgb_features = self.rgb_encoder(images)
        depth_features = self.depth_encoder(depth)
        
        # Target strides for resizing
        layer_to_fpn = {'layer0': 'p2', 'layer1': 'p3', 'layer2': 'p4', 'layer3': 'p5'}
        target_strides = {f'layer{i}': self._out_feature_strides[f'p{i+2}'] for i in range(4)}
        
        # Resize features to match FPN strides
        rgb_features = self._resize_features(rgb_features, target_strides, (H, W))
        depth_features = self._resize_features(depth_features, target_strides, (H, W))
        
        # Fuse features
        fused = {}
        for layer_name in rgb_features.keys():
            rgb_f = rgb_features[layer_name]
            depth_f = depth_features[layer_name]
            
            if self.fusion_type == 'concat':
                fused[layer_name] = torch.cat([rgb_f, depth_f], dim=1)
            elif self.fusion_type == 'add':
                fused[layer_name] = rgb_f + depth_f
            elif self.fusion_type == 'gated':
                concat = torch.cat([rgb_f, depth_f], dim=1)
                gate = self.gates[layer_name](concat)
                fused[layer_name] = gate * rgb_f + (1 - gate) * depth_f
            else:
                raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
        # Project to output channels and rename to FPN names
        outputs = {}
        for layer_name, fpn_name in layer_to_fpn.items():
            if layer_name in fused:
                outputs[fpn_name] = self.proj_layers[fpn_name](fused[layer_name])
        
        return outputs


@BACKBONE_REGISTRY.register()
def build_vit_dual_encoder(cfg, input_shape):
    """Build ViT dual encoder backbone from config."""
    return ViTDualEncoder(cfg)
