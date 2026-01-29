# Copyright (c) Meta Platforms, Inc. and affiliates
# DINO Backbone for RGB-D 3D Object Detection (supports DINOv2 and DINOv3)
"""
DINO Backbone Encoder for RGB-D processing using HuggingFace transformers.

Supports:
- DINOv2 (facebook/dinov2-small, dinov2-base, dinov2-large, dinov2-giant)
- DINOv3 (facebook/dinov3-vits16plus-pretrain-lvd1689m, etc.)

Uses Vision Transformer as backbone with:
- Complete freezing for RGB (domain-invariant features)
- Partial freezing for depth (geometry adaptation)
- Multi-scale feature extraction from intermediate layers
- Optional adaptation transformer encoders before fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, List, Optional, Tuple
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import Backbone, BACKBONE_REGISTRY

import logging
logger = logging.getLogger(__name__)

# Try to import transformers
try:
    from transformers import Dinov2Model, Dinov2Config, AutoModel, AutoConfig
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    logger.warning("transformers library not installed. Install with: pip install transformers")


# Model configurations
MODEL_CONFIGS = {
    # DINOv2 models (HuggingFace)
    'dinov2-small': {
        'hf_name': 'facebook/dinov2-small',
        'embed_dim': 384,
        'num_layers': 12,
        'patch_size': 14,
        'output_layers': [2, 5, 8, 11],
    },
    'dinov2-base': {
        'hf_name': 'facebook/dinov2-base',
        'embed_dim': 768,
        'num_layers': 12,
        'patch_size': 14,
        'output_layers': [2, 5, 8, 11],
    },
    'dinov2-large': {
        'hf_name': 'facebook/dinov2-large',
        'embed_dim': 1024,
        'num_layers': 24,
        'patch_size': 14,
        'output_layers': [5, 11, 17, 23],
    },
    'dinov2-giant': {
        'hf_name': 'facebook/dinov2-giant',
        'embed_dim': 1536,
        'num_layers': 40,
        'patch_size': 14,
        'output_layers': [9, 19, 29, 39],
    },
    # DINOv3 models (HuggingFace)
    'dinov3-vits16': {
        'hf_name': 'facebook/dinov3-vits16plus-pretrain-lvd1689m',
        'embed_dim': 384,
        'num_layers': 12,
        'patch_size': 16,
        'output_layers': [2, 5, 8, 11],
    },
    'dinov3-vitb16': {
        'hf_name': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        'embed_dim': 768,
        'num_layers': 12,
        'patch_size': 16,
        'output_layers': [2, 5, 8, 11],
    },
    'dinov3-vitl16': {
        'hf_name': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
        'embed_dim': 1024,
        'num_layers': 24,
        'patch_size': 16,
        'output_layers': [5, 11, 17, 23],
    },
}


class AdaptationTransformerEncoder(nn.Module):
    """Lightweight conv-based adapter for feature refinement.
    
    Uses depth-wise separable convolutions instead of transformers
    to avoid O(n²) attention complexity on large feature maps.
    
    Args:
        embed_dim: Input/output embedding dimension
        num_layers: Number of conv blocks
        num_heads: Unused (kept for interface compatibility)
        mlp_ratio: Expansion ratio for conv blocks
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,  # Ignored, using bottleneck instead
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # Super lightweight: bottleneck instead of expansion
        # 768 → 256 → 768 (reduces memory 3x vs expansion)
        bottleneck_dim = embed_dim // 3  # 768 → 256
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(
                # Compress
                nn.Conv2d(embed_dim, bottleneck_dim, 1, bias=False),
                nn.BatchNorm2d(bottleneck_dim),
                nn.GELU(),
                # Spatial mixing (depthwise)
                nn.Conv2d(bottleneck_dim, bottleneck_dim, 3, padding=1, groups=bottleneck_dim, bias=False),
                nn.BatchNorm2d(bottleneck_dim),
                nn.GELU(),
                # Expand back
                nn.Conv2d(bottleneck_dim, embed_dim, 1, bias=False),
                nn.BatchNorm2d(embed_dim),
            ))
        self.blocks = nn.ModuleList(layers)
        
        # Initialize weights
        self._init_weights()
        
        logger.info(f"AdaptationEncoder: {num_layers} bottleneck blocks, {embed_dim}→{bottleneck_dim}→{embed_dim}")
    
    def _init_weights(self):
        """Initialize with small random weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Features (B, C, H, W)
            
        Returns:
            Adapted features (B, C, H, W)
        """
        # Apply conv blocks with residual connections
        for block in self.blocks:
            x = x + block(x)  # Residual
        
        return x


class DinoEncoder(nn.Module):
    """DINO encoder for a single modality (RGB or Depth).
    
    Uses HuggingFace transformers to load DINOv2/DINOv3 models from cache.
    Provides multi-scale features by extracting from intermediate transformer blocks.
    
    Args:
        model_name: Model variant (e.g., 'dinov2-large', 'dinov3-vitl16')
        in_channels: Number of input channels (3 for RGB, 1 for depth)
        freeze_all: If True, freeze entire model
        num_frozen_blocks: Number of transformer blocks to freeze (from start)
        output_layers: Which transformer layers to extract features from (0-indexed)
    """
    
    def __init__(
        self, 
        model_name: str = 'dinov3-vitl16',
        in_channels: int = 3,
        freeze_all: bool = False,
        num_frozen_blocks: int = 0,
        output_layers: Optional[List[int]] = None,
    ):
        super().__init__()
        
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers library required. Install with: pip install transformers")
        
        self.model_name = model_name
        self.in_channels = in_channels
        self.freeze_all = freeze_all
        self.num_frozen_blocks = num_frozen_blocks
        
        # Get model config
        if model_name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")
        
        config = MODEL_CONFIGS[model_name]
        self.embed_dim = config['embed_dim']
        self.patch_size = config['patch_size']
        self.num_layers = config['num_layers']
        hf_name = config['hf_name']
        
        # Determine output layers
        if output_layers is None:
            self.output_layers = config['output_layers']
        else:
            self.output_layers = output_layers
        
        # Load model from HuggingFace (will use cache if available)
        logger.info(f"Loading {model_name} from HuggingFace: {hf_name}")
        self.dino = AutoModel.from_pretrained(
            hf_name, 
            local_files_only=True,  # Only use cached files
            output_hidden_states=True,  # Enable intermediate outputs
        )
        
        # Verify pretrained weights loaded correctly
        self._verify_pretrained_weights(model_name)
        
        # Check for register tokens (DINOv3 uses them)
        self.num_register_tokens = getattr(self.dino.config, 'num_register_tokens', 0)
        self.num_prefix_tokens = 1 + self.num_register_tokens  # CLS + registers
        
        # Modify patch embedding if input channels != 3
        if in_channels != 3:
            self._modify_patch_embed(in_channels)
        
        # Apply freezing
        self._apply_freezing()
        
        # Output feature info (virtual strides for FPN compatibility)
        self._out_feature_strides = {
            'layer0': 4,
            'layer1': 8,
            'layer2': 16,
            'layer3': 32,
        }
        self._out_feature_channels = {
            f'layer{i}': self.embed_dim for i in range(4)
        }
    
    def _verify_pretrained_weights(self, model_name: str):
        """Verify that pretrained weights are loaded (not random)."""
        # Check first layer norm stats - pretrained models have non-trivial values
        if hasattr(self.dino, 'embeddings'):
            # Check patch embedding weights
            if hasattr(self.dino.embeddings, 'patch_embeddings'):
                pe = self.dino.embeddings.patch_embeddings
                if isinstance(pe, nn.Conv2d):
                    weight_std = pe.weight.std().item()
                elif hasattr(pe, 'projection'):
                    weight_std = pe.projection.weight.std().item()
                else:
                    weight_std = 0
                
                # Pretrained weights typically have std around 0.01-0.1
                if weight_std < 0.001 or weight_std > 1.0:
                    logger.warning(f"Patch embedding weight std={weight_std:.6f} - may not be pretrained!")
                else:
                    logger.info(f"✓ Pretrained weights verified for {model_name} (patch_embed std={weight_std:.4f})")
        
        # Count total parameters
        total_params = sum(p.numel() for p in self.dino.parameters())
        logger.info(f"  Loaded {total_params:,} parameters from {model_name}")
    
    def _modify_patch_embed(self, in_channels: int):
        """Modify patch embedding to accept different number of input channels."""
        # Get the patch embeddings module - different structures for DINOv2 vs DINOv3
        if hasattr(self.dino, 'embeddings'):
            embeddings = self.dino.embeddings
            # Check if patch_embeddings is a Conv2d directly (DINOv3) or has .projection (DINOv2)
            if isinstance(embeddings.patch_embeddings, nn.Conv2d):
                patch_embed = embeddings.patch_embeddings
                is_direct_conv = True
            elif hasattr(embeddings.patch_embeddings, 'projection'):
                patch_embed = embeddings.patch_embeddings.projection
                is_direct_conv = False
            else:
                raise ValueError(f"Unknown patch_embeddings structure: {type(embeddings.patch_embeddings)}")
        else:
            # Fallback for different model architectures
            patch_embed = self.dino.patch_embed.proj
            is_direct_conv = False
        
        # Create new conv with different input channels
        new_proj = nn.Conv2d(
            in_channels,
            patch_embed.out_channels,
            kernel_size=patch_embed.kernel_size,
            stride=patch_embed.stride,
            padding=patch_embed.padding,
        )
        
        # Initialize with averaged pretrained weights
        with torch.no_grad():
            # Average RGB weights for single-channel input
            new_proj.weight.data = patch_embed.weight.data.mean(dim=1, keepdim=True)
            if in_channels > 1:
                new_proj.weight.data = new_proj.weight.data.expand(-1, in_channels, -1, -1).clone()
            if patch_embed.bias is not None:
                new_proj.bias.data = patch_embed.bias.data.clone()
        
        # Replace the projection
        if hasattr(self.dino, 'embeddings'):
            if is_direct_conv:
                self.dino.embeddings.patch_embeddings = new_proj
            else:
                self.dino.embeddings.patch_embeddings.projection = new_proj
        else:
            self.dino.patch_embed.proj = new_proj
    
    def _apply_freezing(self):
        """Apply freezing based on settings."""
        if self.freeze_all:
            for param in self.dino.parameters():
                param.requires_grad = False
            return
        
        # Freeze embeddings if freezing any blocks
        if self.num_frozen_blocks > 0:
            if hasattr(self.dino, 'embeddings'):
                for param in self.dino.embeddings.parameters():
                    param.requires_grad = False
            # Also freeze rope embeddings if present (DINOv3)
            if hasattr(self.dino, 'rope_embeddings'):
                for param in self.dino.rope_embeddings.parameters():
                    param.requires_grad = False
        
        # Freeze transformer blocks - different structures for different models
        if hasattr(self.dino, 'layer'):
            # DINOv3 structure: model.layer
            blocks = self.dino.layer
        elif hasattr(self.dino, 'encoder') and hasattr(self.dino.encoder, 'layer'):
            # DINOv2 structure: model.encoder.layer
            blocks = self.dino.encoder.layer
        elif hasattr(self.dino, 'blocks'):
            # Fallback: model.blocks
            blocks = self.dino.blocks
        else:
            raise ValueError(f"Could not find transformer blocks in model: {type(self.dino)}")
        
        for i, block in enumerate(blocks):
            if i < self.num_frozen_blocks:
                for param in block.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features.
        
        Args:
            x: Input tensor (B, C, H, W)
            
        Returns:
            Dict of features at different scales {layer0, layer1, layer2, layer3}
        """
        B, C, H, W = x.shape
        
        # Forward through DINO
        outputs = self.dino(x, output_hidden_states=True, return_dict=True)
        
        # Get hidden states from specified layers
        hidden_states = outputs.hidden_states  # Tuple of (B, num_tokens, embed_dim)
        
        # Calculate spatial dimensions
        h = H // self.patch_size
        w = W // self.patch_size
        num_patches = h * w
        
        out = {}
        for i, layer_idx in enumerate(self.output_layers):
            # Get hidden state, exclude CLS and register tokens
            # hidden_states[0] is embeddings, hidden_states[1:] are layer outputs
            feat = hidden_states[layer_idx + 1]  # (B, num_tokens, embed_dim)
            
            # Remove prefix tokens (CLS + register tokens)
            feat = feat[:, self.num_prefix_tokens:, :]  # (B, num_patches, embed_dim)
            
            # Verify we have the right number of patches
            assert feat.shape[1] == num_patches, \
                f"Expected {num_patches} patches, got {feat.shape[1]}"
            
            # Reshape to spatial: (B, N, C) -> (B, C, h, w)
            spatial = feat.permute(0, 2, 1).reshape(B, self.embed_dim, h, w)
            out[f'layer{i}'] = spatial
        
        return out


class DinoDualEncoder(Backbone):
    """Dual DINO encoder for RGB-D processing.
    
    Uses separate DINO encoders for RGB and depth, then fuses features
    to create FPN-compatible multi-scale outputs.
    
    Supports DINOv2 and DINOv3 models from HuggingFace.
    
    Args:
        cfg: Detectron2 config
    """
    
    def __init__(self, cfg):
        super().__init__()
        
        # Get config values
        model_name = cfg.MODEL.BACKBONE.get('DINO_MODEL', 'dinov3-vitl16')
        rgb_freeze = cfg.MODEL.BACKBONE.get('RGB_FREEZE_ALL', True)
        depth_freeze_blocks = cfg.MODEL.BACKBONE.get('DEPTH_NUM_FROZEN_BLOCKS', 12)
        fusion_type = cfg.MODEL.BACKBONE.get('FUSION_TYPE', 'concat')
        out_channels = cfg.MODEL.BACKBONE.get('OUT_CHANNELS', 256)
        
        print(f"Building DinoDualEncoder with {model_name}")
        print(f"  RGB freeze: {rgb_freeze}")
        print(f"  Depth frozen blocks: {depth_freeze_blocks}")
        print(f"  Fusion: {fusion_type}")
        
        # Create RGB encoder (completely frozen)
        self.rgb_encoder = DinoEncoder(
            model_name=model_name,
            in_channels=3,
            freeze_all=rgb_freeze,
        )
        
        # Create depth encoder (partially frozen)
        self.depth_encoder = DinoEncoder(
            model_name=model_name,
            in_channels=1,
            num_frozen_blocks=depth_freeze_blocks,
        )
        
        embed_dim = self.rgb_encoder.embed_dim
        self.patch_size = self.rgb_encoder.patch_size
        self.fusion_type = fusion_type
        
        # Adaptation transformer encoders (TRAINABLE - adapt frozen backbone features)
        use_adaptation = cfg.MODEL.BACKBONE.get('USE_ADAPTATION_ENCODER', False)
        adaptation_layers = cfg.MODEL.BACKBONE.get('ADAPTATION_LAYERS', 2)
        adaptation_heads = cfg.MODEL.BACKBONE.get('ADAPTATION_HEADS', 8)
        
        self.use_adaptation = use_adaptation
        if use_adaptation:
            logger.info(f"Using {adaptation_layers}-layer adaptation encoders for RGB and Depth")
            # Separate adaptation for each modality
            self.rgb_adaptation = AdaptationTransformerEncoder(
                embed_dim=embed_dim,
                num_layers=adaptation_layers,
                num_heads=adaptation_heads,
                mlp_ratio=4.0,
                dropout=0.1,
            )
            self.depth_adaptation = AdaptationTransformerEncoder(
                embed_dim=embed_dim,
                num_layers=adaptation_layers,
                num_heads=adaptation_heads,
                mlp_ratio=4.0,
                dropout=0.1,
            )
        
        # Fusion layers
        if fusion_type == 'concat':
            fusion_in = embed_dim * 2
        else:
            fusion_in = embed_dim
        
        # Project to FPN channels with upsampling
        # Separate projection layers for RGB-only (2D branch) vs fused (3D branch)
        # This prevents 3D gradients from affecting 2D detection
        self.proj_layers = nn.ModuleDict()  # For fused features (3D)
        self.rgb_proj_layers = nn.ModuleDict()  # For RGB-only features (2D)
        strides = [4, 8, 16, 32]
        for i, stride in enumerate(strides):
            # Fused projection (RGB+Depth for 3D)
            self.proj_layers[f'p{i+2}'] = nn.Sequential(
                nn.Conv2d(fusion_in, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            # RGB-only projection (for 2D) - takes just RGB features
            self.rgb_proj_layers[f'p{i+2}'] = nn.Sequential(
                nn.Conv2d(embed_dim, out_channels, 1, bias=False),
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
        
        # Log parameter count
        self._log_parameters()
    
    @property
    def size_divisibility(self) -> int:
        return self.patch_size * 2  # Ensure divisibility by patch size
    
    def output_shape(self) -> Dict[str, ShapeSpec]:
        return {
            name: ShapeSpec(
                channels=self._out_feature_channels[name],
                stride=self._out_feature_strides[name],
            )
            for name in self._out_features
        }
    
    def _resize_to_stride(self, feat: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        """Resize feature map to target spatial size."""
        if feat.shape[2] != target_h or feat.shape[3] != target_w:
            feat = F.interpolate(feat, size=(target_h, target_w), mode='bilinear', align_corners=False)
        return feat
    
    def forward(self, images: torch.Tensor, depth: torch.Tensor = None, return_rgb_only: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass for RGB-D inputs.
        
        Args:
            images: RGB images (B, 3, H, W)
            depth: Depth maps (B, 1, H, W), optional
            return_rgb_only: If True, returns dict with both 'rgb' and 'fused' feature sets
            
        Returns:
            If return_rgb_only=False: FPN-style features {p2, p3, p4, p5}
            If return_rgb_only=True: Dict with 'rgb': {p2..p5} and 'fused': {p2..p5}
        """
        B, _, H, W = images.shape
        
        # Get features from RGB encoder
        rgb_features = self.rgb_encoder(images)
        
        # Target sizes for each FPN level
        target_sizes = {
            'layer0': (H // 4, W // 4),    # p2
            'layer1': (H // 8, W // 8),    # p3
            'layer2': (H // 16, W // 16),  # p4
            'layer3': (H // 32, W // 32),  # p5
        }
        
        layer_to_fpn = {'layer0': 'p2', 'layer1': 'p3', 'layer2': 'p4', 'layer3': 'p5'}
        
        # Build RGB-only outputs (no adaptation, no fusion)
        # Uses separate rgb_proj_layers to isolate from 3D gradients
        rgb_outputs = {}
        for layer_name, fpn_name in layer_to_fpn.items():
            rgb_f = rgb_features[layer_name]
            target_h, target_w = target_sizes[layer_name]
            rgb_f = self._resize_to_stride(rgb_f, target_h, target_w)
            # Project RGB-only to output channels using SEPARATE projection
            rgb_outputs[fpn_name] = self.rgb_proj_layers[fpn_name](rgb_f)
        
        # If no depth provided or only RGB requested without fusion
        if depth is None:
            if return_rgb_only:
                return {'rgb': rgb_outputs, 'fused': rgb_outputs}
            return rgb_outputs
        
        # Get depth features
        depth_features = self.depth_encoder(depth)
        
        # Fuse and project features for 3D
        fused_outputs = {}
        for layer_name, fpn_name in layer_to_fpn.items():
            rgb_f = rgb_features[layer_name]
            depth_f = depth_features[layer_name]
            
            # Resize to target size for this FPN level
            target_h, target_w = target_sizes[layer_name]
            rgb_f = self._resize_to_stride(rgb_f, target_h, target_w)
            depth_f = self._resize_to_stride(depth_f, target_h, target_w)
            
            # Apply adaptation encoders if enabled (now lightweight conv-based)
            if self.use_adaptation:
                rgb_f = self.rgb_adaptation(rgb_f)
                depth_f = self.depth_adaptation(depth_f)
            
            # Fuse
            if self.fusion_type == 'concat':
                fused = torch.cat([rgb_f, depth_f], dim=1)
            elif self.fusion_type == 'add':
                fused = rgb_f + depth_f
            elif self.fusion_type == 'gated':
                concat = torch.cat([rgb_f, depth_f], dim=1)
                gate = self.gates[layer_name](concat)
                fused = gate * rgb_f + (1 - gate) * depth_f
            else:
                raise ValueError(f"Unknown fusion type: {self.fusion_type}")
            
            # Project to output channels
            fused_outputs[fpn_name] = self.proj_layers[fpn_name](fused)
        
        if return_rgb_only:
            return {'rgb': rgb_outputs, 'fused': fused_outputs}
        
        return fused_outputs
    
    def _log_parameters(self):
        """Log parameter statistics for debugging."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"DinoDualEncoder Parameters:")
        logger.info(f"  Total: {total:,}")
        logger.info(f"  Trainable: {trainable:,} ({100*trainable/total:.1f}%)")
        
        # Breakdown by component
        rgb_params = sum(p.numel() for p in self.rgb_encoder.parameters())
        rgb_trainable = sum(p.numel() for p in self.rgb_encoder.parameters() if p.requires_grad)
        logger.info(f"  RGB encoder: {rgb_params:,} total, {rgb_trainable:,} trainable")
        
        depth_params = sum(p.numel() for p in self.depth_encoder.parameters())
        depth_trainable = sum(p.numel() for p in self.depth_encoder.parameters() if p.requires_grad)
        logger.info(f"  Depth encoder: {depth_params:,} total, {depth_trainable:,} trainable")
        
        if self.use_adaptation:
            adapt_params = sum(p.numel() for p in self.rgb_adaptation.parameters())
            adapt_params += sum(p.numel() for p in self.depth_adaptation.parameters())
            logger.info(f"  Adaptation encoders: {adapt_params:,} trainable")


@BACKBONE_REGISTRY.register()
def build_dino_dual_encoder(cfg, input_shape):
    """Build DINO dual encoder backbone from config."""
    return DinoDualEncoder(cfg)
