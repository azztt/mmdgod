# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Layer-wise Learning Rate Decay for ViT/DINOv3 models.

Implements the layer-wise LR decay strategy from:
- BEiT: https://arxiv.org/abs/2106.08254
- ViTDet: https://arxiv.org/abs/2203.16527

Earlier layers get lower learning rates to preserve pretrained features,
while later layers get higher LRs for task adaptation.
"""

import torch
from typing import Dict, List, Set, Optional
from detectron2.solver import build_lr_scheduler as d2_build_lr_scheduler


def get_vit_lr_decay_rate(name: str, lr_decay_rate: float = 0.75, num_layers: int = 12) -> float:
    """
    Calculate the LR decay rate for a parameter based on its layer depth.
    
    For ViT/DINO models, parameters in earlier layers get lower LR multipliers.
    
    Args:
        name: Parameter name (e.g., "backbone.rgb_encoder.model.layer.5....")
        lr_decay_rate: Base decay rate (0.65-0.9 typical)
        num_layers: Number of transformer layers
    
    Returns:
        LR multiplier for this parameter
    """
    # Default: no decay
    layer_id = num_layers
    
    # Check if this is a transformer layer parameter
    # DINOv3/DINOv2 structure: model.layer.{i}.* or model.encoder.layer.{i}.*
    if ".layer." in name:
        # Extract layer number
        parts = name.split(".layer.")
        if len(parts) > 1:
            layer_part = parts[1].split(".")[0]
            try:
                layer_id = int(layer_part)
            except ValueError:
                pass
    elif ".blocks." in name:
        # Alternative ViT structure: blocks.{i}.*
        parts = name.split(".blocks.")
        if len(parts) > 1:
            layer_part = parts[1].split(".")[0]
            try:
                layer_id = int(layer_part)
            except ValueError:
                pass
    
    # Embedding layers (patch_embed, cls_token, pos_embed) get lowest LR
    if any(x in name for x in ["patch_embed", "cls_token", "pos_embed", "mask_token"]):
        layer_id = 0
    
    # Calculate decay: layer 0 gets decay^num_layers, layer num_layers-1 gets decay^1
    # Higher layer_id = later layer = higher LR (less decay)
    scale = lr_decay_rate ** (num_layers - layer_id)
    
    return scale


def get_layer_wise_lr_groups(
    model: torch.nn.Module,
    base_lr: float,
    backbone_lr_multiplier: float = 0.1,
    lr_decay_rate: float = 0.75,
    weight_decay: float = 0.05,
    weight_decay_norm: float = 0.0,
    skip_lr_decay: Optional[Set[str]] = None,
) -> List[Dict]:
    """
    Create parameter groups with layer-wise LR decay for ViT/DINO models.
    
    Args:
        model: The full detection model
        base_lr: Base learning rate for detection head
        backbone_lr_multiplier: Multiplier for backbone (e.g., 0.1)
        lr_decay_rate: Layer-wise decay rate (e.g., 0.75)
        weight_decay: Weight decay for regular params
        weight_decay_norm: Weight decay for norm layers (typically 0)
        skip_lr_decay: Parameter name patterns to skip LR decay
    
    Returns:
        List of parameter groups for optimizer
    """
    if skip_lr_decay is None:
        skip_lr_decay = {"pos_embed", "cls_token", "relative_position_bias_table"}
    
    # Determine number of layers in backbone
    num_layers = 12  # Default for ViT-Base
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        if hasattr(backbone, 'rgb_encoder'):
            # Check DINOv3/v2 structure
            encoder = backbone.rgb_encoder
            if hasattr(encoder, 'model'):
                if hasattr(encoder.model, 'layer'):
                    num_layers = len(encoder.model.layer)
                elif hasattr(encoder.model, 'encoder') and hasattr(encoder.model.encoder, 'layer'):
                    num_layers = len(encoder.model.encoder.layer)
    
    # Group parameters
    param_groups = {}
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Determine if this is a backbone parameter
        is_backbone = "backbone" in name
        
        # Get base LR for this parameter
        if is_backbone:
            param_lr = base_lr * backbone_lr_multiplier
            
            # Apply layer-wise decay for backbone
            should_decay_lr = not any(skip in name for skip in skip_lr_decay)
            if should_decay_lr:
                decay_scale = get_vit_lr_decay_rate(name, lr_decay_rate, num_layers)
                param_lr = param_lr * decay_scale
        else:
            param_lr = base_lr
        
        # Determine weight decay
        if "norm" in name.lower() or "bias" in name.lower() or "ln" in name.lower():
            param_wd = weight_decay_norm
        else:
            param_wd = weight_decay
        
        # Create group key
        group_key = (param_lr, param_wd)
        
        if group_key not in param_groups:
            param_groups[group_key] = {
                "params": [],
                "lr": param_lr,
                "weight_decay": param_wd,
            }
        
        param_groups[group_key]["params"].append(param)
    
    # Convert to list
    groups = list(param_groups.values())
    
    # Log summary
    print(f"Created {len(groups)} parameter groups with layer-wise LR decay:")
    for i, g in enumerate(groups[:5]):  # Show first 5
        print(f"  Group {i}: LR={g['lr']:.6f}, WD={g['weight_decay']:.4f}, params={len(g['params'])}")
    if len(groups) > 5:
        print(f"  ... and {len(groups)-5} more groups")
    
    return groups


def build_optimizer_with_layer_decay(
    cfg,
    model: torch.nn.Module,
) -> torch.optim.Optimizer:
    """
    Build optimizer with layer-wise LR decay for ViT/DINO models.
    
    Args:
        cfg: Detectron2 config
        model: The model to optimize
    
    Returns:
        Configured optimizer
    """
    # Get config values
    base_lr = cfg.SOLVER.BASE_LR
    backbone_multiplier = cfg.SOLVER.BACKBONE_MULTIPLIER
    weight_decay = cfg.SOLVER.WEIGHT_DECAY
    weight_decay_norm = getattr(cfg.SOLVER, 'WEIGHT_DECAY_NORM', 0.0)
    
    # Layer decay settings
    lr_decay_rate = getattr(cfg.SOLVER, 'LR_LAYER_DECAY', 0.75)
    
    # Get parameter groups
    param_groups = get_layer_wise_lr_groups(
        model,
        base_lr=base_lr,
        backbone_lr_multiplier=backbone_multiplier,
        lr_decay_rate=lr_decay_rate,
        weight_decay=weight_decay,
        weight_decay_norm=weight_decay_norm,
    )
    
    # Build optimizer
    optimizer_type = cfg.SOLVER.TYPE.lower()
    
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=base_lr,  # Will be overridden by per-group LR
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    elif optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=base_lr,
            momentum=cfg.SOLVER.MOMENTUM,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer
