# Copyright (c) Meta Platforms, Inc. and affiliates
import torch
from typing import Any, Dict, List, Set
from detectron2.solver.build import maybe_add_gradient_clipping

def build_optimizer(cfg, model):
    """
    Build optimizer with support for:
    - Different optimizers (SGD, Adam, AdamW)
    - Per-parameter learning rates (backbone vs head)
    - Weight decay exclusion for norms and biases
    - Gradient clipping
    """
    norm_module_types = (
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.SyncBatchNorm,
        torch.nn.GroupNorm,
        torch.nn.InstanceNorm1d,
        torch.nn.InstanceNorm2d,
        torch.nn.InstanceNorm3d,
        torch.nn.LayerNorm,
        torch.nn.LocalResponseNorm,
    )
    
    # Get backbone multiplier (default 1.0 = same LR as head)
    backbone_lr_multiplier = getattr(cfg.SOLVER, 'BACKBONE_MULTIPLIER', 1.0)
    
    params: List[Dict[str, Any]] = []
    memo: Set[torch.nn.parameter.Parameter] = set()
    
    for module_name, module in model.named_modules():
        for key, value in module.named_parameters(recurse=False):
            if not value.requires_grad:
                continue
            # Avoid duplicating parameters
            if value in memo:
                continue
            memo.add(value)
            
            lr = cfg.SOLVER.BASE_LR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY

            # Apply backbone LR multiplier
            # Backbone includes: rgb_backbone, depth_backbone, backbone (for single encoder)
            is_backbone = any(bb in module_name for bb in ['rgb_backbone', 'depth_backbone', 'backbone.bottom_up'])
            if is_backbone and backbone_lr_multiplier != 1.0:
                lr = cfg.SOLVER.BASE_LR * backbone_lr_multiplier

            # Norm layers: use WEIGHT_DECAY_NORM (typically 0)
            if isinstance(module, norm_module_types) and (cfg.SOLVER.WEIGHT_DECAY_NORM is not None):
                weight_decay = cfg.SOLVER.WEIGHT_DECAY_NORM
            
            # Bias parameters: optionally different LR and weight decay
            elif key == "bias":
                if (cfg.SOLVER.BIAS_LR_FACTOR is not None):
                    lr = lr * cfg.SOLVER.BIAS_LR_FACTOR  # Scale from current lr, not base
                if (cfg.SOLVER.WEIGHT_DECAY_BIAS is not None):
                    weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS

            # Special parameters that should never have weight decay
            if key in ['priors_dims_per_cat', 'priors_z_scales', 'priors_z_stats']:
                weight_decay = 0.0

            params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # Log parameter group summary
    backbone_params = sum(p["params"][0].numel() for p in params if p["lr"] < cfg.SOLVER.BASE_LR)
    head_params = sum(p["params"][0].numel() for p in params if p["lr"] >= cfg.SOLVER.BASE_LR)
    if backbone_lr_multiplier != 1.0:
        print(f"[Optimizer] Backbone params: {backbone_params:,} (lr={cfg.SOLVER.BASE_LR * backbone_lr_multiplier:.6f})")
        print(f"[Optimizer] Head params: {head_params:,} (lr={cfg.SOLVER.BASE_LR:.6f})")

    if cfg.SOLVER.TYPE == 'sgd':
        optimizer = torch.optim.SGD(
            params, 
            cfg.SOLVER.BASE_LR, 
            momentum=cfg.SOLVER.MOMENTUM, 
            nesterov=cfg.SOLVER.NESTEROV, 
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
    elif cfg.SOLVER.TYPE == 'adam':
        optimizer = torch.optim.Adam(params, cfg.SOLVER.BASE_LR, eps=1e-02)
    elif cfg.SOLVER.TYPE == 'adam+amsgrad':
        optimizer = torch.optim.Adam(params, cfg.SOLVER.BASE_LR, amsgrad=True, eps=1e-02)
    elif cfg.SOLVER.TYPE == 'adamw':
        optimizer = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR, eps=1e-02)
    elif cfg.SOLVER.TYPE == 'adamw+amsgrad':
        optimizer = torch.optim.AdamW(params, cfg.SOLVER.BASE_LR, amsgrad=True, eps=1e-02)
    else:
        raise ValueError('{} is not supported as an optimizer.'.format(cfg.SOLVER.TYPE))

    optimizer = maybe_add_gradient_clipping(cfg, optimizer)
    return optimizer

def freeze_bn(network):

    for _, module in network.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.eval()
            module.track_running_stats = False
