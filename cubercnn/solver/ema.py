# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Exponential Moving Average (EMA) for model weights.

EMA maintains a moving average of model weights, which often leads to
better and more stable performance at inference time.

Usage:
    model_ema = ModelEMA(model, decay=0.9999)
    
    # During training:
    for data in dataloader:
        loss = model(data)
        loss.backward()
        optimizer.step()
        model_ema.update(model)  # Update EMA after each step
    
    # For evaluation:
    model_ema.apply_shadow(model)  # Apply EMA weights
    evaluate(model)
    model_ema.restore(model)  # Restore training weights
"""

import copy
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class ModelEMA:
    """
    Exponential Moving Average of model weights.
    
    Maintains shadow weights that are updated as:
        shadow = decay * shadow + (1 - decay) * model_weight
    
    Args:
        model: The model to track
        decay: EMA decay rate (default: 0.9999)
            Higher = smoother/slower update, more stable
            Lower = faster update, more responsive to recent changes
            Typical values: 0.999 to 0.99999
        warmup_iters: Number of iterations to use lower decay for warmup
            During warmup, decay = min(decay, (1 + iter) / (10 + iter))
        device: Device to store EMA weights (None = same as model)
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_iters: int = 2000,
        device: Optional[torch.device] = None,
    ):
        self.decay = decay
        self.warmup_iters = warmup_iters
        self.device = device
        self.num_updates = 0
        
        # Create shadow copy of model weights
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                if device is not None:
                    self.shadow[name] = param.data.clone().to(device)
                else:
                    self.shadow[name] = param.data.clone()
        
        logger.info(f"ModelEMA initialized with decay={decay}, "
                   f"tracking {len(self.shadow)} parameters")
    
    def _get_decay(self) -> float:
        """Get current decay rate with optional warmup."""
        if self.warmup_iters > 0 and self.num_updates < self.warmup_iters:
            # Ramp up decay during warmup
            # Start with lower decay for faster initial tracking
            warmup_decay = (1 + self.num_updates) / (10 + self.num_updates)
            return min(self.decay, warmup_decay)
        return self.decay
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """
        Update EMA weights with current model weights.
        
        Should be called after each optimizer step.
        """
        decay = self._get_decay()
        self.num_updates += 1
        
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                # EMA update: shadow = decay * shadow + (1 - decay) * param
                if self.device is not None:
                    new_val = param.data.to(self.device)
                else:
                    new_val = param.data
                
                self.shadow[name].mul_(decay).add_(new_val, alpha=1 - decay)
    
    def apply_shadow(self, model: nn.Module):
        """
        Apply EMA weights to model (for evaluation).
        
        Saves current model weights to self.backup for later restoration.
        """
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model: nn.Module):
        """
        Restore original model weights after evaluation.
        """
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
    
    def state_dict(self) -> Dict[str, Any]:
        """Return EMA state for checkpointing."""
        return {
            'shadow': self.shadow,
            'num_updates': self.num_updates,
            'decay': self.decay,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load EMA state from checkpoint."""
        self.shadow = state_dict['shadow']
        self.num_updates = state_dict.get('num_updates', 0)
        # decay is set at init, don't override
        logger.info(f"Loaded EMA state with {self.num_updates} updates")


class ModelEMAWithBuffers(ModelEMA):
    """
    EMA that also tracks buffers (BatchNorm running stats, etc.)
    
    Use this if your model has BatchNorm layers that you want to smooth.
    For frozen DINOv3 backbones, the base ModelEMA is usually sufficient.
    """
    
    def __init__(
        self,
        model: nn.Module,
        decay: float = 0.9999,
        warmup_iters: int = 2000,
        device: Optional[torch.device] = None,
    ):
        super().__init__(model, decay, warmup_iters, device)
        
        # Also track buffers (running_mean, running_var, etc.)
        self.shadow_buffers = {}
        for name, buf in model.named_buffers():
            if device is not None:
                self.shadow_buffers[name] = buf.data.clone().to(device)
            else:
                self.shadow_buffers[name] = buf.data.clone()
        
        self.backup_buffers = {}
        logger.info(f"ModelEMAWithBuffers tracking {len(self.shadow_buffers)} buffers")
    
    @torch.no_grad()
    def update(self, model: nn.Module):
        """Update EMA for both parameters and buffers."""
        super().update(model)
        
        decay = self._get_decay()
        for name, buf in model.named_buffers():
            if name in self.shadow_buffers:
                if self.device is not None:
                    new_val = buf.data.to(self.device)
                else:
                    new_val = buf.data
                self.shadow_buffers[name].mul_(decay).add_(new_val, alpha=1 - decay)
    
    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights and buffers to model."""
        super().apply_shadow(model)
        
        self.backup_buffers = {}
        for name, buf in model.named_buffers():
            if name in self.shadow_buffers:
                self.backup_buffers[name] = buf.data.clone()
                buf.data.copy_(self.shadow_buffers[name])
    
    def restore(self, model: nn.Module):
        """Restore original weights and buffers."""
        super().restore(model)
        
        for name, buf in model.named_buffers():
            if name in self.backup_buffers:
                buf.data.copy_(self.backup_buffers[name])
        self.backup_buffers = {}
    
    def state_dict(self) -> Dict[str, Any]:
        """Return full EMA state for checkpointing."""
        state = super().state_dict()
        state['shadow_buffers'] = self.shadow_buffers
        return state
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load full EMA state from checkpoint."""
        super().load_state_dict(state_dict)
        if 'shadow_buffers' in state_dict:
            self.shadow_buffers = state_dict['shadow_buffers']
