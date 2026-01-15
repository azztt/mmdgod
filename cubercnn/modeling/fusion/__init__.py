# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Fusion modules for RGB-D feature combination.

This module provides various strategies for fusing RGB and depth features
for domain-generalized 3D object detection.
"""
from .gated_fusion import GatedFusion
from .depth_aware_fusion import DepthAwareAttentionFusion
from .simple_fusion import ConcatFusion, AddFusion, AttentionFusion

__all__ = [
    'GatedFusion', 
    'DepthAwareAttentionFusion',
    'ConcatFusion', 
    'AddFusion',
    'AttentionFusion',
]
