# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Object-Level Domain Generalization Augmentations.

Implements object-wise style transfer (Object Style Swap) and 
Frequency Space Domain Randomization (FSDR) for domain generalization.

Key augmentations:
- ObjectStyleSwap: Swaps appearance/style of objects between images of same class
- DGFSDR: Frequency Space Domain Randomization using DCT decomposition
"""
import cv2
import numpy as np
import random
import copy
from typing import Dict, List, Optional, Tuple, Any
from detectron2.data import transforms as T


class ObjectStyleSwap:
    """Object-wise style transfer augmentation.
    
    Swaps the appearance/style of objects in the image with objects of the same
    category from a reference pool. Uses histogram matching to transfer color
    distributions while preserving object shape.
    
    This augmentation helps with domain generalization by:
    1. Creating novel object appearances not seen during training
    2. Breaking spurious correlations between object appearance and background
    3. Encouraging the model to focus on shape rather than texture
    
    Args:
        p (float): Probability of applying augmentation to each object. Default: 0.3.
        blend_mode (str): How to blend styled object back. One of:
            - 'replace': Direct replacement (hard edges)
            - 'alpha': Alpha blending for smoother transitions
            Default: 'alpha'.
        alpha (float): Blending alpha for 'alpha' mode. Default: 0.7.
        min_area (int): Minimum object area to consider. Default: 100.
        expand_ratio (float): How much to expand bbox for extraction. Default: 0.1.
    """
    
    def __init__(
        self,
        p: float = 0.3,
        blend_mode: str = 'alpha',
        alpha: float = 0.7,
        min_area: int = 100,
        expand_ratio: float = 0.1,
    ):
        self.p = p
        self.blend_mode = blend_mode
        self.alpha = alpha
        self.min_area = min_area
        self.expand_ratio = expand_ratio
        
        # Category index maps category_id -> list of (image_path, bbox, mask)
        self.category_index: Dict[int, List[Tuple[str, List[int], Optional[Any]]]] = {}
        
    def build_category_index(self, dataset_dicts: List[Dict]) -> None:
        """Build index of objects by category from dataset.
        
        Args:
            dataset_dicts: List of dataset annotation dicts (Detectron2 format).
        """
        self.category_index.clear()
        
        for d in dataset_dicts:
            image_path = d.get('file_name', '')
            for ann in d.get('annotations', []):
                category_id = ann.get('category_id')
                bbox = ann.get('bbox')  # [x1, y1, x2, y2] or [x, y, w, h]
                
                if category_id is None or bbox is None:
                    continue
                
                # Convert to [x1, y1, x2, y2] if needed
                if len(bbox) == 4:
                    if ann.get('bbox_mode', 0) == 1:  # XYWH format
                        x, y, w, h = bbox
                        bbox = [x, y, x + w, y + h]
                
                # Get segmentation mask if available
                mask = ann.get('segmentation', None)
                
                if category_id not in self.category_index:
                    self.category_index[category_id] = []
                
                self.category_index[category_id].append((image_path, bbox, mask))
        
        # Log index stats
        total_objects = sum(len(v) for v in self.category_index.values())
        print(f"[ObjectStyleSwap] Built index with {len(self.category_index)} categories, "
              f"{total_objects} total objects")
    
    def _histogram_matching_region(
        self,
        source: np.ndarray,
        reference: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply histogram matching to transfer color distribution.
        
        Args:
            source: Source image region to be styled (H, W, C).
            reference: Reference image to match histogram to (H, W, C).
            mask: Optional mask indicating valid pixels.
            
        Returns:
            Styled source image with reference's color distribution.
        """
        result = source.copy().astype(np.float32)
        
        for c in range(min(source.shape[2], reference.shape[2])):
            src_channel = source[:, :, c].flatten()
            ref_channel = reference[:, :, c].flatten()
            
            if mask is not None:
                src_mask = mask.flatten().astype(bool)
                ref_mask = mask.flatten().astype(bool) if mask.shape == reference.shape[:2] else np.ones(ref_channel.shape, dtype=bool)
            else:
                src_mask = np.ones(src_channel.shape, dtype=bool)
                ref_mask = np.ones(ref_channel.shape, dtype=bool)
            
            src_valid = src_channel[src_mask]
            ref_valid = ref_channel[ref_mask]
            
            if len(src_valid) == 0 or len(ref_valid) == 0:
                continue
            
            # Compute CDFs
            src_hist, src_bins = np.histogram(src_valid, bins=256, range=(0, 255), density=True)
            ref_hist, ref_bins = np.histogram(ref_valid, bins=256, range=(0, 255), density=True)
            
            src_cdf = np.cumsum(src_hist)
            src_cdf = src_cdf / src_cdf[-1] if src_cdf[-1] > 0 else src_cdf
            
            ref_cdf = np.cumsum(ref_hist)
            ref_cdf = ref_cdf / ref_cdf[-1] if ref_cdf[-1] > 0 else ref_cdf
            
            # Create mapping function
            lookup = np.zeros(256)
            for i in range(256):
                j = np.searchsorted(ref_cdf, src_cdf[i])
                lookup[i] = min(j, 255)
            
            # Apply mapping
            result_channel = result[:, :, c].flatten()
            result_channel = lookup[np.clip(result_channel, 0, 255).astype(np.int32)]
            result[:, :, c] = result_channel.reshape(source.shape[:2])
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def _extract_object_region(
        self,
        image: np.ndarray,
        bbox: List[int],
        mask: Optional[Any] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract object region with soft ellipse mask.
        
        Args:
            image: Full image (H, W, C).
            bbox: Bounding box [x1, y1, x2, y2].
            mask: Optional segmentation mask.
            
        Returns:
            Tuple of (cropped_region, soft_mask).
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = [int(b) for b in bbox]
        
        # Expand bbox slightly
        dx = int((x2 - x1) * self.expand_ratio)
        dy = int((y2 - y1) * self.expand_ratio)
        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(w, x2 + dx)
        y2 = min(h, y2 + dy)
        
        region = image[y1:y2, x1:x2].copy()
        
        if region.size == 0:
            return region, np.zeros((0, 0), dtype=np.float32)
        
        # Create soft ellipse mask
        region_h, region_w = region.shape[:2]
        
        if mask is not None:
            # Use provided mask
            try:
                from pycocotools import mask as mask_util
                if isinstance(mask, dict):
                    binary_mask = mask_util.decode(mask)
                elif isinstance(mask, list):
                    # Polygon format
                    rle = mask_util.frPyObjects(mask, h, w)
                    binary_mask = mask_util.decode(rle)
                    if binary_mask.ndim == 3:
                        binary_mask = binary_mask[:, :, 0]
                else:
                    binary_mask = None
                
                if binary_mask is not None:
                    soft_mask = binary_mask[y1:y2, x1:x2].astype(np.float32)
                    # Gaussian blur for soft edges
                    soft_mask = cv2.GaussianBlur(soft_mask, (5, 5), 0)
                    return region, soft_mask
            except Exception:
                pass
        
        # Fallback: create ellipse mask
        soft_mask = np.zeros((region_h, region_w), dtype=np.float32)
        center = (region_w // 2, region_h // 2)
        axes = (region_w // 2, region_h // 2)
        cv2.ellipse(soft_mask, center, axes, 0, 0, 360, 1.0, -1)
        soft_mask = cv2.GaussianBlur(soft_mask, (5, 5), 0)
        
        return region, soft_mask
    
    def _blend_region(
        self,
        target_img: np.ndarray,
        styled_region: np.ndarray,
        bbox: List[int],
        mask: np.ndarray,
    ) -> np.ndarray:
        """Blend styled region back into target image.
        
        Args:
            target_img: Target image to blend into (H, W, C).
            styled_region: Styled object region (h, w, C).
            bbox: Target bounding box [x1, y1, x2, y2].
            mask: Soft mask for blending (h, w).
            
        Returns:
            Image with blended styled region.
        """
        result = target_img.copy()
        h, w = target_img.shape[:2]
        x1, y1, x2, y2 = [int(b) for b in bbox]
        
        # Expand bbox
        dx = int((x2 - x1) * self.expand_ratio)
        dy = int((y2 - y1) * self.expand_ratio)
        x1 = max(0, x1 - dx)
        y1 = max(0, y1 - dy)
        x2 = min(w, x2 + dx)
        y2 = min(h, y2 + dy)
        
        target_h = y2 - y1
        target_w = x2 - x1
        
        if target_h <= 0 or target_w <= 0:
            return result
        
        # Resize styled region to target size
        styled_resized = cv2.resize(styled_region, (target_w, target_h))
        mask_resized = cv2.resize(mask, (target_w, target_h))
        
        if self.blend_mode == 'replace':
            result[y1:y2, x1:x2] = styled_resized
        elif self.blend_mode == 'alpha':
            # Alpha blending
            mask_3d = mask_resized[:, :, np.newaxis] * self.alpha
            result[y1:y2, x1:x2] = (
                mask_3d * styled_resized.astype(np.float32) +
                (1 - mask_3d) * result[y1:y2, x1:x2].astype(np.float32)
            ).astype(np.uint8)
        
        return result
    
    def __call__(
        self,
        image: np.ndarray,
        annotations: List[Dict],
        image_cache: Optional[Dict[str, np.ndarray]] = None,
    ) -> np.ndarray:
        """Apply object style swap augmentation.
        
        Args:
            image: Input image (H, W, C).
            annotations: List of annotation dicts with 'category_id' and 'bbox'.
            image_cache: Optional cache of loaded images.
            
        Returns:
            Augmented image with swapped object styles.
        """
        if len(self.category_index) == 0:
            return image
        
        result = image.copy()
        image_cache = image_cache or {}
        
        for ann in annotations:
            if random.random() > self.p:
                continue
            
            category_id = ann.get('category_id')
            bbox = ann.get('bbox')
            
            if category_id is None or bbox is None:
                continue
            
            if category_id not in self.category_index:
                continue
            
            # Get candidates of same category
            candidates = self.category_index[category_id]
            if len(candidates) == 0:
                continue
            
            # Random sample a reference object
            ref_path, ref_bbox, ref_mask = random.choice(candidates)
            
            # Load reference image
            if ref_path in image_cache:
                ref_image = image_cache[ref_path]
            else:
                try:
                    ref_image = cv2.imread(ref_path)
                    if ref_image is None:
                        continue
                    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
                    image_cache[ref_path] = ref_image
                except Exception:
                    continue
            
            # Extract reference region
            ref_region, ref_soft_mask = self._extract_object_region(ref_image, ref_bbox, ref_mask)
            
            if ref_region.size == 0:
                continue
            
            # Extract source region
            source_region, source_soft_mask = self._extract_object_region(result, bbox, ann.get('segmentation'))
            
            if source_region.size == 0:
                continue
            
            # Check minimum area
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) if len(bbox) == 4 else 0
            if area < self.min_area:
                continue
            
            # Apply histogram matching
            styled_region = self._histogram_matching_region(source_region, ref_region, source_soft_mask)
            
            # Blend back
            result = self._blend_region(result, styled_region, bbox, source_soft_mask)
        
        return result


class DGFSDR:
    """Frequency Space Domain Randomization for Domain Generalization.
    
    Uses DCT (Discrete Cosine Transform) to decompose images into frequency
    components and applies histogram matching on select frequency bands while
    preserving others. This simulates domain shift while maintaining semantic
    content.
    
    Key idea: Low frequencies encode semantic structure, high frequencies encode
    texture/style. By randomizing certain frequency bands, we can create diverse
    domain shifts while preserving object structure.
    
    Args:
        p (float): Probability of applying transform. Default: 0.5.
        variant_bands (List[Tuple[int, int]]): Frequency bands to randomize.
            Default: [(0, 2), (32, 64)] (very low and high frequencies).
        block_size (int): DCT block size. Default: 64.
    """
    
    def __init__(
        self,
        p: float = 0.5,
        variant_bands: Optional[List[Tuple[int, int]]] = None,
        block_size: int = 64,
    ):
        self.p = p
        self.variant_bands = variant_bands or [(0, 2), (32, 64)]
        self.block_size = block_size
        
        # Reference images for histogram matching
        self.reference_images: List[str] = []
    
    def set_reference_images(self, image_paths: List[str]) -> None:
        """Set pool of reference images for histogram matching."""
        self.reference_images = image_paths
        print(f"[DGFSDR] Set {len(image_paths)} reference images")
    
    def _dct2(self, block: np.ndarray) -> np.ndarray:
        """2D DCT on a block."""
        try:
            from scipy.fftpack import dct
            return dct(dct(block.T, norm='ortho').T, norm='ortho')
        except ImportError:
            # Fallback to OpenCV DCT
            return cv2.dct(block.astype(np.float32))
    
    def _idct2(self, block: np.ndarray) -> np.ndarray:
        """2D inverse DCT on a block."""
        try:
            from scipy.fftpack import idct
            return idct(idct(block.T, norm='ortho').T, norm='ortho')
        except ImportError:
            # Fallback to OpenCV IDCT  
            return cv2.idct(block.astype(np.float32))
    
    def _histogram_match_channel(
        self,
        source: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        """Apply histogram matching to a single channel."""
        s_hist, s_bins = np.histogram(source.flatten(), bins=256, range=(-1000, 1000), density=True)
        r_hist, r_bins = np.histogram(reference.flatten(), bins=256, range=(-1000, 1000), density=True)
        
        s_cdf = np.cumsum(s_hist)
        s_cdf = s_cdf / s_cdf[-1] if s_cdf[-1] > 0 else s_cdf
        
        r_cdf = np.cumsum(r_hist)
        r_cdf = r_cdf / r_cdf[-1] if r_cdf[-1] > 0 else r_cdf
        
        # Build lookup
        result = np.interp(source.flatten(), s_bins[:-1], np.interp(s_cdf, r_cdf, r_bins[:-1]))
        return result.reshape(source.shape)
    
    def __call__(
        self,
        image: np.ndarray,
        reference_image: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Apply FSDR transformation.
        
        Args:
            image: Input image (H, W, C).
            reference_image: Optional reference image for histogram matching.
                If not provided, randomly samples from reference pool.
            
        Returns:
            Transformed image with randomized frequency components.
        """
        if random.random() > self.p:
            return image
        
        # Load reference image if needed
        if reference_image is None:
            if len(self.reference_images) > 0:
                ref_path = random.choice(self.reference_images)
                try:
                    reference_image = cv2.imread(ref_path)
                    if reference_image is not None:
                        reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)
                except Exception:
                    return image
            else:
                return image
        
        if reference_image is None:
            return image
        
        result = image.astype(np.float32)
        ref = reference_image.astype(np.float32)
        
        h, w, c = result.shape
        ref_h, ref_w = ref.shape[:2]
        
        # Process each channel
        for ch in range(c):
            src_ch = result[:, :, ch]
            ref_ch = ref[:, :, ch] if ch < ref.shape[2] else ref[:, :, 0]
            
            # Resize reference to match source
            if ref_ch.shape != src_ch.shape:
                ref_ch = cv2.resize(ref_ch, (w, h))
            
            # Apply DCT
            src_dct = self._dct2(src_ch)
            ref_dct = self._dct2(ref_ch)
            
            # Histogram match on variant bands
            for band_start, band_end in self.variant_bands:
                band_end = min(band_end, min(h, w))
                if band_start >= band_end:
                    continue
                
                src_band = src_dct[band_start:band_end, band_start:band_end]
                ref_band = ref_dct[band_start:band_end, band_start:band_end]
                
                if src_band.size == 0 or ref_band.size == 0:
                    continue
                
                # Histogram match the band
                matched_band = self._histogram_match_channel(src_band, ref_band)
                src_dct[band_start:band_end, band_start:band_end] = matched_band
            
            # Inverse DCT
            result[:, :, ch] = self._idct2(src_dct)
        
        return np.clip(result, 0, 255).astype(np.uint8)


class DataAugmentationDG(T.Augmentation):
    """Detectron2-compatible wrapper for DG augmentations.
    
    Combines ObjectStyleSwap and DGFSDR for domain generalization.
    
    Args:
        use_object_style_swap (bool): Enable object style swap. Default: True.
        use_fsdr (bool): Enable FSDR. Default: True.
        object_swap_prob (float): Probability for object swap. Default: 0.3.
        fsdr_prob (float): Probability for FSDR. Default: 0.5.
    """
    
    def __init__(
        self,
        use_object_style_swap: bool = True,
        use_fsdr: bool = True,
        object_swap_prob: float = 0.3,
        fsdr_prob: float = 0.5,
        **kwargs
    ):
        super().__init__()
        
        self.object_style_swap = ObjectStyleSwap(p=object_swap_prob) if use_object_style_swap else None
        self.fsdr = DGFSDR(p=fsdr_prob) if use_fsdr else None
        
    def get_transform(self, image: np.ndarray) -> T.Transform:
        """Returns transform to apply."""
        return DGTransform(
            object_style_swap=self.object_style_swap,
            fsdr=self.fsdr,
        )


class DGTransform(T.Transform):
    """Transform wrapper for DG augmentations."""
    
    def __init__(
        self,
        object_style_swap: Optional[ObjectStyleSwap] = None,
        fsdr: Optional[DGFSDR] = None,
    ):
        super().__init__()
        self.object_style_swap = object_style_swap
        self.fsdr = fsdr
    
    def apply_image(self, img: np.ndarray) -> np.ndarray:
        """Apply transform to image."""
        if self.fsdr is not None:
            img = self.fsdr(img)
        return img
    
    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        """Coordinates unchanged by style transfer."""
        return coords
