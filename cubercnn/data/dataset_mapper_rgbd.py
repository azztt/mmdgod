# Copyright (c) Meta Platforms, Inc. and affiliates
# Modified for RGB-D support
"""
DatasetMapper3D with RGB-D support for domain generalization.

This extends the base DatasetMapper3D to:
1. Load depth maps from file alongside RGB images
2. Apply synchronized augmentations to both RGB and depth
3. Support domain generalization transforms (FSDR, etc.)
"""
import copy
import os.path as osp
import torch
import numpy as np
from PIL import Image
from detectron2.structures import BoxMode, Keypoints
from detectron2.data import detection_utils
from detectron2.data import transforms as T
from detectron2.data import DatasetMapper
from detectron2.structures import Boxes, BoxMode, Instances

from .dataset_mapper import transform_instance_annotations, annotations_to_instances


class DatasetMapper3D_RGBD(DatasetMapper):
    """Dataset mapper that loads both RGB and depth images.
    
    Expects dataset_dict to contain:
        - file_name: Path to RGB image
        - depth_file_name: Path to depth map (.npy, .png, .exr)
        - K: Camera intrinsics matrix (3x3)
        - annotations: List of 3D bounding box annotations
    
    Args:
        cfg: Detectron2 config
        is_train: Whether in training mode
        augmentations: List of augmentation transforms
        depth_max: Maximum depth value for normalization (default: 8.0)
        depth_normalize: Whether to normalize depth to [-1, 1] (default: True)
        depth_norm_mode: Normalization mode - "fixed", "per_sample", or "percentile"
    """
    
    def __init__(
        self,
        cfg,
        is_train: bool = True,
        augmentations: list = None,
        depth_max: float = 8.0,
        depth_normalize: bool = True,
        depth_norm_mode: str = "fixed",
        depth_norm_percentile: float = 95,
    ):
        # Build default augmentations if none provided
        if augmentations is None:
            augmentations = self._build_default_augmentations(cfg, is_train)
        
        super().__init__(
            is_train=is_train,
            augmentations=augmentations,
            image_format=cfg.INPUT.FORMAT,
            use_instance_mask=False,
            use_keypoint=False,
        )
        
        self.depth_max = depth_max
        self.depth_normalize = depth_normalize
        self.depth_norm_mode = depth_norm_mode
        self.depth_norm_percentile = depth_norm_percentile
        self.cfg = cfg
        
        # Dataset-specific unknown categories
        if hasattr(cfg, 'DATASETS') and hasattr(cfg.DATASETS, 'DATASET_ID_TO_UNKNOWN_CATS'):
            self.dataset_id_to_unknown_cats = cfg.DATASETS.DATASET_ID_TO_UNKNOWN_CATS
        else:
            self.dataset_id_to_unknown_cats = {}
    
    @staticmethod
    def _build_default_augmentations(cfg, is_train: bool):
        """Build default augmentations for RGB-D data.
        
        Args:
            cfg: Detectron2 config
            is_train: Whether in training mode
            
        Returns:
            List of augmentation transforms
        """
        import detectron2.data.transforms as T
        
        if is_train:
            min_size = cfg.INPUT.MIN_SIZE_TRAIN
            max_size = cfg.INPUT.MAX_SIZE_TRAIN
            sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING if hasattr(cfg.INPUT, 'MIN_SIZE_TRAIN_SAMPLING') else "choice"
            
            augmentations = [
                T.ResizeShortestEdge(min_size, max_size, sample_style),
            ]
            
            # Add horizontal flip if enabled
            if hasattr(cfg.INPUT, 'RANDOM_FLIP') and cfg.INPUT.RANDOM_FLIP != "none":
                augmentations.append(
                    T.RandomFlip(
                        horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                        vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                    )
                )
        else:
            min_size = cfg.INPUT.MIN_SIZE_TEST
            max_size = cfg.INPUT.MAX_SIZE_TEST
            augmentations = [
                T.ResizeShortestEdge(min_size, max_size, "choice"),
            ]
        
        return augmentations
    
    def _load_depth(self, depth_path: str) -> np.ndarray:
        """Load depth map from file.
        
        Supports:
            - .npy: Numpy array files
            - .npz: Compressed numpy files
            - .png/.jpg: Image files (depth encoded as uint16 or grayscale)
            - .exr: OpenEXR files
        
        Args:
            depth_path: Path to depth file
            
        Returns:
            depth: (H, W) numpy array of depth values in meters
        """
        ext = osp.splitext(depth_path)[1].lower()
        
        if ext == '.npy':
            depth = np.load(depth_path)
        elif ext == '.npz':
            data = np.load(depth_path)
            # Try common keys
            for key in ['depth', 'arr_0', 'data']:
                if key in data:
                    depth = data[key]
                    break
            else:
                depth = data[list(data.keys())[0]]
        elif ext == '.exr':
            try:
                import cv2
                depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
                if depth is not None and len(depth.shape) == 3:
                    depth = depth[:, :, 0]
            except ImportError:
                raise ImportError("OpenCV required for .exr depth files")
        else:
            # Image file - common for SUNRGBD
            depth_img = Image.open(depth_path)
            depth = np.array(depth_img, dtype=np.float32)
            
            # Handle uint16 encoding (SUNRGBD uses depth_shift of 1000)
            if depth_img.mode == 'I;16' or depth.dtype == np.uint16:
                depth = depth / 1000.0  # Convert mm to meters
            elif depth.max() > 100:
                # Likely encoded in mm
                depth = depth / 1000.0
        
        # Ensure float32
        depth = depth.astype(np.float32)
        
        # Handle NaN/Inf values
        depth = np.nan_to_num(depth, nan=0.0, posinf=self.depth_max, neginf=0.0)
        
        return depth
    
    def _process_depth(self, depth: np.ndarray) -> tuple:
        """Process depth map: clip, normalize, and convert to tensor format.
        
        Supports multiple normalization modes for domain generalization:
        - "fixed": Use fixed depth_max to normalize (default, domain-specific)
        - "per_sample": Z-score normalization per image (domain-invariant)
        - "percentile": Normalize by per-image percentile (robust to outliers)
        
        Args:
            depth: (H, W) depth array in meters
            
        Returns:
            depth: (1, H, W) processed depth tensor-ready array
            depth_stats: dict with normalization stats (for inverse transform)
        """
        # Get valid depth mask (non-zero)
        valid_mask = depth > 0.01  # Ignore very small values as invalid
        
        depth_stats = {}
        
        if self.depth_norm_mode == "per_sample":
            # Z-score normalization: (x - mean) / std
            # This makes the depth distribution domain-invariant
            if valid_mask.sum() > 100:  # Need enough valid pixels
                depth_mean = depth[valid_mask].mean()
                depth_std = depth[valid_mask].std()
                if depth_std < 0.01:  # Avoid division by zero
                    depth_std = 1.0
            else:
                depth_mean = self.depth_max / 2.0
                depth_std = self.depth_max / 4.0
            
            depth_stats['mean'] = depth_mean
            depth_stats['std'] = depth_std
            depth_stats['mode'] = 'per_sample'
            
            # Normalize: z-score then scale to [-1, 1] range
            # Assuming 3 std covers most values
            depth_normalized = (depth - depth_mean) / depth_std
            depth_normalized = np.clip(depth_normalized / 3.0, -1.0, 1.0)
            depth = depth_normalized
            
        elif self.depth_norm_mode == "percentile":
            # Percentile normalization: robust to outliers
            if valid_mask.sum() > 100:
                depth_p95 = np.percentile(depth[valid_mask], self.depth_norm_percentile)
                depth_p5 = np.percentile(depth[valid_mask], 100 - self.depth_norm_percentile)
            else:
                depth_p95 = self.depth_max
                depth_p5 = 0.0
            
            depth_stats['p5'] = depth_p5
            depth_stats['p95'] = depth_p95
            depth_stats['mode'] = 'percentile'
            
            # Clip and normalize to [-1, 1]
            depth = np.clip(depth, depth_p5, depth_p95)
            if depth_p95 - depth_p5 > 0.01:
                depth = (depth - depth_p5) / (depth_p95 - depth_p5) * 2.0 - 1.0
            else:
                depth = np.zeros_like(depth)
                
        else:  # "fixed" mode (default)
            depth_stats['max'] = self.depth_max
            depth_stats['mode'] = 'fixed'
            
            # Clip to max depth
            depth = np.clip(depth, 0.0, self.depth_max)
            
            # Normalize to [-1, 1] if requested
            if self.depth_normalize:
                depth = (depth / self.depth_max) * 2.0 - 1.0
        
        # Add channel dimension: (H, W) -> (1, H, W)
        depth = depth[np.newaxis, :, :]
        
        return depth, depth_stats
    
    def _apply_geometric_transforms_to_depth(
        self, 
        depth: np.ndarray, 
        transforms: T.TransformList
    ) -> np.ndarray:
        """Apply geometric transforms (flip, resize, crop) to depth map.
        
        Args:
            depth: (H, W) or (1, H, W) depth array
            transforms: List of transforms applied to RGB
            
        Returns:
            depth: Transformed depth array
        """
        # Handle channel dimension
        had_channel = False
        if depth.ndim == 3:
            depth = depth[0]  # (1, H, W) -> (H, W)
            had_channel = True
        
        for transform in transforms:
            if isinstance(transform, T.HFlipTransform):
                depth = depth[:, ::-1].copy()
            elif isinstance(transform, T.VFlipTransform):
                depth = depth[::-1, :].copy()
            elif isinstance(transform, T.ResizeTransform):
                # Use nearest interpolation for depth to avoid artifacts
                from PIL import Image
                depth_img = Image.fromarray(depth, mode='F')
                depth_img = depth_img.resize(
                    (transform.new_w, transform.new_h), 
                    Image.Resampling.NEAREST
                )
                depth = np.array(depth_img)
            elif isinstance(transform, T.CropTransform):
                depth = depth[
                    transform.y0:transform.y0 + transform.h,
                    transform.x0:transform.x0 + transform.w
                ]
        
        if had_channel:
            depth = depth[np.newaxis, :, :]
        
        return depth

    def __call__(self, dataset_dict):
        """Process a single sample, loading both RGB and depth.
        
        Args:
            dataset_dict: Dictionary with sample info including:
                - file_name: RGB image path
                - depth_file_name: Depth map path
                - K: Camera intrinsics
                - annotations: 3D box annotations
                
        Returns:
            dataset_dict: Processed dictionary with 'image' and 'depth' tensors
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        
        # Load RGB image
        image = detection_utils.read_image(
            dataset_dict["file_name"], 
            format=self.image_format
        )
        detection_utils.check_image_size(dataset_dict, image)
        
        # Load depth if available
        depth = None
        if "depth_file_name" in dataset_dict and dataset_dict["depth_file_name"]:
            depth_path = dataset_dict["depth_file_name"]
            try:
                depth = self._load_depth(depth_path)
            except Exception as e:
                print(f"Warning: Failed to load depth from {depth_path}: {e}")
                depth = None
        
        # If no depth available, create dummy depth (all zeros)
        if depth is None:
            depth = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Apply augmentations to RGB
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        
        # Apply same geometric transforms to depth
        depth = self._apply_geometric_transforms_to_depth(depth, transforms)
        
        # Process depth (clip, normalize, add channel)
        depth, depth_stats = self._process_depth(depth if depth.ndim == 2 else depth[0])
        
        image_shape = image.shape[:2]  # h, w
        
        # Convert to tensors
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        dataset_dict["depth"] = torch.as_tensor(
            np.ascontiguousarray(depth)
        )
        
        # Store depth normalization stats for potential inverse transform
        dataset_dict["depth_stats"] = depth_stats
        
        # At inference, no need for additional processing
        if not self.is_train:
            return dataset_dict
        
        # Process annotations
        if "annotations" in dataset_dict:
            dataset_id = dataset_dict.get('dataset_id', 0)
            K = np.array(dataset_dict['K'])
            
            unknown_categories = self.dataset_id_to_unknown_cats.get(dataset_id, set())
            
            # Transform and filter annotations
            annos = [
                transform_instance_annotations(obj, transforms, K=K)
                for obj in dataset_dict.pop("annotations") 
                if obj.get("iscrowd", 0) == 0
            ]
            
            # Convert to instance format
            instances = annotations_to_instances(annos, image_shape, unknown_categories)
            dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
        
        return dataset_dict


class DatasetMapper3D_RGBD_DG(DatasetMapper3D_RGBD):
    """Extended mapper with domain generalization augmentations.
    
    Adds domain generalization specific augmentations:
    - FSDR: Frequency Space Domain Randomization
    - Object Style Swap: Object-wise appearance transfer
    - Photometric jitter (ColorJitter)
    - Gaussian noise (RGB and depth)
    - Depth noise/dropout
    - Scene-aware augmentations
    
    Args:
        cfg: Detectron2 config
        is_train: Whether in training mode
        augmentations: List of augmentation transforms
        depth_max: Maximum depth value
        depth_normalize: Whether to normalize depth
        depth_norm_mode: Normalization mode - "fixed", "per_sample", or "percentile"
        depth_norm_percentile: Percentile for percentile normalization
    """
    
    def __init__(
        self,
        cfg,
        is_train: bool = True,
        augmentations: list = None,
        depth_max: float = 8.0,
        depth_normalize: bool = True,
        depth_norm_mode: str = "fixed",
        depth_norm_percentile: float = 95,
    ):
        super().__init__(
            cfg=cfg,
            is_train=is_train,
            augmentations=augmentations,
            depth_max=depth_max,
            depth_normalize=depth_normalize,
            depth_norm_mode=depth_norm_mode,
            depth_norm_percentile=depth_norm_percentile,
        )
        
        # Read augmentation settings from config
        aug_cfg = getattr(cfg, 'AUG', None)
        dg_cfg = getattr(cfg, 'DG', None)
        
        # FSDR settings
        self.use_fsdr = False
        self.fsdr_prob = 0.5
        if dg_cfg is not None:
            fsdr_cfg = getattr(dg_cfg, 'FSDR', None)
            if fsdr_cfg is not None:
                self.use_fsdr = getattr(fsdr_cfg, 'ENABLED', False)
                self.fsdr_prob = getattr(fsdr_cfg, 'PROBABILITY', 0.5)
        
        # Object Style Swap settings
        self.use_object_style_swap = False
        self.object_style_swap_prob = 0.3
        if dg_cfg is not None:
            oss_cfg = getattr(dg_cfg, 'OBJECT_STYLE_SWAP', None)
            if oss_cfg is not None:
                self.use_object_style_swap = getattr(oss_cfg, 'ENABLED', False)
        
        # Color jitter settings
        self.use_photometric = False
        self.photometric_prob = 0.8
        self.color_jitter_params = {'brightness': 0.4, 'contrast': 0.4, 'saturation': 0.4, 'hue': 0.1}
        if aug_cfg is not None:
            cj_cfg = getattr(aug_cfg, 'COLOR_JITTER', None)
            if cj_cfg is not None:
                self.use_photometric = getattr(cj_cfg, 'ENABLED', False)
                self.photometric_prob = getattr(cj_cfg, 'PROBABILITY', 0.8)
                self.color_jitter_params = {
                    'brightness': getattr(cj_cfg, 'BRIGHTNESS', 0.4),
                    'contrast': getattr(cj_cfg, 'CONTRAST', 0.4),
                    'saturation': getattr(cj_cfg, 'SATURATION', 0.4),
                    'hue': getattr(cj_cfg, 'HUE', 0.1),
                }
        
        # Gaussian noise settings (RGB)
        self.use_gaussian_noise = False
        self.gaussian_noise_prob = 0.5
        self.gaussian_noise_std_range = [0.01, 0.05]
        if aug_cfg is not None:
            gn_cfg = getattr(aug_cfg, 'GAUSSIAN_NOISE', None)
            if gn_cfg is not None:
                self.use_gaussian_noise = getattr(gn_cfg, 'ENABLED', False)
                self.gaussian_noise_prob = getattr(gn_cfg, 'PROBABILITY', 0.5)
                self.gaussian_noise_std_range = list(getattr(gn_cfg, 'STD_RANGE', [0.01, 0.05]))
        
        # Depth dropout settings
        self.use_depth_dropout = False
        self.depth_dropout_prob = 0.3
        self.depth_dropout_num_drops = [1, 5]
        self.depth_dropout_size = [0.02, 0.1]
        if aug_cfg is not None:
            dd_cfg = getattr(aug_cfg, 'DEPTH_DROPOUT', None)
            if dd_cfg is not None:
                self.use_depth_dropout = getattr(dd_cfg, 'ENABLED', False)
                self.depth_dropout_prob = getattr(dd_cfg, 'PROBABILITY', 0.3)
                self.depth_dropout_num_drops = list(getattr(dd_cfg, 'NUM_DROPS', [1, 5]))
                self.depth_dropout_size = list(getattr(dd_cfg, 'DROP_SIZE', [0.02, 0.1]))
        
        # Depth noise settings
        self.use_depth_noise = False
        self.depth_noise_prob = 0.5
        self.depth_noise_std_range = [0.01, 0.03]
        if aug_cfg is not None:
            dn_cfg = getattr(aug_cfg, 'DEPTH_NOISE', None)
            if dn_cfg is not None:
                self.use_depth_noise = getattr(dn_cfg, 'ENABLED', False)
                self.depth_noise_prob = getattr(dn_cfg, 'PROBABILITY', 0.5)
                self.depth_noise_std_range = list(getattr(dn_cfg, 'STD_RANGE', [0.01, 0.03]))
        
        # Initialize Object Style Swap if enabled
        self.object_style_swap = None
        if self.use_object_style_swap:
            try:
                from .dg_augmentations import ObjectStyleSwap
                self.object_style_swap = ObjectStyleSwap(
                    p=self.object_style_swap_prob,
                    blend_mode='alpha',
                    alpha=0.7,
                )
            except ImportError:
                print("Warning: Could not import ObjectStyleSwap augmentation")
        
        # Initialize FSDR if enabled
        self.fsdr = None
        if self.use_fsdr:
            try:
                from .dg_augmentations import DGFSDR
                self.fsdr = DGFSDR(p=self.fsdr_prob)
            except ImportError:
                pass  # Fall back to inline implementation
        
        # Initialize color jitter transform
        if self.use_photometric:
            import torchvision.transforms as TV
            self.color_jitter = TV.ColorJitter(**self.color_jitter_params)
    
    def build_object_style_index(self, dataset_dicts):
        """Build category index for object style swap from dataset.
        
        Call this after loading dataset to enable object style swap.
        
        Args:
            dataset_dicts: List of dataset annotation dictionaries
        """
        if self.object_style_swap is not None:
            self.object_style_swap.build_category_index(dataset_dicts)
    
    def set_fsdr_references(self, image_paths):
        """Set reference images for FSDR histogram matching.
        
        Args:
            image_paths: List of paths to reference images
        """
        if self.fsdr is not None:
            self.fsdr.set_reference_images(image_paths)
    
    def _apply_fsdr(self, image: np.ndarray, scene_type: str = None) -> np.ndarray:
        """Apply Frequency Space Domain Randomization.
        
        FSDR decomposes an image into frequency bands and randomly perturbs
        the amplitude and phase to simulate domain shift while preserving
        semantic content.
        
        Args:
            image: (H, W, C) RGB image
            scene_type: Optional scene type for scene-aware FSDR
            
        Returns:
            Augmented image
        """
        import numpy as np
        
        # Convert to float if needed
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            convert_back = True
        else:
            convert_back = False
        
        result = np.zeros_like(image)
        
        for c in range(image.shape[2]):
            channel = image[:, :, c]
            
            # FFT
            f_transform = np.fft.fft2(channel)
            f_shift = np.fft.fftshift(f_transform)
            
            # Get amplitude and phase
            amplitude = np.abs(f_shift)
            phase = np.angle(f_shift)
            
            # Perturb amplitude (log-scale)
            amplitude_log = np.log(amplitude + 1e-8)
            
            # Random perturbation factors
            amp_scale = np.random.uniform(0.8, 1.2)
            amp_shift = np.random.uniform(-0.1, 0.1)
            
            amplitude_log = amplitude_log * amp_scale + amp_shift
            amplitude = np.exp(amplitude_log)
            
            # Perturb phase slightly (preserve structure)
            phase_noise = np.random.uniform(-0.1, 0.1, phase.shape)
            phase = phase + phase_noise
            
            # Reconstruct
            f_shift = amplitude * np.exp(1j * phase)
            f_ishift = np.fft.ifftshift(f_shift)
            reconstructed = np.fft.ifft2(f_ishift)
            result[:, :, c] = np.real(reconstructed)
        
        # Clip to valid range
        result = np.clip(result, 0, 1)
        
        if convert_back:
            result = (result * 255).astype(np.uint8)
        
        return result
    
    def _apply_photometric_jitter(self, image: np.ndarray) -> np.ndarray:
        """Apply photometric jitter to RGB image.
        
        Args:
            image: (H, W, C) RGB image
            
        Returns:
            Jittered image
        """
        # Convert to PIL for torchvision transforms
        pil_img = Image.fromarray(image)
        jittered = self.color_jitter(pil_img)
        return np.array(jittered)
    
    def _apply_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise to RGB image.
        
        Args:
            image: (H, W, C) RGB image (uint8 or float)
            
        Returns:
            Noisy image
        """
        # Convert to float if needed
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            convert_back = True
        else:
            convert_back = False
        
        # Random std from range
        std = np.random.uniform(self.gaussian_noise_std_range[0], self.gaussian_noise_std_range[1])
        noise = np.random.randn(*image.shape).astype(np.float32) * std
        
        noisy_image = np.clip(image + noise, 0, 1)
        
        if convert_back:
            noisy_image = (noisy_image * 255).astype(np.uint8)
        
        return noisy_image
    
    def _apply_depth_noise(self, depth: np.ndarray) -> np.ndarray:
        """Apply Gaussian noise to depth map.
        
        Args:
            depth: (H, W) or (1, H, W) depth array
            
        Returns:
            Noisy depth
        """
        # Handle channel dimension
        if depth.ndim == 3:
            depth = depth[0]
            add_channel = True
        else:
            add_channel = False
        
        depth = depth.copy()
        
        # Random std from range
        std = np.random.uniform(self.depth_noise_std_range[0], self.depth_noise_std_range[1])
        
        # Only add noise where depth is valid
        valid_mask = depth > 0
        noise = np.random.randn(*depth.shape).astype(np.float32) * std
        depth[valid_mask] = np.maximum(0, depth[valid_mask] + noise[valid_mask])
        
        if add_channel:
            depth = depth[np.newaxis, :, :]
        
        return depth
    
    def _apply_depth_dropout(self, depth: np.ndarray) -> np.ndarray:
        """Apply random dropout to depth map.
        
        Simulates missing depth values common in real depth sensors.
        
        Args:
            depth: (H, W) or (1, H, W) depth array
            
        Returns:
            Depth with random regions zeroed out
        """
        # Handle channel dimension
        if depth.ndim == 3:
            depth = depth[0]
            add_channel = True
        else:
            add_channel = False
        
        depth = depth.copy()
        H, W = depth.shape
        
        # Random rectangles dropout
        num_drops = np.random.randint(self.depth_dropout_num_drops[0], self.depth_dropout_num_drops[1] + 1)
        min_size, max_size = self.depth_dropout_size
        
        for _ in range(num_drops):
            # Random rectangle size
            drop_h = np.random.randint(int(H * min_size), max(int(H * max_size), int(H * min_size) + 1))
            drop_w = np.random.randint(int(W * min_size), max(int(W * max_size), int(W * min_size) + 1))
            
            # Random position
            y = np.random.randint(0, max(H - drop_h, 1))
            x = np.random.randint(0, max(W - drop_w, 1))
            
            # Zero out region
            depth[y:y+drop_h, x:x+drop_w] = 0.0
        
        if add_channel:
            depth = depth[np.newaxis, :, :]
        
        return depth
    
    def __call__(self, dataset_dict):
        """Process sample with domain generalization augmentations."""
        dataset_dict = copy.deepcopy(dataset_dict)
        
        # Load RGB image
        image = detection_utils.read_image(
            dataset_dict["file_name"], 
            format=self.image_format
        )
        detection_utils.check_image_size(dataset_dict, image)
        
        # Load depth if available
        depth = None
        if "depth_file_name" in dataset_dict and dataset_dict["depth_file_name"]:
            try:
                depth = self._load_depth(dataset_dict["depth_file_name"])
            except Exception as e:
                print(f"Warning: Failed to load depth: {e}")
        
        if depth is None:
            depth = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # Apply domain generalization augmentations (training only)
        if self.is_train:
            # Object Style Swap (before geometric augmentations)
            if self.object_style_swap is not None and "annotations" in dataset_dict:
                image = self.object_style_swap(image, dataset_dict["annotations"])
            
            # FSDR augmentation
            if self.use_fsdr:
                if self.fsdr is not None:
                    image = self.fsdr(image)
                elif np.random.random() < self.fsdr_prob:
                    scene_type = dataset_dict.get('scene_type', None)
                    image = self._apply_fsdr(image, scene_type)
            
            # Photometric jitter (color jitter)
            if self.use_photometric and np.random.random() < self.photometric_prob:
                image = self._apply_photometric_jitter(image)
            
            # Gaussian noise on RGB
            if self.use_gaussian_noise and np.random.random() < self.gaussian_noise_prob:
                image = self._apply_gaussian_noise(image)
        
        # Apply standard geometric augmentations
        aug_input = T.AugInput(image)
        transforms = self.augmentations(aug_input)
        image = aug_input.image
        
        # Apply same transforms to depth
        depth = self._apply_geometric_transforms_to_depth(depth, transforms)
        
        # Domain generalization augmentations for depth (training only)
        if self.is_train:
            # Depth dropout
            if self.use_depth_dropout and np.random.random() < self.depth_dropout_prob:
                depth = self._apply_depth_dropout(depth)
            
            # Depth Gaussian noise
            if self.use_depth_noise and np.random.random() < self.depth_noise_prob:
                depth = self._apply_depth_noise(depth)
        
        # Process depth (clip, normalize)
        depth, depth_stats = self._process_depth(depth if depth.ndim == 2 else depth[0])
        
        image_shape = image.shape[:2]
        
        # Convert to tensors
        dataset_dict["image"] = torch.as_tensor(
            np.ascontiguousarray(image.transpose(2, 0, 1))
        )
        dataset_dict["depth"] = torch.as_tensor(
            np.ascontiguousarray(depth)
        )
        
        # Store depth normalization stats for potential inverse transform
        dataset_dict["depth_stats"] = depth_stats
        
        if not self.is_train:
            return dataset_dict
        
        # Process annotations
        if "annotations" in dataset_dict:
            dataset_id = dataset_dict.get('dataset_id', 0)
            K = np.array(dataset_dict['K'])
            unknown_categories = self.dataset_id_to_unknown_cats.get(dataset_id, set())
            
            annos = [
                transform_instance_annotations(obj, transforms, K=K)
                for obj in dataset_dict.pop("annotations") 
                if obj.get("iscrowd", 0) == 0
            ]
            
            instances = annotations_to_instances(annos, image_shape, unknown_categories)
            dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
        
        return dataset_dict
