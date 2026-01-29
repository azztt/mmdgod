# Copyright (c) Meta Platforms, Inc. and affiliates
# Extension for RGB-D domain-generalized 3D detection
"""
RGBD Dataset Registration Module

This module handles registration and loading of RGB-D datasets for domain-generalized
3D object detection. It extends the standard Omni3D data loading to include depth
information from manifests that have `depth_file_path` field.

Manifest format (COCO-style):
- images: [{file_path, depth_file_path, K, width, height, scene_type, ...}]
- annotations: [{image_id, category_id, bbox_3d, ...}]
- categories: [{id, name}]
"""

import json
import os
import logging
import contextlib
import io
import numpy as np
from typing import List, Dict, Any, Optional

from pycocotools.coco import COCO
from fvcore.common.timer import Timer
from detectron2.utils.file_io import PathManager
from detectron2.structures import BoxMode
from detectron2.data import MetadataCatalog, DatasetCatalog

from cubercnn import util
from cubercnn.data.datasets import is_ignore, get_filter_settings_from_cfg

logger = logging.getLogger(__name__)

# =============================================================================
# Dataset Registration Functions
# =============================================================================

def register_rgbd_datasets(cfg):
    """
    Register RGB-D datasets based on config.
    
    This should be called at the start of training to register all datasets
    specified in the config. The datasets use manifest files that include
    depth_file_path for each image.
    
    Args:
        cfg: Config with DATASETS section containing:
            - DATA_ROOT: Root directory for dataset files
            - MANIFEST_ROOT: Directory containing manifest JSON files
            - TRAIN, VAL, TEST: Tuples of dataset names
    """
    data_root = cfg.DATASETS.DATA_ROOT
    manifest_root = cfg.DATASETS.MANIFEST_ROOT
    
    filter_settings = get_filter_settings_from_cfg(cfg)
    
    # Register training datasets
    for dataset_name in cfg.DATASETS.TRAIN:
        if dataset_name not in DatasetCatalog:
            manifest_file = get_manifest_file_for_dataset(dataset_name, cfg)
            if manifest_file and os.path.exists(manifest_file):
                register_single_rgbd_dataset(
                    dataset_name, manifest_file, data_root, filter_settings
                )
                logger.info(f"Registered training dataset: {dataset_name}")
            else:
                logger.warning(f"Manifest not found for training dataset: {dataset_name}")
    
    # Register validation datasets
    for dataset_name in cfg.DATASETS.VAL:
        if dataset_name not in DatasetCatalog:
            manifest_file = get_manifest_file_for_dataset(dataset_name, cfg)
            if manifest_file and os.path.exists(manifest_file):
                register_single_rgbd_dataset(
                    dataset_name, manifest_file, data_root, filter_settings, filter_empty=False
                )
                logger.info(f"Registered validation dataset: {dataset_name}")
            else:
                logger.warning(f"Manifest not found for validation dataset: {dataset_name}")
    
    # Register test datasets  
    for dataset_name in cfg.DATASETS.TEST:
        if dataset_name not in DatasetCatalog:
            manifest_file = get_manifest_file_for_dataset(dataset_name, cfg)
            if manifest_file and os.path.exists(manifest_file):
                register_single_rgbd_dataset(
                    dataset_name, manifest_file, data_root, filter_settings, filter_empty=False
                )
                logger.info(f"Registered test dataset: {dataset_name}")
            else:
                logger.warning(f"Manifest not found for test dataset: {dataset_name}")


def get_manifest_file_for_dataset(dataset_name: str, cfg) -> Optional[str]:
    """
    Get manifest file path for a dataset name.
    
    Handles mapping from dataset names to manifest files. Supports:
    - Direct manifest file specification in config
    - Convention-based naming (dataset_name -> dataset_name.json)
    
    Args:
        dataset_name: Name like "hypersim_train_rgbd", "sunrgbd_val_rgbd"
        cfg: Config with DATASETS section
        
    Returns:
        Full path to manifest JSON file, or None if not found
    """
    manifest_root = cfg.DATASETS.MANIFEST_ROOT
    
    # Check if specific manifest files are provided in config (and not empty)
    if hasattr(cfg.DATASETS, 'TRAIN_MANIFEST') and cfg.DATASETS.TRAIN_MANIFEST and dataset_name in cfg.DATASETS.TRAIN:
        manifest_file = cfg.DATASETS.TRAIN_MANIFEST
        if not os.path.isabs(manifest_file):
            manifest_file = os.path.join(manifest_root, manifest_file)
        return manifest_file
        
    if hasattr(cfg.DATASETS, 'VAL_MANIFEST') and cfg.DATASETS.VAL_MANIFEST and dataset_name in cfg.DATASETS.VAL:
        manifest_file = cfg.DATASETS.VAL_MANIFEST
        if not os.path.isabs(manifest_file):
            manifest_file = os.path.join(manifest_root, manifest_file)
        return manifest_file
        
    if hasattr(cfg.DATASETS, 'TEST_MANIFEST') and cfg.DATASETS.TEST_MANIFEST and dataset_name in cfg.DATASETS.TEST:
        manifest_file = cfg.DATASETS.TEST_MANIFEST
        if not os.path.isabs(manifest_file):
            manifest_file = os.path.join(manifest_root, manifest_file)
        return manifest_file
    
    # Convention-based: convert dataset name to manifest filename
    # e.g., "hypersim_train_rgbd" -> "hypersim_train_filtered.json"
    name_mappings = {
        "hypersim_train_rgbd": "hypersim_train_filtered.json",
        "hypersim_val_rgbd": "hypersim_val_filtered.json",
        "sunrgbd_val_rgbd": "sunrgbd_val_filtered.json",
        "sunrgbd_train_rgbd": "sunrgbd_train_filtered.json",
        "multiscan_val_rgbd": "multiscan_val_filtered.json",
        "scannetpp_val_rgbd": "scannetpp_val_filtered.json",
    }
    
    if dataset_name in name_mappings:
        return os.path.join(manifest_root, name_mappings[dataset_name])
    
    # Fallback: try direct name conversion
    base_name = dataset_name.replace("_rgbd", "_filtered.json")
    potential_path = os.path.join(manifest_root, base_name)
    if os.path.exists(potential_path):
        return potential_path
    
    # Try without _filtered suffix
    base_name = dataset_name.replace("_rgbd", ".json")
    potential_path = os.path.join(manifest_root, base_name)
    if os.path.exists(potential_path):
        return potential_path
    
    logger.warning(f"Could not find manifest for dataset: {dataset_name}")
    return None


def register_single_rgbd_dataset(
    dataset_name: str,
    manifest_file: str,
    data_root: str,
    filter_settings: Dict[str, Any],
    filter_empty: bool = True
):
    """
    Register a single RGB-D dataset.
    
    Args:
        dataset_name: Name to register the dataset under
        manifest_file: Path to manifest JSON file
        data_root: Root directory for image/depth files
        filter_settings: Filtering settings for annotations
        filter_empty: Whether to filter out images without valid annotations
    """
    # Pre-load category info from manifest for metadata
    # This allows accessing metadata before the full dataset is loaded
    try:
        import json
        with open(manifest_file, 'r') as f:
            manifest = json.load(f)
        categories = manifest.get('categories', [])
        cat_ids = sorted([c['id'] for c in categories])
        thing_classes = [c['name'] for c in sorted(categories, key=lambda x: x['id'])]
        id_map = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    except Exception as e:
        logger.warning(f"Could not pre-load categories from {manifest_file}: {e}")
        thing_classes = []
        id_map = {}
    
    DatasetCatalog.register(
        dataset_name,
        lambda mf=manifest_file, dr=data_root, fs=filter_settings, fe=filter_empty: 
            load_rgbd_json(mf, dr, dataset_name, fs, filter_empty=fe)
    )
    
    MetadataCatalog.get(dataset_name).set(
        json_file=manifest_file,
        image_root=data_root,
        evaluator_type="coco",
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=id_map
    )


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_rgbd_json(
    json_file: str,
    image_root: str,
    dataset_name: str,
    filter_settings: Dict[str, Any],
    filter_empty: bool = True
) -> List[Dict[str, Any]]:
    """
    Load RGB-D dataset from JSON manifest.
    
    This extends the standard Omni3D loading to include depth_file_path
    and scene_type information for domain-generalized training.
    
    IMPORTANT: If omni3d_model metadata is already set, we use its category
    mapping to ensure consistency with the model's classifier indices.
    
    Args:
        json_file: Path to manifest JSON
        image_root: Root directory for image files
        dataset_name: Name of the dataset
        filter_settings: Settings for filtering annotations
        filter_empty: Whether to filter images without valid annotations
        
    Returns:
        List of dataset dictionaries with image and annotation info
    """
    timer = Timer()
    json_file = PathManager.get_local_path(json_file)
    
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)
    
    if timer.seconds() > 1:
        logger.info(f"Loading {json_file} takes {timer.seconds():.2f} seconds.")
    
    # Build category mapping from manifest
    cat_ids = sorted(coco_api.getCatIds())
    cats = coco_api.loadCats(cat_ids)
    manifest_cat_id_to_name = {c["id"]: c["name"] for c in cats}
    manifest_thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
    
    # Check if global model metadata is already set (with config order)
    # If so, use it for consistent category mapping
    omni3d_meta = MetadataCatalog.get('omni3d_model')
    if hasattr(omni3d_meta, 'thing_classes') and omni3d_meta.thing_classes:
        # Use global model's category order (from config)
        global_thing_classes = omni3d_meta.thing_classes
        global_name_to_id = {name: i for i, name in enumerate(global_thing_classes)}
        
        # Map manifest category IDs to global contiguous IDs via category names
        id_map = {}
        for cat_id, cat_name in manifest_cat_id_to_name.items():
            if cat_name in global_name_to_id:
                id_map[cat_id] = global_name_to_id[cat_name]
            else:
                logger.debug(f"Category '{cat_name}' (id={cat_id}) not in global model categories")
        
        thing_classes = global_thing_classes
        logger.info(f"Using global model metadata for category mapping ({len(id_map)} mappings)")
    else:
        # Fallback: create local mapping (legacy behavior)
        thing_classes = manifest_thing_classes
        id_map = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    
    # Set metadata for this specific dataset
    meta = MetadataCatalog.get(dataset_name)
    # Only set if not already set (from register_single_rgbd_dataset)
    if not hasattr(meta, 'thing_classes') or not meta.thing_classes:
        meta.thing_classes = thing_classes
        meta.thing_dataset_id_to_contiguous_id = id_map
    
    # Also set global model metadata if not exists or empty
    omni3d_meta = MetadataCatalog.get('omni3d_model')
    existing_classes = omni3d_meta.get('thing_classes', None)
    if not existing_classes:  # None or empty list
        omni3d_meta.thing_classes = thing_classes
        omni3d_meta.thing_dataset_id_to_contiguous_id = id_map
    
    # Sort indices for reproducible results
    img_ids = sorted(coco_api.imgs.keys())
    imgs = coco_api.loadImgs(img_ids)
    anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
    
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(coco_api.anns)
    
    if total_num_valid_anns < total_num_anns:
        logger.info(
            f"{json_file} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )
    
    imgs_anns = list(zip(imgs, anns))
    logger.info(f"Loaded {len(imgs_anns)} images in RGBD format from {json_file}")
    
    dataset_dicts = []
    
    # Annotation keys to pass along (Omni3D format)
    ann_keys = [
        "bbox", "bbox3D_cam", "bbox2D_proj", "bbox2D_trunc", "bbox2D_tight",
        "center_cam", "dimensions", "pose", "R_cam", "category_id",
    ]
    
    invalid_count = 0
    
    for (img_dict, anno_dict_list) in imgs_anns:
        has_valid_annotation = False
        
        record = {}
        
        # RGB image path (required)
        record["file_name"] = os.path.join(image_root, img_dict["file_path"])
        
        # Depth image path (RGBD extension)
        if "depth_file_path" in img_dict:
            record["depth_file_name"] = os.path.join(image_root, img_dict["depth_file_path"])
        else:
            logger.warning(f"Image {img_dict['id']} has no depth_file_path")
            record["depth_file_name"] = None
        
        # Standard image metadata
        record["dataset_id"] = img_dict.get("dataset_id", 0)
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        record["K"] = img_dict["K"]
        record["image_id"] = img_dict["id"]
        
        # Depth dimensions (may differ from RGB)
        record["depth_height"] = img_dict.get("depth_height", img_dict["height"])
        record["depth_width"] = img_dict.get("depth_width", img_dict["width"])
        
        # Scene type for domain-aware augmentations
        record["scene_type"] = img_dict.get("scene_type", "unknown")
        
        # Optional keys
        if "p2" in img_dict:
            record["p2"] = img_dict["p2"]
        
        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == img_dict["id"]
            
            obj = {key: anno[key] for key in ann_keys if key in anno}
            obj["bbox_mode"] = BoxMode.XYWH_ABS
            
            annotation_category_id = obj.get("category_id")
            if annotation_category_id is None:
                continue
            
            # Skip if category not in our mapping and not ignored
            ignore_names = filter_settings.get('ignore_names', [])
            category_name = anno.get('category_name', '')
            
            if annotation_category_id not in id_map and category_name not in ignore_names:
                continue
            
            # Determine if annotation should be ignored
            ignore = is_ignore(anno, filter_settings, img_dict["height"])
            
            # Skip ignored annotations entirely instead of setting category_id = -1
            # This prevents issues with detectron2's histogram printing and loss computation
            if ignore:
                continue
            
            # Skip if category not in mapping
            if annotation_category_id not in id_map:
                continue
            
            obj['iscrowd'] = False
            obj['ignore'] = False  # We've already filtered out ignored ones
            
            # Determine 2D bbox to use (prefer tight, then trunc, then proj)
            if filter_settings.get('modal_2D_boxes') and 'bbox2D_tight' in anno and anno['bbox2D_tight'][0] != -1:
                obj['bbox'] = BoxMode.convert(anno['bbox2D_tight'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            elif filter_settings.get('trunc_2D_boxes') and 'bbox2D_trunc' in anno and not np.all([val == -1 for val in anno['bbox2D_trunc']]):
                obj['bbox'] = BoxMode.convert(anno['bbox2D_trunc'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            elif 'bbox2D_proj' in anno and anno['bbox2D_proj'][0] != -1:
                obj['bbox'] = BoxMode.convert(anno['bbox2D_proj'], BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
            else:
                # Fallback to standard bbox if available
                if 'bbox' not in obj or obj['bbox'] is None:
                    continue
            
            # Copy R_cam to pose
            if 'R_cam' in anno:
                obj['pose'] = anno['R_cam']
            
            # Set contiguous category_id
            obj["category_id"] = id_map[annotation_category_id]
            
            objs.append(obj)
            has_valid_annotation = True
        
        if has_valid_annotation or (not filter_empty):
            record["annotations"] = objs
            dataset_dicts.append(record)
        else:
            invalid_count += 1
    
    logger.info(f"Filtered out {invalid_count}/{len(imgs_anns)} images without valid annotations")
    logger.info(f"Dataset {dataset_name}: {len(dataset_dicts)} images, "
                f"{sum(len(d['annotations']) for d in dataset_dicts)} annotations")
    
    return dataset_dicts


def register_and_store_rgbd_model_metadata(
    datasets: List[str],
    output_dir: str,
    filter_settings: Optional[Dict[str, Any]] = None,
    category_names: Optional[List[str]] = None
):
    """
    Register model metadata for RGB-D training.
    
    This creates/loads category metadata from all training datasets
    and stores it for consistent category handling during training.
    
    IMPORTANT: If category_names is provided (from config CATEGORY_NAMES),
    it will be used to define the canonical category order. This ensures
    the model's classifier indices match the metadata indices.
    
    Args:
        datasets: List of dataset names to get categories from
        output_dir: Directory to save/load metadata
        filter_settings: Optional filtering settings
        category_names: List of category names in the order they should appear
                       (typically from cfg.DATASETS.CATEGORY_NAMES)
    """
    output_file = os.path.join(output_dir, 'category_meta_rgbd.json')
    
    if os.path.exists(output_file):
        metadata = util.load_json(output_file)
        thing_classes = metadata['thing_classes']
        id_map = metadata['thing_dataset_id_to_contiguous_id']
        id_map = {int(k): v for k, v in id_map.items()}
    else:
        # Collect all categories from datasets with their original IDs
        cat_id_to_name = {}
        
        for dataset_name in datasets:
            meta = MetadataCatalog.get(dataset_name)
            
            # Try to get original category ID mapping
            if hasattr(meta, 'thing_dataset_id_to_contiguous_id'):
                ds_id_map = meta.thing_dataset_id_to_contiguous_id
                if hasattr(meta, 'thing_classes'):
                    # Map original IDs to names
                    for orig_id, cont_id in ds_id_map.items():
                        if cont_id < len(meta.thing_classes):
                            cat_name = meta.thing_classes[cont_id]
                            cat_id_to_name[orig_id] = cat_name
            elif hasattr(meta, 'thing_classes'):
                # Fallback: assume sequential IDs
                for i, cat_name in enumerate(meta.thing_classes):
                    if i not in cat_id_to_name:
                        cat_id_to_name[i] = cat_name
        
        # Build contiguous mapping using config order if provided
        if category_names:
            # Use the config-defined order (CRITICAL for matching model classifier)
            thing_classes = list(category_names)
            cat_name_to_contiguous = {name: i for i, name in enumerate(thing_classes)}
            logger.info(f"Using config-defined category order: {thing_classes[:5]}...")
        else:
            # Fallback: alphabetical order (NOT recommended - may cause mismatch)
            logger.warning("No category_names provided! Using alphabetical order - "
                          "this may cause mismatch with model classifier!")
            unique_names = sorted(set(cat_id_to_name.values()))
            thing_classes = unique_names
            cat_name_to_contiguous = {name: i for i, name in enumerate(unique_names)}
        
        # Map original IDs to contiguous IDs
        id_map = {}
        for orig_id, cat_name in cat_id_to_name.items():
            if cat_name in cat_name_to_contiguous:
                id_map[orig_id] = cat_name_to_contiguous[cat_name]
            else:
                logger.warning(f"Category '{cat_name}' (id={orig_id}) not in category_names, skipping")
        
        os.makedirs(output_dir, exist_ok=True)
        util.save_json(output_file, {
            'thing_classes': thing_classes,
            'thing_dataset_id_to_contiguous_id': id_map,
        })
    
    MetadataCatalog.get('omni3d_model').thing_classes = thing_classes
    MetadataCatalog.get('omni3d_model').thing_dataset_id_to_contiguous_id = id_map
    
    logger.info(f"Model metadata: {len(thing_classes)} categories, {len(id_map)} ID mappings")


# =============================================================================
# Quick Registration Helpers
# =============================================================================

def register_hypersim_rgbd(cfg):
    """Quick registration for Hypersim RGBD datasets."""
    data_root = cfg.DATASETS.DATA_ROOT
    manifest_root = cfg.DATASETS.MANIFEST_ROOT
    filter_settings = get_filter_settings_from_cfg(cfg)
    
    datasets_to_register = [
        ("hypersim_train_rgbd", "hypersim_train_filtered.json", True),
        ("hypersim_val_rgbd", "hypersim_val_filtered.json", False),
    ]
    
    for name, manifest, filter_empty in datasets_to_register:
        if name not in DatasetCatalog:
            manifest_path = os.path.join(manifest_root, manifest)
            if os.path.exists(manifest_path):
                register_single_rgbd_dataset(name, manifest_path, data_root, filter_settings, filter_empty)
                logger.info(f"Registered {name}")


def register_sunrgbd_rgbd(cfg):
    """Quick registration for SUNRGBD datasets."""
    data_root = cfg.DATASETS.DATA_ROOT
    manifest_root = cfg.DATASETS.MANIFEST_ROOT
    filter_settings = get_filter_settings_from_cfg(cfg)
    
    datasets_to_register = [
        ("sunrgbd_val_rgbd", "sunrgbd_val_filtered.json", False),
    ]
    
    for name, manifest, filter_empty in datasets_to_register:
        if name not in DatasetCatalog:
            manifest_path = os.path.join(manifest_root, manifest)
            if os.path.exists(manifest_path):
                register_single_rgbd_dataset(name, manifest_path, data_root, filter_settings, filter_empty)
                logger.info(f"Registered {name}")


def register_all_rgbd_datasets(cfg):
    """
    Register all available RGB-D datasets.
    
    This is a convenience function that registers all known RGBD datasets
    if their manifest files exist.
    """
    register_hypersim_rgbd(cfg)
    register_sunrgbd_rgbd(cfg)
    
    # Additional datasets
    data_root = cfg.DATASETS.DATA_ROOT
    manifest_root = cfg.DATASETS.MANIFEST_ROOT
    filter_settings = get_filter_settings_from_cfg(cfg)
    
    additional_datasets = [
        ("multiscan_val_rgbd", "multiscan_val_filtered.json", False),
        ("scannetpp_val_rgbd", "scannetpp_val_filtered.json", False),
    ]
    
    for name, manifest, filter_empty in additional_datasets:
        if name not in DatasetCatalog:
            manifest_path = os.path.join(manifest_root, manifest)
            if os.path.exists(manifest_path):
                register_single_rgbd_dataset(name, manifest_path, data_root, filter_settings, filter_empty)
                logger.info(f"Registered {name}")
