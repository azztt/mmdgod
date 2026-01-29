# Copyright (c) Meta Platforms, Inc. and affiliates
from detectron2.config import CfgNode as CN

def get_cfg_defaults(cfg):

    # A list of category names which will be used
    cfg.DATASETS.CATEGORY_NAMES = []

    # The category names which will be treated as ignore
    # e.g., not counting as background during training
    # or as false positives during evaluation.
    cfg.DATASETS.IGNORE_NAMES = []
    
    # ===========================================================================
    # RGB-D Dataset Configuration
    # ===========================================================================
    # Root directory for dataset files (images, depth maps)
    cfg.DATASETS.DATA_ROOT = ""
    
    # Root directory for manifest JSON files
    cfg.DATASETS.MANIFEST_ROOT = ""
    
    # Optional: specific manifest file names for each split
    cfg.DATASETS.TRAIN_MANIFEST = ""
    cfg.DATASETS.VAL_MANIFEST = ""
    cfg.DATASETS.TEST_MANIFEST = ""
    
    # Validation dataset (not in detectron2 defaults)
    cfg.DATASETS.VAL = ()
    
    # ===========================================================================
    # Domain Generalization (DG) Configuration
    # ===========================================================================
    cfg.DG = CN()
    cfg.DG.ENABLED = False
    
    # Frequency Space Domain Randomization (FSDR)
    cfg.DG.FSDR = CN()
    cfg.DG.FSDR.ENABLED = False
    cfg.DG.FSDR.PROBABILITY = 0.5
    cfg.DG.FSDR.VARIANT_BANDS = [[0, 2], [32, 64]]
    cfg.DG.FSDR.BLOCK_SIZE = 64
    
    # Object Style Swap
    cfg.DG.OBJECT_STYLE_SWAP = CN()
    cfg.DG.OBJECT_STYLE_SWAP.ENABLED = False
    
    # ===========================================================================
    # Basic Augmentations Configuration
    # ===========================================================================
    cfg.AUG = CN()
    cfg.AUG.ENABLED = False  # Enable augmentation mapper (DatasetMapper3D_RGBD_DG)
    
    # Photometric augmentations (RGB only)
    cfg.AUG.COLOR_JITTER = CN()
    cfg.AUG.COLOR_JITTER.ENABLED = False
    cfg.AUG.COLOR_JITTER.PROBABILITY = 0.8
    cfg.AUG.COLOR_JITTER.BRIGHTNESS = 0.4
    cfg.AUG.COLOR_JITTER.CONTRAST = 0.4
    cfg.AUG.COLOR_JITTER.SATURATION = 0.4
    cfg.AUG.COLOR_JITTER.HUE = 0.1
    
    # Gaussian noise (RGB)
    cfg.AUG.GAUSSIAN_NOISE = CN()
    cfg.AUG.GAUSSIAN_NOISE.ENABLED = False
    cfg.AUG.GAUSSIAN_NOISE.PROBABILITY = 0.5
    cfg.AUG.GAUSSIAN_NOISE.STD_RANGE = [0.01, 0.05]  # Standard deviation range
    
    # Depth dropout (simulate sensor noise/missing data)
    cfg.AUG.DEPTH_DROPOUT = CN()
    cfg.AUG.DEPTH_DROPOUT.ENABLED = False
    cfg.AUG.DEPTH_DROPOUT.PROBABILITY = 0.3
    cfg.AUG.DEPTH_DROPOUT.NUM_DROPS = [1, 5]  # Number of dropout regions
    cfg.AUG.DEPTH_DROPOUT.DROP_SIZE = [0.02, 0.1]  # Size as fraction of image
    
    # Depth Gaussian noise
    cfg.AUG.DEPTH_NOISE = CN()
    cfg.AUG.DEPTH_NOISE.ENABLED = False
    cfg.AUG.DEPTH_NOISE.PROBABILITY = 0.5
    cfg.AUG.DEPTH_NOISE.STD_RANGE = [0.01, 0.03]  # Noise std in meters

    # ===========================================================================
    # RGB-D Model Configuration (Dual Encoder)
    # ===========================================================================
    cfg.MODEL.USE_DUAL_ENCODER = False
    
    # RGB encoder settings
    cfg.MODEL.RGB_RESNET_DEPTH = 50
    cfg.MODEL.RGB_FROZEN = True
    cfg.MODEL.RGB_FROZEN_STAGES = 4
    
    # Depth encoder settings
    cfg.MODEL.DEPTH_RESNET_DEPTH = 50
    cfg.MODEL.DEPTH_FROZEN = False
    cfg.MODEL.DEPTH_FROZEN_STAGES = 2
    
    # Fusion type: 'concat', 'add', 'gated', 'windowed'
    cfg.MODEL.FUSION_TYPE = "concat"
    
    # ===========================================================================
    # DINOv2/DINOv3 Backbone Configuration
    # ===========================================================================
    cfg.MODEL.BACKBONE.DINO_MODEL = "dinov3-vitl16"  # dinov2-small, dinov2-base, dinov2-large, dinov2-giant, dinov3-vits16, dinov3-vitb16, dinov3-vitl16
    cfg.MODEL.BACKBONE.DINOV2_MODEL = "dinov2_vitl14"  # Legacy: dinov2_vits14, vitb14, vitl14, vitg14
    cfg.MODEL.BACKBONE.RGB_FREEZE_ALL = True   # Freeze RGB encoder completely
    cfg.MODEL.BACKBONE.DEPTH_NUM_FROZEN_BLOCKS = 12  # Partially freeze depth encoder
    cfg.MODEL.BACKBONE.FUSION_TYPE = "concat"  # concat, add, gated
    cfg.MODEL.BACKBONE.OUT_CHANNELS = 256  # Output channels after projection
    
    # Adaptation transformer encoders (placed after frozen backbone, before fusion)
    cfg.MODEL.BACKBONE.USE_ADAPTATION_ENCODER = False  # Add adaptation transformer encoders
    cfg.MODEL.BACKBONE.ADAPTATION_LAYERS = 2  # Number of transformer encoder layers
    cfg.MODEL.BACKBONE.ADAPTATION_HEADS = 8   # Number of attention heads
    
    # Depth normalization
    cfg.MODEL.DEPTH_PIXEL_MEAN = 0.0
    cfg.MODEL.DEPTH_PIXEL_STD = 1.0
    
    # Maximum depth value for depth encoder normalization (meters)
    # Depth values beyond this are clipped before encoding
    cfg.MODEL.DEPTH_MAX = 20.0  # Covers ~97% of Hypersim objects
    
    # Per-sample depth normalization for domain generalization
    # Options: "fixed" (use DEPTH_MAX), "per_sample" (z-score per image), "percentile" (per-image percentile)
    cfg.MODEL.DEPTH_NORM_MODE = "fixed"
    cfg.MODEL.DEPTH_NORM_PERCENTILE = 95  # Used when DEPTH_NORM_MODE="percentile"

    # Should the datasets appear with the same probabilty
    # in batches (e.g., the imbalance from small and large
    # datasets will be accounted for during sampling)
    cfg.DATALOADER.BALANCE_DATASETS = False
    
    # Prefetching settings for faster data loading
    cfg.DATALOADER.PREFETCH_FACTOR = 4  # Each worker prefetches this many batches
    cfg.DATALOADER.PERSISTENT_WORKERS = True  # Keep workers alive between epochs

    # The thresholds for when to treat a known box
    # as ignore based on too heavy of truncation or 
    # too low of visibility in the image. This affects
    # both training and evaluation ignores.
    cfg.DATASETS.TRUNCATION_THRES = 0.99
    cfg.DATASETS.VISIBILITY_THRES = 0.01
    cfg.DATASETS.MIN_HEIGHT_THRES = 0.00
    cfg.DATASETS.MAX_DEPTH = 1e8

    # Whether modal 2D boxes should be loaded, 
    # or if the full 3D projected boxes should be used.
    cfg.DATASETS.MODAL_2D_BOXES = False

    # Whether truncated 2D boxes should be loaded, 
    # or if the 3D full projected boxes should be used.
    cfg.DATASETS.TRUNC_2D_BOXES = True

    # Threshold used for matching and filtering boxes
    # inside of ignore regions, within the RPN and ROIHeads
    cfg.MODEL.RPN.IGNORE_THRESHOLD = 0.5

    # Configuration for cube head
    cfg.MODEL.ROI_CUBE_HEAD = CN()
    cfg.MODEL.ROI_CUBE_HEAD.NAME = "CubeHead"
    cfg.MODEL.ROI_CUBE_HEAD.POOLER_RESOLUTION = 7
    cfg.MODEL.ROI_CUBE_HEAD.POOLER_SAMPLING_RATIO = 0
    cfg.MODEL.ROI_CUBE_HEAD.POOLER_TYPE = "ROIAlignV2"

    # Settings for the cube head features
    cfg.MODEL.ROI_CUBE_HEAD.NUM_CONV = 0
    cfg.MODEL.ROI_CUBE_HEAD.CONV_DIM = 256
    cfg.MODEL.ROI_CUBE_HEAD.NUM_FC = 2
    cfg.MODEL.ROI_CUBE_HEAD.FC_DIM = 1024
    
    # =========================================================================
    # DETR3DHead / Transformer settings (used when NAME="DETR3DHead")
    # =========================================================================
    cfg.MODEL.ROI_CUBE_HEAD.USE_TRANSFORMER = False  # Legacy flag
    cfg.MODEL.ROI_CUBE_HEAD.FEATURE_DIM = 256  # D_MODEL for DETR3DHead
    cfg.MODEL.ROI_CUBE_HEAD.D_MODEL = 256  # Alias for FEATURE_DIM
    cfg.MODEL.ROI_CUBE_HEAD.NUM_LAYERS = 3  # Number of cross-attention layers
    cfg.MODEL.ROI_CUBE_HEAD.NUM_DECODER_LAYERS = 3  # Alias for NUM_LAYERS
    cfg.MODEL.ROI_CUBE_HEAD.NUM_HEADS = 8  # Attention heads
    cfg.MODEL.ROI_CUBE_HEAD.NHEAD = 8  # Alias for NUM_HEADS
    cfg.MODEL.ROI_CUBE_HEAD.DIM_FEEDFORWARD = 1024  # FFN dimension
    cfg.MODEL.ROI_CUBE_HEAD.DROPOUT = 0.1
    cfg.MODEL.ROI_CUBE_HEAD.USE_FLASH_ATTENTION = True
    
    # =========================================================================
    # Pure DETR3D settings (when using PureDETR3D meta-architecture)
    # =========================================================================
    cfg.MODEL.ROI_CUBE_HEAD.NUM_QUERIES = 300  # Number of learnable object queries (DDETR: 300)
    cfg.MODEL.ROI_CUBE_HEAD.N_POINTS = 4  # Sampling points per level for deformable attention
    cfg.MODEL.ROI_CUBE_HEAD.TEST_SCORE_THRESH = 0.05  # Score threshold at test time
    cfg.MODEL.ROI_CUBE_HEAD.TEST_NMS_THRESH = 0.5  # NMS threshold at test time
    cfg.MODEL.ROI_CUBE_HEAD.TEST_TOPK_PER_IMAGE = 300  # Max predictions per image
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_CLS = 2.0  # Classification loss weight

    # =========================================================================
    # DETR3D Head settings (cross-attention based 3D head) - nested config
    # =========================================================================
    cfg.MODEL.ROI_CUBE_HEAD.DETR3D = CN()
    cfg.MODEL.ROI_CUBE_HEAD.DETR3D.D_MODEL = 256       # Model dimension (same as query dim)
    cfg.MODEL.ROI_CUBE_HEAD.DETR3D.QUERY_DIM = 256     # Query embedding dimension
    cfg.MODEL.ROI_CUBE_HEAD.DETR3D.NUM_CROSS_ATTENTION_LAYERS = 4
    cfg.MODEL.ROI_CUBE_HEAD.DETR3D.NUM_HEADS = 8       # Legacy alias
    cfg.MODEL.ROI_CUBE_HEAD.DETR3D.N_HEADS = 8         # Deformable attention heads
    cfg.MODEL.ROI_CUBE_HEAD.DETR3D.N_POINTS = 4        # Sampling points per head per level (deformable attention)
    cfg.MODEL.ROI_CUBE_HEAD.DETR3D.FFN_DIM = 1024      # Legacy alias
    cfg.MODEL.ROI_CUBE_HEAD.DETR3D.DIM_FEEDFORWARD = 1024  # FFN dimension
    cfg.MODEL.ROI_CUBE_HEAD.DETR3D.DROPOUT = 0.1
    
    # the style to predict Z with currently supported
    # options --> ['direct', 'sigmoid', 'log', 'clusters']
    cfg.MODEL.ROI_CUBE_HEAD.Z_TYPE = "direct"

    # the style to predict pose with currently supported
    # options --> ['6d', 'euler', 'quaternion']
    cfg.MODEL.ROI_CUBE_HEAD.POSE_TYPE = "6d"

    # Whether to scale all 3D losses by inverse depth
    cfg.MODEL.ROI_CUBE_HEAD.INVERSE_Z_WEIGHT = False

    # Virtual depth puts all predictions of depth into
    # a shared virtual space with a shared focal length. 
    cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_DEPTH = True
    cfg.MODEL.ROI_CUBE_HEAD.VIRTUAL_FOCAL = 512.0

    # If true, then all losses are computed using the 8 corners
    # such that they are all in a shared scale space. 
    # E.g., their scale correlates with their impact on 3D IoU.
    # This way no manual weights need to be set.
    cfg.MODEL.ROI_CUBE_HEAD.DISENTANGLED_LOSS = True

    # When > 1, the outputs of the 3D head will be based on
    # a 2D scale clustering, based on 2D proposal height/width.
    # This parameter describes the number of bins to cluster.
    cfg.MODEL.ROI_CUBE_HEAD.CLUSTER_BINS = 1

    # Whether batch norm is enabled during training. 
    # If false, all BN weights will be frozen. 
    cfg.MODEL.USE_BN = True

    # Whether to predict the pose in allocentric space. 
    # The allocentric space may correlate better with 2D 
    # images compared to egocentric poses. 
    cfg.MODEL.ROI_CUBE_HEAD.ALLOCENTRIC_POSE = True

    # Whether to use chamfer distance for disentangled losses
    # of pose. This avoids periodic issues of rotation but 
    # may prevent the pose "direction" from being interpretable.
    cfg.MODEL.ROI_CUBE_HEAD.CHAMFER_POSE = True

    # Should the prediction heads share FC features or not. 
    # These include groups of uv, z, whl, pose.
    cfg.MODEL.ROI_CUBE_HEAD.SHARED_FC = True

    # Check for stable gradients. When inf is detected, skip the update. 
    # This prevents an occasional bad sample from exploding the model. 
    # The threshold below is the allows percent of bad samples. 
    # 0.0 is off, and 0.01 is recommended for minor robustness to exploding.
    cfg.MODEL.STABILIZE = 0.01
    
    # Whether or not to use the dimension priors
    cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_ENABLED = True

    # How prior dimensions should be computed? 
    # The supported modes are ["exp", "sigmoid"]
    # where exp is unbounded and sigmoid is bounded
    # between +- 3 standard deviations from the mean.
    cfg.MODEL.ROI_CUBE_HEAD.DIMS_PRIORS_FUNC = 'exp'

    # weight for confidence loss. 0 is off.
    cfg.MODEL.ROI_CUBE_HEAD.USE_CONFIDENCE = 1.0

    # Loss weights for XY, Z, Dims, Pose
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_3D = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_XY = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_Z = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_DIMS = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_POSE = 1.0
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_JOINT = 0.25  # Corner loss weight
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_GIOU = 2.0  # GIoU loss weight (Deformable DETR)
    
    # Focal loss parameters (Deformable DETR)
    cfg.MODEL.ROI_CUBE_HEAD.FOCAL_ALPHA = 0.25  # Focal loss alpha (class balance)
    cfg.MODEL.ROI_CUBE_HEAD.FOCAL_GAMMA = 2.0  # Focal loss gamma (focus on hard examples)

    cfg.MODEL.DLA = CN()

    # Supported types for DLA backbones are...
    # dla34, dla46_c, dla46x_c, dla60x_c, dla60, dla60x, dla102x, dla102x2, dla169
    cfg.MODEL.DLA.TYPE = 'dla34'

    # Only available for dla34, dla60, dla102
    cfg.MODEL.DLA.TRICKS = False

    # A joint loss for the disentangled loss.
    # All predictions are computed using a corner
    # or chamfers loss depending on chamfer_pose!
    # Recommened to keep this weight small: [0.05, 0.5]
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_JOINT = 1.0

    # Classification loss weight (DETR3D uses 2.0 vs bbox weight 0.25)
    cfg.MODEL.ROI_CUBE_HEAD.LOSS_W_CLS = 1.0

    # Inference thresholds for filtering predictions
    cfg.MODEL.ROI_CUBE_HEAD.TEST_SCORE_THRESH = 0.05  # Minimum score to keep prediction
    cfg.MODEL.ROI_CUBE_HEAD.TEST_NMS_THRESH = 0.5  # NMS IoU threshold
    cfg.MODEL.ROI_CUBE_HEAD.TEST_TOPK_PER_IMAGE = 100  # Max predictions per image

    # =========================================================================
    # Depth Completion (Auxiliary Self-Supervised Task)
    # =========================================================================
    cfg.MODEL.DEPTH_COMPLETION = CN()
    cfg.MODEL.DEPTH_COMPLETION.ENABLED = False
    cfg.MODEL.DEPTH_COMPLETION.FEATURE_DIM = 256
    cfg.MODEL.DEPTH_COMPLETION.NUM_DECODER_LAYERS = 4
    cfg.MODEL.DEPTH_COMPLETION.USE_SKIP_CONNECTIONS = True
    cfg.MODEL.DEPTH_COMPLETION.SPARSITY = 0.1        # Keep 10% of pixels
    cfg.MODEL.DEPTH_COMPLETION.SPARSITY_MIN = 0.05   # Min sparsity for augmentation
    cfg.MODEL.DEPTH_COMPLETION.SPARSITY_MAX = 0.2    # Max sparsity for augmentation
    cfg.MODEL.DEPTH_COMPLETION.LOSS_TYPE = "l1"      # l1, l2, or berhu
    cfg.MODEL.DEPTH_COMPLETION.LOSS_WEIGHT = 1.0

    # sgd, adam, adam+amsgrad, adamw, adamw+amsgrad
    cfg.SOLVER.TYPE = 'sgd'
    
    # Learning rate multiplier for backbone parameters
    # Useful when training with pretrained backbones (e.g., 0.1 = 10x lower LR)
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    
    # Weight decay for bias parameters (None = use WEIGHT_DECAY)
    cfg.SOLVER.WEIGHT_DECAY_BIAS = None
    
    # Bias LR factor (None = use BASE_LR)
    cfg.SOLVER.BIAS_LR_FACTOR = None
    
    # ===========================================================================
    # EMA (Exponential Moving Average) Configuration
    # ===========================================================================
    cfg.SOLVER.EMA = CN()
    cfg.SOLVER.EMA.ENABLED = False
    cfg.SOLVER.EMA.DECAY = 0.9999  # EMA decay rate (higher = smoother)
    cfg.SOLVER.EMA.WARMUP_ITERS = 2000  # Warmup iterations for EMA
    cfg.SOLVER.EMA.USE_EMA_FOR_EVAL = True  # Use EMA weights for evaluation

    cfg.MODEL.RESNETS.TORCHVISION = True
    cfg.TEST.DETECTIONS_PER_IMAGE = 100

    cfg.TEST.VISIBILITY_THRES = 1/2.0
    cfg.TEST.TRUNCATION_THRES = 1/2.0
    
    # Validation split for train/val split evaluation
    cfg.TEST.VAL_SPLIT = 0.2  # Fraction of training data for validation
    cfg.TEST.FULL_EVAL_PERIOD = 0  # Frequency for full evaluation on external test sets (0 = disabled)

    cfg.INPUT.RANDOM_FLIP = "horizontal"

    # When True, we will use localization uncertainty
    # as the new IoUness score in the RPN.
    cfg.MODEL.RPN.OBJECTNESS_UNCERTAINTY = 'IoUness'

    # If > 0.0 this is the scaling factor that will be applied to
    # an RoI 2D box before doing any pooling to give more context. 
    # Ex. 1.5 makes width and height 50% larger. 
    cfg.MODEL.ROI_CUBE_HEAD.SCALE_ROI_BOXES = 0.0

    # weight path specifically for pretraining (no checkpointables will be loaded)
    cfg.MODEL.WEIGHTS_PRETRAIN = ''