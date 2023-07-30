from detectron2.config.defaults import _C
from detectron2.config import CfgNode as CN


# ---------------------------------------------------------------------------- #
# Additional Configs
# ---------------------------------------------------------------------------- #
_C.MODEL.MOBILENET = False
_C.MODEL.BACKBONE.ANTI_ALIAS = False
_C.MODEL.RESNETS.DEFORM_INTERVAL = 1
_C.INPUT.HFLIP_TRAIN = True
_C.INPUT.CROP.CROP_INSTANCE = True
_C.INPUT.IS_ROTATE = False
_C.DATASETS.DATASETS_DIR = None
# ---------------------------------------------------------------------------- #
# PSR Options
# ---------------------------------------------------------------------------- #
_C.MODEL.PSR = CN()

# Instance hyper-parameters
_C.MODEL.PSR.PAR_IN_FEATURES = ["p2", "p3", "p4", "p5", "p6"]
_C.MODEL.PSR.FPN_PAR_STRIDES = [8, 8, 16, 32, 32]
_C.MODEL.PSR.FPN_SCALE_RANGES = ((1, 96), (48, 192), (96, 384), (192, 768), (384, 2048))
_C.MODEL.PSR.SIGMA = 0.2
# Channel size for the partition head.
_C.MODEL.PSR.PAR_IN_CHANNELS = 256
_C.MODEL.PSR.PAR_CHANNELS = 512
# Convolutions to use in the partition head.
_C.MODEL.PSR.NUM_PAR_CONVS = 4
_C.MODEL.PSR.USE_DCN_IN_PAR = False
_C.MODEL.PSR.TYPE_DCN = 'DCN'
_C.MODEL.PSR.NUM_GRIDS = [40, 36, 24, 16, 12]

_C.MODEL.PSR.NUM_RANKS = 5
_C.MODEL.PSR.NUM_KERNELS = 256
_C.MODEL.PSR.NORM = "GN"
_C.MODEL.PSR.USE_COORD_CONV = True
_C.MODEL.PSR.PRIOR_PROB = 0.01
# redesign the instance branch

_C.MODEL.PSR.ATTENTION_DEPTH = 3

# Mask hyper-parameters.
# Channel size for the mask tower.
_C.MODEL.PSR.MASK_IN_FEATURES = ["p2", "p3", "p4", "p5"]
_C.MODEL.PSR.MASK_IN_CHANNELS = 256
_C.MODEL.PSR.MASK_CHANNELS = 128
_C.MODEL.PSR.NUM_MASKS = 256

# Test cfg.
_C.MODEL.PSR.NMS_PRE = 500
_C.MODEL.PSR.SCORE_THR = 0.1
_C.MODEL.PSR.UPDATE_THR = 0.05
_C.MODEL.PSR.MASK_THR = 0.5
_C.MODEL.PSR.MAX_PER_IMG = 100
_C.MODEL.PSR.NMS_TYPE = "mask"

# Loss cfg.
_C.MODEL.PSR.LOSS = CN()
_C.MODEL.PSR.LOSS.FOCAL_USE_SIGMOID = True
_C.MODEL.PSR.LOSS.FOCAL_ALPHA = 0.25
_C.MODEL.PSR.LOSS.FOCAL_GAMMA = 2.0
_C.MODEL.PSR.LOSS.PARTITION_WEIGHT = 1.0
_C.MODEL.PSR.LOSS.DICE_WEIGHT = 3.0

# [Optimizer]
_C.SOLVER.OPTIMIZER = "SGD"
