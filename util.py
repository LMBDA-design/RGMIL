from torchvision import datasets, transforms
import torch
from torchvision.transforms._transforms_video import RandomResizedCropVideo, NormalizeVideo, RandomHorizontalFlipVideo
from timm.data import create_transform
from timm.data.transforms import RandomResizedCropAndInterpolation
import numpy as np
import os
import yaml
import pandas as pd
import pingouin as pg
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = './imagenet'
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 0
# could be overwritten by command line argument
_C.DATA.RESAMPLE = False
_C.DATA.UNBC = True
_C.DATA.WHICH_ID = 1
# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = 'swin'
# Model name
_C.MODEL.NAME = 'swin_tiny_patch4_window7_224'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
_C.MODEL.PRETRAINED_RESNET = ''
# Which kind of pretrain models
_C.MODEL.WHICH_PRETRAIN = 'pain'
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 4
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

# Swin MLP parameters
_C.MODEL.SWIN_MLP = CN()
_C.MODEL.SWIN_MLP.PATCH_SIZE = 4
_C.MODEL.SWIN_MLP.IN_CHANS = 3
_C.MODEL.SWIN_MLP.EMBED_DIM = 96
_C.MODEL.SWIN_MLP.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN_MLP.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN_MLP.WINDOW_SIZE = 7
_C.MODEL.SWIN_MLP.MLP_RATIO = 4.
_C.MODEL.SWIN_MLP.APE = False
_C.MODEL.SWIN_MLP.PATCH_NORM = True

# swin resnet paremeters
_C.MODEL.REGRESSION = False
_C.MODEL.CENTERLOSS = False

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 1e-3
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 2
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False
# whether use cross validation
_C.TRAIN.CROSS_VALIDATION = 0
_C.TRAIN.WEIGHT1 = 0.1
_C.TRAIN.WEIGHT2 = 1.0
_C.TRAIN.WEIGHT3 = 1.0

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True
# Whether to use SequentialSampler as validation sampler
_C.TEST.SEQUENTIAL = False

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


# def get_transforms(train=False):
#     if train:
#         data_transform = create_transform(228, is_training=True)
#         data_transform = transforms.Compose([data_transform,
#                                              # transforms.RandomRotation(degrees=(-30, 30)),
#                                              transforms.Grayscale(num_output_channels=3),
#                                              transforms.RandomCrop(224),
#                                              ])
#     else:
#         data_transform = transforms.Compose([
#             transforms.Resize(228),
#             transforms.Grayscale(num_output_channels=3),
#             transforms.TenCrop(224),
#             transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
#             transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
#         ])
#     return data_transform

def get_transforms(train=False):  # for res18
    if train:
        data_transform = transforms.Compose([transforms.Resize(228),
                                             transforms.RandomCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Grayscale(num_output_channels=3),
                                             transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
                                             # transforms.RandomRotation(degrees=(-30, 30)),
                                             transforms.RandomHorizontalFlip(),
                                             ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize(228),
            transforms.Grayscale(num_output_channels=3),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
        ])
    return data_transform


class TransReshape(torch.nn.Module):
    def __init__(self):
        super(TransReshape, self).__init__()

    def forward(self, imgs):
        imgs = imgs.permute(1, 0, 2, 3)
        return imgs


class BTransReshape(torch.nn.Module):
    def __init__(self):
        super(BTransReshape, self).__init__()

    def forward(self, imgs):
        imgs = imgs.permute(1, 0, 2, 3, 4)
        return imgs


class NormalReshape(torch.nn.Module):
    def __init__(self):
        super(NormalReshape, self).__init__()

    def forward(self, imgs):
        imgs = imgs.permute(1, 0, 2, 3)
        return imgs


class SwapReshape(torch.nn.Module):
    def __init__(self):
        super(SwapReshape, self).__init__()

    def forward(self, imgs):
        imgs = imgs.permute(1, 0, 2, 3, 4)
        return imgs


def get_btransforms(train=False):  # for res18
    if train:
        data_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            TransReshape(),
            NormalizeVideo(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            RandomResizedCropVideo(224),
            RandomHorizontalFlipVideo(),
            NormalReshape(),
        ])
    else:
        data_transform = transforms.Compose([
            transforms.Resize(228),
            transforms.Grayscale(num_output_channels=3),
            transforms.TenCrop(224),
            transforms.Lambda(lambda crops: torch.stack([(crop) for crop in crops])),
            transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
            SwapReshape()
        ])
    return data_transform


def get_sampler(dataset_train):
    target = dataset_train.targets
    # print(target.shape,target[0])
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler_train = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
    return sampler_train


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()
    return config


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.pretrained:
        config.MODEL.PRETRAINED = args.pretrained
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config


def MAE(outs, targets):
    return torch.sum(torch.abs(outs - targets)) / outs.size(0)


def MSE(outs, targets):
    return torch.sum(torch.pow(outs - targets, 2)) / outs.size(0)


def PCC(outs, targets):
    mean_outs = torch.sum(outs) / outs.size(0)
    mean_targets = torch.sum(targets) / outs.size(0)
    var_outs = torch.pow(torch.sum(torch.pow(outs - mean_outs, 2)) / outs.size(0), 1 / 2)
    var_targets = torch.pow(torch.sum(torch.pow(targets - mean_targets, 2)) / outs.size(0), 1 / 2)
    return torch.sum(torch.abs((outs - mean_outs) * (targets - mean_targets))) / (var_outs * var_targets * outs.size(0))


def ICC(outs, targets):
    outs = outs.squeeze().cpu().numpy()
    targets = targets.squeeze().cpu().numpy()
    df1 = pd.DataFrame(columns=['prediction'])
    df2 = pd.DataFrame(columns=['prediction'])
    df1['prediction'] = outs
    df2['prediction'] = targets
    df1.insert(0, 'reader', np.ones(df1.shape[0]))
    df2.insert(0, 'reader', np.ones(df2.shape[0]) * 2)
    df1.insert(0, 'frame', range(df1.shape[0]))
    df2.insert(0, 'frame', range(df2.shape[0]))
    data = pd.concat([df1, df2])
    icc = pg.intraclass_corr(data=data, targets='frame', raters='reader', ratings='prediction')
    return icc['ICC'][2]  # icc[3,1]
