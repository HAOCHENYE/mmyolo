_base_ = ['.default_runtime']

from mmcv.transforms import LoadImageFromFile
from mmdet.datasets.transforms import Albu, PackDetInputs, RandomFlip
from mmdet.evaluation import CocoMetric
from mmdet.models import CrossEntropyLoss, DetDataPreprocessor
from mmdet.models.task_modules import YOLOAnchorGenerator
from mmengine.dataset.sampler import DefaultSampler
from mmengine.dataset.utils import pseudo_collate
from mmengine.evaluator import Evaluator
from mmengine.hooks import CheckpointHook, EMAHook
from mmengine.runner.experimental_runner import Runner
from torch.utils.data import DataLoader

from mmyolo.datasets import (BatchShapePolicy, LetterResize, LoadAnnotations,
                             Mosaic, YOLOv5CocoDataset, YOLOv5HSVRandomAug,
                             YOLOv5KeepRatioResize, YOLOv5RandomAffine)
from mmyolo.engine.hooks import YOLOv5ParamSchedulerHook
from mmyolo.models import (IoULoss, YOLODetector, YOLOv5CSPDarknet, YOLOv5Head,
                           YOLOv5HeadModule, YOLOv5PAFPN)
from .default_runtime import *

# dataset settings
data_root = 'data/coco/'

# parameters that often need to be modified
num_classes = 80
img_scale = (640, 640)  # width, height
deepen_factor = 0.33
widen_factor = 0.5
max_epochs = 300
save_epoch_intervals = 10
train_batch_size_per_gpu = 16
train_num_workers = 8
val_batch_size_per_gpu = 1
val_num_workers = 2

# persistent_workers must be False if num_workers is 0.
persistent_workers = True

# Base learning rate for optim_wrapper
base_lr = 0.01

# only on Val
batch_shapes_cfg = BatchShapePolicy(
    batch_size=val_batch_size_per_gpu,
    img_size=img_scale[0],
    size_divisor=32,
    extra_pad_ratio=0.5)

anchors = [
    [(10, 13), (16, 30), (33, 23)],  # P3/8
    [(30, 61), (62, 45), (59, 119)],  # P4/16
    [(116, 90), (156, 198), (373, 326)]  # P5/32
]
strides = [8, 16, 32]
num_det_layers = 3

# # single-scale training is recommended to
# # be turned on, which can speed up training.
env_cfg = dict(cudnn_benchmark=True)
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=save_epoch_intervals
)

test_cfg = dict(
    multi_label=True,
    nms_pre=30000,
    score_thr=0.001,
    nms=dict(type='nms', iou_threshold=0.65),
    max_per_img=300
)

model = YOLODetector(
    data_preprocessor=DetDataPreprocessor(
        mean=[0., 0., 0.],
        std=[255., 255., 255.],
        bgr_to_rgb=True),
    backbone=YOLOv5CSPDarknet(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    neck=YOLOv5PAFPN(
        deepen_factor=deepen_factor,
        widen_factor=widen_factor,
        in_channels=[256, 512, 1024],
        out_channels=[256, 512, 1024],
        num_csp_blocks=3,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=YOLOv5Head(
        head_module=YOLOv5HeadModule(
            num_classes=num_classes,
            in_channels=[256, 512, 1024],
            widen_factor=widen_factor,
            featmap_strides=strides,
            num_base_priors=3,
        ),
        prior_generator=YOLOAnchorGenerator(
            base_sizes=anchors,
            strides=strides),
        # scaled based on number of detection layers
        loss_cls=CrossEntropyLoss(
            use_sigmoid=True,
            reduction='mean',
            loss_weight=0.5 * (num_classes / 80 * 3 / num_det_layers)),
        loss_bbox=IoULoss(
            iou_mode='ciou',
            bbox_format='xywh',
            eps=1e-7,
            reduction='mean',
            loss_weight=0.05 * (3 / num_det_layers),
            return_iou=True),
        loss_obj=CrossEntropyLoss(
            use_sigmoid=True,
            reduction='mean',
            loss_weight=1.0 * ((img_scale[0] / 640)**2 * 3 / num_det_layers)),
        prior_match_thr=4.,
        obj_level_weights=[4., 1., 0.4],
        test_cfg=test_cfg,
        train_cfg=train_cfg),
    test_cfg=test_cfg)


albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

pre_transform = [
    LoadImageFromFile(file_client_args=file_client_args),
    LoadAnnotations(with_bbox=True)

]

train_pipeline = [
    *pre_transform,
    Mosaic(
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    YOLOv5RandomAffine(
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(0.5, 1.5),
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    Albu(
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    YOLOv5HSVRandomAug(),
    RandomFlip(prob=0.5),
    PackDetInputs(
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_dataset = YOLOv5CocoDataset(
    data_root=data_root,
    ann_file='annotations/instances_train2017.json',
    data_prefix=dict(img='train2017/'),
    filter_cfg=dict(filter_empty_gt=False, min_size=32),
    pipeline=train_pipeline
)

train_dataloader = DataLoader(
    batch_size=train_batch_size_per_gpu,
    num_workers=0,
    persistent_workers=False,
    pin_memory=True,
    sampler=DefaultSampler(dataset=train_dataset, shuffle=True),
    collate_fn=pseudo_collate,
    dataset=train_dataset,
)

test_pipeline = [
    LoadImageFromFile(file_client_args=file_client_args),
    YOLOv5KeepRatioResize(scale=img_scale),
    LetterResize(
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    LoadAnnotations(with_bbox=True),
    PackDetInputs(
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataset = YOLOv5CocoDataset(
    data_root=data_root,
    test_mode=True,
    data_prefix=dict(img='val2017/'),
    ann_file='annotations/instances_val2017.json',
    pipeline=test_pipeline,
    batch_shapes_cfg=batch_shapes_cfg
)

val_dataloader = DataLoader(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=DefaultSampler(dataset=val_dataset, shuffle=False),
    dataset=val_dataset,
    collate_fn=pseudo_collate
)

test_dataloader = val_dataloader

param_scheduler = None
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='SGD',
        lr=base_lr,
        momentum=0.937,
        weight_decay=0.0005,
        nesterov=True,
        batch_size_per_gpu=train_batch_size_per_gpu),
    constructor='YOLOv5OptimizerConstructor')

default_hooks = dict(
    param_scheduler=YOLOv5ParamSchedulerHook(
        scheduler_type='linear',
        lr_factor=0.01,
        max_epochs=max_epochs),
    checkpoint=CheckpointHook(
        interval=save_epoch_intervals,
        save_best='auto',
        max_keep_ckpts=3))

custom_hooks = [
    EMAHook(
        ema_type='ExpMomentumEMA',
        momentum=0.0001,
        update_buffers=True,
        strict_load=False,
        priority=49)
]

val_evaluator = Evaluator(
    metrics=CocoMetric(
        proposal_nums=(100, 1, 10),
        ann_file=data_root + 'annotations/instances_val2017.json',
        metric='bbox'
    )
)
test_evaluator = val_evaluator


val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')


runner = Runner(
    model=model,
    work_dir='./work_dirs',
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    test_dataloader=test_dataloader,
    train_cfg=train_cfg,
    val_cfg=val_cfg,
    test_cfg=test_cfg,
    optim_wrapper=optim_wrapper,
    param_scheduler=param_scheduler,
    val_evaluator=val_evaluator,
    test_evaluator=test_evaluator,
    default_hooks=default_hooks,
    custom_hooks=custom_hooks,
    env_cfg=env_cfg,
    log_processor=log_processor,
    log_level=log_level,
    visualizer=visualizer,
    default_scope=default_scope,
    experiment_name='test_config',
    cfg=dict()
)

# runner.train()