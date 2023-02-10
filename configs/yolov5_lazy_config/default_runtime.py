from mmdet.engine.hooks import DetVisualizationHook
from mmdet.visualization import DetLocalVisualizer

from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.runner import LogProcessor
from mmengine.visualization import LocalVisBackend

default_scope = 'mmyolo'

vis_backends = [LocalVisBackend(save_dir='tmp')]
visualizer = DetLocalVisualizer.get_instance(name='visualizer', vis_backends=vis_backends)
log_processor = LogProcessor(window_size=50, by_epoch=True)

log_level = 'INFO'
load_from = None
resume = False

file_client_args = dict(backend='disk')

default_hooks = dict(
    timer=IterTimerHook(),
    logger=LoggerHook(interval=50),
    param_scheduler=ParamSchedulerHook(),
    checkpoint=CheckpointHook(interval=1),
    sampler_seed=DistSamplerSeedHook(),
    visualization=DetVisualizationHook())

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
