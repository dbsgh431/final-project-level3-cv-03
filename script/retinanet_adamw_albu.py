_base_ = [
    '/opt/ml/final-project-level3-cv-03/mmdetection/configs/_base_/models/retinanet_r50_fpn.py',
    '/opt/ml/final-project-level3-cv-03/mmdetection/configs/_base_/datasets/coco_detection.py',
    '/opt/ml/final-project-level3-cv-03/mmdetection/configs/_base_/schedules/schedule_1x.py',
    '/opt/ml/final-project-level3-cv-03/mmdetection/configs/_base_/default_runtime.py'
]

model = dict(bbox_head=dict(num_classes=1))
classes = ('pothole',)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

albu_pipeline = [
    dict(type='OneOf',
         transforms=[
             dict(type='RandomFog', p=1),
             dict(type='RandomRain', p=1),
             dict(type='RandomShadow', p=1),
             dict(type='RandomSnow', p=1),
             dict(type='RandomSunFlare', p=1)
         ], p=1)
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1280, 720), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Albu', transforms=albu_pipeline,
         bbox_params=dict(
             type='BboxParams',
             format='pascal_voc',
             label_fields=['gt_labels'],
             min_visibility=0,
             filter_lost_elements=True),
         keymap={
             'img' : 'image',
             'gt_masks' : 'masks',
             'gt_bboxes' : 'bboxes'
         },
         update_pad_shape=False,
         skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1280, 720),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(samples_per_gpu=4,
            train=dict(ann_file = '/opt/ml/input/annotation/train.json',
                       img_prefix = '/opt/ml/input/train/',
                       classes = classes,
                       pipeline = train_pipeline),
            val = dict(ann_file = '/opt/ml/input/annotation/test.json',
                       img_prefix = '/opt/ml/input/test/',
                       classes = classes,
                       pipeline = test_pipeline),
            test = dict(ann_file  = '/opt/ml/input/annotation/test.json',
                        img_prefix = '/opt/ml/input/test/',
                        classes = classes,
                        pipeline = test_pipeline))
# optimizer
optimizer = dict(_delete_ = True,
                 type='AdamW', lr=0.001, weight_decay=0.0001)
runner = dict(type='EpochBasedRunner', max_epochs=30)

import wandb

wandb.login()
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='WandbLoggerHook',interval=50,
            init_kwargs=dict(
                project='Final Project',
                entity = 'aitech4_cv3',
                name = "test"))])