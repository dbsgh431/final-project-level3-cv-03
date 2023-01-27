import json
# dataset settings
dataset_type = 'CocoDataset'
img_norm_cfg = dict(
    mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225], to_rgb=True)
cfg = json.load(open("/opt/ml/configs.json", "r"))

classes = ("pothole",)

albu_train_transforms = [ 
    # dict(type='Resize', height=512, width=512, p=1),
    dict(type='Blur', blur_limit=3, p=0.5),
]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=cfg['train_anno_path'],
        img_prefix=cfg['train_img_path'],
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=cfg['valid_anno_path'],
        img_prefix=cfg['valid_img_path'],
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=cfg['test_anno_path'],
        img_prefix=cfg['test_img_path'],
        classes=classes,
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
