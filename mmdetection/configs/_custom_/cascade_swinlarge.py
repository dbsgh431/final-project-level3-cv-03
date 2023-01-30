_base_ = [
    '/opt/ml/final-project-level3-cv-03/mmdetection/configs/_custom_/cascade_rcnn_r50_fpn.py',
    '/opt/ml/final-project-level3-cv-03/mmdetection/configs/_custom_/coco_detection.py',
    '/opt/ml/final-project-level3-cv-03/mmdetection/configs/_base_/schedules/schedule_2x.py',
    '/opt/ml/final-project-level3-cv-03/mmdetection/configs/_custom_/default_runtime.py'
]

#model
pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth'  # noqa

model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[192, 384, 768, 1536]))

#scheduler
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
lr_config = dict( _delete_=True, policy='poly', power=0.9, min_lr=1e-4, by_epoch=False)

runner = dict(max_epochs=50)
evaluation = dict(
    interval = 1,
    save_best = 'bbox_mAP',
    metric='bbox'
)
#default_runtime