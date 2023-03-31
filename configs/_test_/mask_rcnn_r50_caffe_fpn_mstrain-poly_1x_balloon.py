_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

debug = False

dataset_type = 'CocoDataset'
classes = ('内插', '负极耳OK', '正极耳OK')

data_root = 'data/tfw/YKUNKUN/'

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes)),
        mask_head=dict(num_classes=len(classes))))

img_scale = (1560, 1560)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=img_scale,
        ratio_range=[0.75, 1.25],
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=img_scale),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
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
    # persistent_workers=False if debug else True,
    # samples_per_gpu=1,
    # workers_per_gpu=1 if debug else 4,
    train=dict(
        pipeline=train_pipeline,
        ann_file=data_root + 'train.json',
        classes=classes,
        img_prefix=data_root + 'train/'),
    val=dict(
        pipeline=test_pipeline,
        ann_file=data_root + 'val.json',
        classes=classes,
        img_prefix=data_root + 'val/'),
    test=dict(
        pipeline=test_pipeline,
        ann_file=data_root + 'val.json',
        classes=classes,
        img_prefix=data_root + 'val/'))

# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
# load_from = 'checkpoints/base_mask7.pth'
auto_scale_lr = dict(enable=True, base_batch_size=16)
# runner = dict(type='EpochBasedRunner', max_epochs=12)
