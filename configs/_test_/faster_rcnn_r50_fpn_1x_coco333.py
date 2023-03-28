_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
custom_imports = dict(
    imports=[
        'mmdet.core.hook.hk'
    ],
    allow_failed_imports=False)

debug = False

data_root = 'data/tfw/KUNKUN/z/'

model = dict(roi_head=dict(bbox_head=dict(num_classes=5)))
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=5),
        mask_head=dict(num_classes=5)))

data = dict(
    persistent_workers=False if debug else True,
    samples_per_gpu=1,
    workers_per_gpu=1 if debug else 4,
    train=dict(
        ann_file=data_root + 'train.json',
        img_prefix=data_root + 'train/'),
    val=dict(
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'val/'),
    test=dict(
        ann_file=data_root + 'val.json',
        img_prefix=data_root + 'val/'))

custom_hooks = [dict(type='MyHook'), dict(type='NumClassCheckHook')]

runner = dict(type='EpochBasedRunner', max_epochs=1)
