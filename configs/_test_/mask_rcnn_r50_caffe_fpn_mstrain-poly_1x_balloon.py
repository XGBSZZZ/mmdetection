_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

debug = False

dataset_type = 'CocoDataset'
classes = ('熔珠', '爆点', '熔珠-向内溢出-无焊黑', '熔珠-向外溢出-无焊黑', '轻微熔钉', '熔珠-向内溢出-焊黑', '焊黑', '熔珠-锡渣', '存疑', 'OK', '熔珠-焊缝边缘', '电解液', '针孔', '焊灰', '熔钉')

data_root = 'data/tfw/KUNKUN2/z/'
# optimizer = dict(type='Adam', lr=0.01, betas=(0.9, 0.999), eps=1e-8)
# optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
# optimizer = dict(
#     _delete_=True,
#     type='AdamW',
#     lr=0.001,
#     weight_decay=0.05,
#     paramwise_cfg=dict(norm_decay_mult=0., bypass_duplicate=True))
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[8, 11])

# auto_scale_lr = dict(enable=False, base_batch_size=16)

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=len(classes)),
        mask_head=dict(num_classes=len(classes))))

data = dict(
    persistent_workers=False if debug else True,
    samples_per_gpu=1,
    workers_per_gpu=1 if debug else 4,
    train=dict(
        ann_file=data_root + 'train.json',
        classes=classes,
        img_prefix=data_root + 'train/'),
    val=dict(
        ann_file=data_root + 'val.json',
        classes=classes,
        img_prefix=data_root + 'val/'),
    test=dict(
        ann_file=data_root + 'val.json',
        classes=classes,
        img_prefix=data_root + 'val/'))

# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
load_from = 'checkpoints/mask_base3.pth'

runner = dict(type='EpochBasedRunner', max_epochs=12)
