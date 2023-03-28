_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

custom_imports = dict(
    imports=[
        'mmdet.core.hook.hk'
    ],
    allow_failed_imports=False)

debug = False

model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=5),
        mask_head=dict(num_classes=5)))

dataset_type = 'CocoDataset'
classes = ('熔珠-焊缝边缘','熔珠','熔钉','电解液','轻微熔钉')

data_root = 'data/tfw/KUNKUN/z/'

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

custom_hooks = [dict(type='MyHook'), dict(type='NumClassCheckHook')]

# 我们可以使用预训练的 Mask R-CNN 来获取更好的性能
# load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

runner = dict(type='EpochBasedRunner', max_epochs=12)
