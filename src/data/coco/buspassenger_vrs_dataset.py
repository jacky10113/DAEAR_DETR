"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

COCO dataset which returns image_id for evaluation.
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""

import torch
import torch.utils.data

import torchvision
torchvision.disable_beta_transforms_warning()

from torchvision import datapoints

from pycocotools import mask as coco_mask

from src.core import register

from .coco_dataset import CocoDetection,ConvertCocoPolysToMask
from src.data.transforms import MosaicTransform
from src.data.transforms import CutMixTransform
from src.data.transforms  import MixUpTransform

__all__ = ['BuspassengerVRSDetection']


@register
class BuspassengerVRSDetection(CocoDetection):
    __inject__ = ['transforms']
    __share__ = ['remap_mscoco_category']
    
    def __init__(self, img_folder, ann_file, transforms, return_masks, remap_mscoco_category=False):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks, remap_mscoco_category)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        # ['boxes', 'masks', 'labels']:
        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'], 
                format=datapoints.BoundingBoxFormat.XYXY, 
                spatial_size=img.size[::-1]) # h w

        if 'masks' in target:
            target['masks'] = datapoints.Mask(target['masks'])

        # 数据增强：按照顺序执行增强变换
        if self._transforms is not None:
            for transform in self._transforms.transforms:
                if isinstance(transform, (MosaicTransform, CutMixTransform,MixUpTransform)):
                    img, target = transform(img, target, self)  # 传入整个数据集
                else:
                    img, target = transform(img, target)
            
        return img, target

    def get_raw_item(self, idx):
        """
        获取未经过变换的原始图像和目标。
        """
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)

        if 'boxes' in target:
            target['boxes'] = datapoints.BoundingBox(
                target['boxes'], 
                format=datapoints.BoundingBoxFormat.XYXY, 
                spatial_size=img.size[::-1]
            )

        if 'masks' in target:
            target['masks'] = datapoints.Mask(target['masks'])

        return img, target
 


buspassenger_category2name = {
    1: 'buspassenger',
     
}
buspassenger_category2label = {k: i for i, k in enumerate(buspassenger_category2name.keys())}
buspassenger_label2category = {v: k for k, v in buspassenger_category2label.items()}