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

__all__ = ['BuspassengerDetection']


@register
class BuspassengerDetection(CocoDetection):
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
    


 


buspassenger_category2name = {
    1: 'buspassenger',
     
}
buspassenger_category2label = {k: i for i, k in enumerate(buspassenger_category2name.keys())}
buspassenger_label2category = {v: k for k, v in buspassenger_category2label.items()}