""""by lyuwenyu
"""

import random
import numpy as np
import torch 
import torch.nn as nn 
import cv2
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision import datapoints

import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from PIL import Image 
from typing import Any, Dict, List, Optional

from src.core import register, GLOBAL_CONFIG
from typing import List, Tuple, Dict, Any


__all__ = ['Compose', ]


RandomPhotometricDistort = register(T.RandomPhotometricDistort)
RandomZoomOut = register(T.RandomZoomOut)
# RandomIoUCrop = register(T.RandomIoUCrop)
RandomHorizontalFlip = register(T.RandomHorizontalFlip)
Resize = register(T.Resize)
ToImageTensor = register(T.ToImageTensor)
ConvertDtype = register(T.ConvertDtype)
SanitizeBoundingBox = register(T.SanitizeBoundingBox)
RandomCrop = register(T.RandomCrop)
Normalize = register(T.Normalize)

 
@register
class Compose(T.Compose):
    def __init__(self, ops) -> None:
        transforms = []
        if ops is not None:
            for op in ops:
                if isinstance(op, dict):
                    name = op.pop('type')
                    transfom = getattr(GLOBAL_CONFIG[name]['_pymodule'], name)(**op)
                    transforms.append(transfom)
                    # op['type'] = name
                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError('')
        else:
            transforms =[EmptyTransform(), ]
 
        super().__init__(transforms=transforms)

    def __call__(self, img, target):
        for transform in self.transforms:
            if isinstance(transform, (MosaicTransform, CutMixTransform,MixUpTransform)):
                img, target = transform(img, target, self.dataset)  # 确保传递数据集
            else:
                img, target = transform(img, target)
        return img, target
 

@register
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register
class PadToSize(T.Pad):
    _transformed_types = (
        Image.Image,
        datapoints.Image,
        datapoints.Video,
        datapoints.Mask,
        datapoints.BoundingBox,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sz = F.get_spatial_size(flat_inputs[0])
        h, w = self.spatial_size[0] - sz[0], self.spatial_size[1] - sz[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, spatial_size, fill=0, padding_mode='constant') -> None:
        if isinstance(spatial_size, int):
            spatial_size = (spatial_size, spatial_size)
        
        self.spatial_size = spatial_size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:        
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p 

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register
class ConvertBox(T.Transform):
    _transformed_types = (
        datapoints.BoundingBox,
    )
    def __init__(self, out_fmt='', normalize=False) -> None:
        super().__init__()
        self.out_fmt = out_fmt
        self.normalize = normalize

        self.data_fmt = {
            'xyxy': datapoints.BoundingBoxFormat.XYXY,
            'cxcywh': datapoints.BoundingBoxFormat.CXCYWH
        }

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:  
        if self.out_fmt:
            spatial_size = inpt.spatial_size
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.out_fmt)
            inpt = datapoints.BoundingBox(inpt, format=self.data_fmt[self.out_fmt], spatial_size=spatial_size)
        
        if self.normalize:
            inpt = inpt / torch.tensor(inpt.spatial_size[::-1]).tile(2)[None]

        return inpt

@register
class MosaicTransform(T.Transform):
    def __init__(self, input_size: List[int], p: float = 0.3):
        """
        初始化 MosaicTransform。

        Args:
            input_size (List[int]): 输出图像的大小 [宽度, 高度]。
            p (float): 应用 Mosaic 增强的概率。
        """
        super().__init__()
        self.input_size = input_size  # [960, 576]
        self.p = p
        self.num_images = 4  # 拼接的图像数量

    def __call__(self, img: Image.Image, target: Dict[str, Any], dataset: Any) -> (Image.Image, Dict[str, Any]):
        """
        执行 Mosaic 增强。

        Args:
            img (Image.Image): 当前图像。
            target (Dict[str, Any]): 当前图像的目标（如边界框、标签等）。
            dataset (Any): 数据集对象，用于随机采样其他图像。

        Returns:
            (Image.Image, Dict[str, Any]): 增强后的图像及其目标。
        """
        if random.random() > self.p:
            return img, target

        # 确保数据集中有足够的图像进行拼接
        if len(dataset) < self.num_images:
            return img, target

        # 随机选择3个其他图像
        indices = random.sample(range(len(dataset)), self.num_images - 1)
        imgs = [img]
        targets_list = [target]
        for idx in indices:
            other_img, other_target = dataset.get_raw_item(idx)  # 使用 get_raw_item 获取原始图像
            imgs.append(other_img)
            targets_list.append(other_target)

        # 定义输出图像的大小
        out_w, out_h = self.input_size

        # 计算每个子图像的大小（假设拼接为2x2网格）
        sub_w = out_w // 2
        sub_h = out_h // 2

        # 创建一个空白画布
        mosaic_img = Image.new('RGB', (out_w, out_h), (114, 114, 114))  # 使用灰色填充

        # 初始化合并后的目标
        merged_boxes = []
        merged_labels = []
        merged_masks = []

        # 定义每个子图像的位置
        positions = [
            (0, 0),               # 左上角
            (sub_w, 0),           # 右上角
            (0, sub_h),           # 左下角
            (sub_w, sub_h),       # 右下角
        ]

        for i in range(self.num_images):
            # 转换图像模式并调整大小
            img_i = imgs[i].convert('RGB').resize((sub_w, sub_h), Image.BILINEAR)
            
            # 获取粘贴位置
            left, upper = positions[i]
            right, lower = left + sub_w, upper + sub_h
            
            # 使用四元组指定粘贴区域
            mosaic_img.paste(img_i, (left, upper, right, lower))

            target_i = targets_list[i]
            
            # 处理边界框
            if 'boxes' in target_i:
                boxes = target_i['boxes'].data.clone()
                spatial_size = target_i['boxes'].spatial_size  # (H, W)
                scale_x = sub_w / spatial_size[1]
                scale_y = sub_h / spatial_size[0]
                boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x + left
                boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y + upper
                merged_boxes.append(boxes)
                merged_labels.append(target_i['labels'])
            
            # 处理掩码
            if 'masks' in target_i:
                masks = target_i['masks'].data.clone()
                # 调整掩码大小
                masks_resized = F.resize(masks, (sub_h, sub_w))
                # 创建一个空白掩码并粘贴调整后的掩码
                blank_mask = torch.zeros((masks_resized.size(0), out_h, out_w), dtype=masks_resized.dtype)
                blank_mask[:, upper:upper+sub_h, left:left+sub_w] = masks_resized
                merged_masks.append(blank_mask)

        # 合并所有边界框和标签
        if merged_boxes:
            merged_boxes = torch.cat(merged_boxes, dim=0)
            merged_labels = torch.cat(merged_labels, dim=0)
            merged_target = {
                'boxes': datapoints.BoundingBox(
                    merged_boxes, 
                    format=datapoints.BoundingBoxFormat.XYXY, 
                    spatial_size=(out_h, out_w)
                ),
                'labels': merged_labels,
                'image_id': target.get('image_id', -1),
            }
        else:
            merged_target = target

        # 合并所有掩码
        if merged_masks:
            merged_masks = torch.cat(merged_masks, dim=0)
            merged_target['masks'] = datapoints.Mask(merged_masks)
        
        return mosaic_img, merged_target

@register
class CutMixTransform(T.Transform):
    def __init__(self, input_size: List[int], p: float = 0.3):
        """
        初始化 CutMixTransform。

        Args:
            input_size (List[int]): 输出图像的大小 [宽度, 高度]。
            p (float): 应用 CutMix 增强的概率。
        """
        super().__init__()
        self.input_size = input_size  # [960, 576]
        self.p = p

    def __call__(self, img: Image.Image, target: Dict[str, Any], dataset: Any) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        执行 CutMix 增强。

        Args:
            img (Image.Image): 当前图像。
            target (Dict[str, Any]): 当前图像的目标（如边界框、标签等）。
            dataset (Any): 数据集对象，用于随机采样另一张图像。

        Returns:
            Tuple[Image.Image, Dict[str, Any]]: 增强后的图像及其目标。
        """
        if random.random() > self.p:
            return img, target

        # 确保数据集中有足够的图像进行混合
        if len(dataset) < 2:
            return img, target

        # 随机选择另一张图像
        idx = random.randint(0, len(dataset) - 1)
        other_img, other_target = dataset.get_raw_item(idx)  # 使用 get_raw_item 获取原始图像

        # 定义输出图像的大小
        out_w, out_h = self.input_size

        # 调整两张图像的大小
        img = img.resize((out_w, out_h), Image.BILINEAR)
        other_img = other_img.resize((out_w, out_h), Image.BILINEAR)

        # 随机生成一个裁剪区域
        lam = random.uniform(0.3, 0.7)  # 控制裁剪区域的大小
        cut_w = int(out_w * lam)
        cut_h = int(out_h * lam)

        # 随机选择裁剪区域的位置
        cx = random.randint(0, out_w - cut_w)
        cy = random.randint(0, out_h - cut_h)
        bbox_cut = (cx, cy, cx + cut_w, cy + cut_h)

        # 创建新的空白图像
        new_img = img.copy()
        new_img.paste(other_img.crop(bbox_cut), (cx, cy, cx + cut_w, cy + cut_h))

        # 合并目标
        merged_boxes = []
        merged_labels = []

        # 处理原图的目标
        if 'boxes' in target:
            boxes = target['boxes'].data.clone()
            # 保留在裁剪区域之外的边界框
            mask = ~self._boxes_inside_cut(boxes, bbox_cut)
            boxes = boxes[mask]
            labels = target['labels'][mask]
            merged_boxes.append(boxes)
            merged_labels.append(labels)

        # 处理另一张图像的目标
        if 'boxes' in other_target:
            boxes = other_target['boxes'].data.clone()
            labels = other_target['labels'].clone()

            # 将边界框坐标平移到新的位置
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * out_w / other_img.width
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * out_h / other_img.height

            # 保留在裁剪区域内的边界框，并调整其位置
            mask = self._boxes_inside_cut(boxes, bbox_cut)
            boxes = boxes[mask]
            labels = labels[mask]

            # 将边界框移到裁剪区域的位置
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(min=cx, max=cx + cut_w) - cx
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(min=cy, max=cy + cut_h) - cy

            # 仅保留有面积的边界框
            valid = (boxes[:, 2] - boxes[:, 0] > 1) & (boxes[:, 3] - boxes[:, 1] > 1)
            boxes = boxes[valid]
            labels = labels[valid]

            merged_boxes.append(boxes)
            merged_labels.append(labels)

        if merged_boxes:
            merged_boxes = torch.cat(merged_boxes, dim=0)
            merged_labels = torch.cat(merged_labels, dim=0)
            merged_target = {
                'boxes': datapoints.BoundingBox(
                    merged_boxes,
                    format=datapoints.BoundingBoxFormat.XYXY,
                    spatial_size=(out_h, out_w)
                ),
                'labels': merged_labels,
                'image_id': target.get('image_id', -1),
            }
        else:
            merged_target = target

        # 处理掩码（如果存在）
        if 'masks' in target or 'masks' in other_target:
            merged_masks = []
            # 处理原图的掩码
            if 'masks' in target:
                masks = target['masks'].data.clone()
                # 保留在裁剪区域之外的掩码
                masks = masks[:, cy:cy + cut_h, cx:cx + cut_w]
                merged_masks.append(masks)

            # 处理另一张图像的掩码
            if 'masks' in other_target:
                masks = other_target['masks'].data.clone()
                # 调整掩码大小
                masks_resized = F.resize(masks, (cut_h, cut_w))
                # 创建一个空白掩码并粘贴调整后的掩码
                blank_mask = torch.zeros((masks_resized.size(0), out_h, out_w), dtype=masks_resized.dtype)
                blank_mask[:, cy:cy + cut_h, cx:cx + cut_w] = masks_resized
                merged_masks.append(blank_mask)

            if merged_masks:
                merged_masks = torch.cat(merged_masks, dim=0)
                merged_target['masks'] = datapoints.Mask(merged_masks)

        return new_img, merged_target

    def _boxes_inside_cut(self, boxes: torch.Tensor, bbox_cut: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        检查边界框是否在裁剪区域内。

        Args:
            boxes (torch.Tensor): 边界框，形状为 [N, 4]。
            bbox_cut (Tuple[int, int, int, int]): 裁剪区域 (cx, cy, cx + cut_w, cy + cut_h)。

        Returns:
            torch.Tensor: 布尔张量，指示每个边界框是否在裁剪区域内。
        """
        cx, cy, cx2, cy2 = bbox_cut
        # 检查边界框是否完全在裁剪区域内
        return (boxes[:, 0] >= cx) & (boxes[:, 1] >= cy) & (boxes[:, 2] <= cx2) & (boxes[:, 3] <= cy2)
    

@register
class MixUpTransform(T.Transform):
    def __init__(self, input_size: List[int], p: float = 0.3, alpha: float = 1.0):
        """
        初始化 MixUpTransform。

        Args:
            input_size (List[int]): 输出图像的大小 [宽度, 高度]。
            p (float): 应用 MixUp 增强的概率。
            alpha (float): Beta 分布的参数，用于生成混合比例 lambda。
        """
        super().__init__()
        self.input_size = input_size  # [960, 576]
        self.p = p
        self.alpha = alpha

    def __call__(self, img1: Image.Image, target1: Dict[str, Any], dataset: Any) -> Tuple[Image.Image, Dict[str, Any]]:
        """
        执行 MixUp 增强。

        Args:
            img1 (Image.Image): 当前图像。
            target1 (Dict[str, Any]): 当前图像的目标（如边界框、标签等）。
            dataset (Any): 数据集对象，用于随机采样另一张图像。

        Returns:
            Tuple[Image.Image, Dict[str, Any]]: 增强后的图像及其目标。
        """
        if random.random() > self.p:
            return img1, target1

        # 确保数据集中有足够的图像进行混合
        if len(dataset) < 2:
            return img1, target1

        # 随机选择另一张图像
        idx = random.randint(0, len(dataset) - 1)
        # 为避免选择同一张图像，您可以启用以下代码（仅在定义了 current_idx 的情况下）
        # while idx == dataset.current_idx:
        #     idx = random.randint(0, len(dataset) - 1)
        img2, target2 = dataset.get_raw_item(idx)  # 使用 get_raw_item 获取原始图像

        # 定义输出图像的大小
        out_w, out_h = self.input_size

        # 调整两张图像的大小
        img1_resized = img1.resize((out_w, out_h), Image.BILINEAR)
        img2_resized = img2.resize((out_w, out_h), Image.BILINEAR)

        # 生成混合比例 lambda
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0

        # 混合图像
        mixed_img = Image.blend(img1_resized, img2_resized, alpha=lam)

        # 合并目标
        merged_boxes = []
        merged_labels = []
        merged_masks = []

        # 处理原图的目标
        if 'boxes' in target1:
            boxes1 = target1['boxes'].data.clone()
            labels1 = target1['labels'].clone()
            merged_boxes.append(boxes1)
            merged_labels.append(labels1)

        # 处理另一张图像的目标
        if 'boxes' in target2:
            boxes2 = target2['boxes'].data.clone()
            labels2 = target2['labels'].clone()
            merged_boxes.append(boxes2)
            merged_labels.append(labels2)

        if merged_boxes:
            merged_boxes = torch.cat(merged_boxes, dim=0)
            merged_labels = torch.cat(merged_labels, dim=0)
            merged_target = {
                'boxes': datapoints.BoundingBox(
                    merged_boxes,
                    format=datapoints.BoundingBoxFormat.XYXY,
                    spatial_size=(out_h, out_w)
                ),
                'labels': merged_labels,
                'image_id': target1.get('image_id', -1),
            }
        else:
            merged_target = target1

        # 处理掩码（如果存在）
        if 'masks' in target1 or 'masks' in target2:
            merged_masks = []
            # 处理原图的掩码
            if 'masks' in target1:
                masks1 = target1['masks'].data.clone()
                merged_masks.append(masks1)

            # 处理另一张图像的掩码
            if 'masks' in target2:
                masks2 = target2['masks'].data.clone()
                merged_masks.append(masks2)

            if merged_masks:
                merged_masks = torch.cat(merged_masks, dim=0)
                merged_target['masks'] = datapoints.Mask(merged_masks)

        return mixed_img, merged_target

