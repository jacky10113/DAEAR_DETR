import torch 
import torch.utils.data as data

from src.core import register


__all__ = ['DataLoader']


@register
class DataLoader(data.DataLoader):
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string



@register
def default_collate_fn(items):
    '''default collate_fn
    '''    
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]

@register
def enhanced_collate_fn(items):
    """
    支持单张图片增强和多张图片增强（Mosaic, CutMix, MixUp等）。
    """
    images = [x[0] for x in items]
    targets = [x[1] for x in items]
    
    # 判断是否有多图增强操作 (如 Mosaic, MixUp 等)
    if isinstance(images[0], list):  # 如果传入的图像是列表，则多图增强
        # Flatten 图像和标签列表
        images = [img for sublist in images for img in sublist]
        targets = [tgt for sublist in targets for tgt in sublist]
        #print("多维增强")
    else:
        #print("单图增强")
        # 单图增强，直接堆叠为 Tensor
        images = torch.stack(images, dim=0)
    
    return images, targets

