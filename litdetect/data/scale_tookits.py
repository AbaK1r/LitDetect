from typing import List, Union, Dict, Tuple

import numpy as np
import torch
from PIL import Image


def xyxy2xywh(
    bbox: Union[List[float], tuple]
) -> List[float]:
    """
    将边界框坐标从 xyxy 格式 (x1, y1, x2, y2) 转换为 xywh 格式 (cx, cy, w, h)

    Args:
        bbox: 长度为4的列表或元组，表示 [x1, y1, x2, y2]

    Returns:
        转换后的 [cx, cy, w, h]
    """
    x1, y1, x2, y2 = map(float, bbox)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]

def xywh2xyxy(
    bbox: Union[List[float], tuple]
) -> List[float]:
    """
    将边界框坐标从 xywh 格式 (cx, cy, w, h) 转换为 xyxy 格式 (x1, y1, x2, y2)

    Args:
        bbox: 长度为4的列表或元组，表示 [cx, cy, w, h]

    Returns:
        转换后的 [x1, y1, x2, y2]
    """
    cx, cy, w, h = map(float, bbox)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]

def letterbox(image, target_size, fill=128):
    """
    保持比例缩放图像并在两侧填充指定颜色
    :param image: PIL.Image 输入图像
    :param target_size: (width, height) 目标尺寸
    :param fill: 填充颜色值（RGB单通道）
    :return:
        new_image: 缩放并填充后的PIL.Image
        scale_params: 包含缩放信息的字典，必须包含:
            'ratio': 缩放比例 (float)
            'pad': (left, top) 填充偏移量 (Tuple[float, float])
            'orig_size': (orig_w, orig_h) 原始的图像尺寸 (Tuple[int, int])
            'target_size': (target_w, target_h) 缩放+填充后的图像尺寸 (Tuple[int, int])
    """
    target_w, target_h = target_size
    orig_w, orig_h = image.size

    # 计算缩放比例和新的尺寸
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)

    # 缩放图像
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # 创建新图像并填充
    new_image = Image.new('RGB', (target_w, target_h), (fill, fill, fill))
    left_pad = (target_w - new_w) // 2
    top_pad = (target_h - new_h) // 2
    new_image.paste(image, (left_pad, top_pad))

    return (new_image,
            {
                'ratio': ratio,
                'pad': (left_pad, top_pad),
                'orig_size': (orig_w, orig_h),
                'target_size': (target_w, target_h)
            })

def apply_scale_to_coords(
    bbox: Union[List[float], torch.Tensor, np.ndarray],
    scale_params: Dict[str, Union[float, Tuple[float, float], Tuple[int, int]]],
    box_format: str = 'xyxy'
) -> List[float]:
    """
    将边界框坐标从原始图像映射到缩放+填充后的图像空间

    Args:
        bbox: 边界框坐标，格式由 box_format 指定
        scale_params: 包含缩放信息的字典，必须包含:
            'ratio': 缩放比例 (float)
            'pad': (left, top) 填充偏移量 (Tuple[float, float])
            'target_size': (target_w, target_h) 缩放+填充后的图像尺寸 (Tuple[int, int])
        box_format: 边界框格式，'xyxy' 或 'xywh'

    Returns:
        缩放+填充后图像上的边界框坐标（与输入相同的格式）

    Raises:
        ValueError: 当 box_format 不是 'xyxy' 或 'xywh' 时
    """
    # 参数验证
    if box_format not in ('xyxy', 'xywh'):
        raise ValueError(f"Invalid box_format: '{box_format}'. Must be 'xyxy' or 'xywh'")

    # 解包参数并类型转换
    ratio = float(scale_params['ratio'])
    pad_left, pad_top = map(float, scale_params['pad'])
    target_w, target_h = map(int, scale_params['target_size'])

    # 转换为numpy数组处理
    if torch.is_tensor(bbox):
        bbox = bbox.cpu().numpy()
    bbox = np.asarray(bbox, dtype=np.float32).tolist()

    if box_format == 'xyxy':
        x1, y1, x2, y2 = bbox
    else:  # xywh
        cx, cy, w, h = bbox
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2

    # 应用缩放和填充
    def _convert(val: float, pad: float, limit: float) -> float:
        """先缩放再添加填充，并限制在 [0, limit] 范围内"""
        return max(0.0, min(val * ratio + pad, limit))

    x1_scaled = _convert(x1, pad_left, target_w)
    y1_scaled = _convert(y1, pad_top, target_h)
    x2_scaled = _convert(x2, pad_left, target_w)
    y2_scaled = _convert(y2, pad_top, target_h)

    # 返回与输入相同的格式
    if box_format == 'xyxy':
        return [x1_scaled, y1_scaled, x2_scaled, y2_scaled]
    else:
        return [
            (x1_scaled + x2_scaled) / 2,  # cx
            (y1_scaled + y2_scaled) / 2,  # cy
            x2_scaled - x1_scaled,        # w
            y2_scaled - y1_scaled         # h
        ]

def recover_original_coords(
    bbox: Union[List[float], torch.Tensor, np.ndarray],
    scale_params: Dict[str, Union[float, Tuple[float, float], Tuple[int, int]]],
    box_format: str = 'xyxy'
) -> List[float]:
    """
    将边界框坐标从填充后的图像恢复到原始图像坐标

    Args:
        bbox: 边界框坐标，格式由 box_format 指定
        scale_params: 包含缩放信息的字典，必须包含:
            'ratio': 缩放比例 (float)
            'pad': (left, top) 填充偏移量 (Tuple[float, float])
            'orig_size': (orig_w, orig_h) 原始图像尺寸 (Tuple[int, int])
        box_format: 边界框格式，'xyxy' 或 'xywh'

    Returns:
        原始图像上的边界框坐标（与输入相同的格式）

    Raises:
        ValueError: 当 box_format 不是 'xyxy' 或 'xywh' 时
    """
    # 参数验证
    if box_format not in ('xyxy', 'xywh'):
        raise ValueError(f"Invalid box_format: '{box_format}'. Must be 'xyxy' or 'xywh'")

    # 解包参数并类型转换
    ratio = float(scale_params['ratio'])
    pad_left, pad_top = map(float, scale_params['pad'])
    orig_w, orig_h = map(int, scale_params['orig_size'])

    # 转换为numpy数组处理
    if torch.is_tensor(bbox):
        bbox = bbox.cpu().numpy()
    bbox = np.asarray(bbox, dtype=np.float32).tolist()

    if box_format == 'xyxy':
        x1, y1, x2, y2 = bbox
    else:  # xywh
        cx, cy, w, h = bbox
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2

    # 去除填充并反缩放
    def _convert(val: float, pad: float, limit: float) -> float:
        """去除填充后反缩放，并限制在 [0, limit] 范围内"""
        return max(0.0, min((val - pad) / ratio, limit))

    x1_orig = _convert(x1, pad_left, orig_w)
    y1_orig = _convert(y1, pad_top, orig_h)
    x2_orig = _convert(x2, pad_left, orig_w)
    y2_orig = _convert(y2, pad_top, orig_h)

    # 返回与输入相同的格式
    if box_format == 'xyxy':
        return [x1_orig, y1_orig, x2_orig, y2_orig]
    else:
        return [
            (x1_orig + x2_orig) / 2,  # cx
            (y1_orig + y2_orig) / 2,  # cy
            x2_orig - x1_orig,        # w
            y2_orig - y1_orig         # h
        ]
