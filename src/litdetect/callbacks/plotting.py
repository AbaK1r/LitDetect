import colorsys
import math
import random
from typing import Union, Optional, Dict

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


def plot_images(
        images: Union[np.ndarray, torch.Tensor],
        batch_idx: Union[np.ndarray, torch.Tensor],
        cls: Union[np.ndarray, torch.Tensor],
        bboxes: Union[np.ndarray, torch.Tensor] = np.zeros((0, 4), dtype=np.float32),
        confs: Optional[Union[np.ndarray, torch.Tensor]] = None,
        fname: str = "images.jpg",
        names: Optional[Dict[int, str]] = None,
        max_size: int = 1920,
        max_subplots: int = 16,
        save: bool = True,
        conf_thres: float = 0.25,
) -> Optional[np.ndarray]:
    """
    Plot image grid with bounding boxes and class labels using only PIL (支持中文)
    - 修复了图像网格排列方式（从左到右，从上到下）
    - 移除了多余的行
    - 修复了边界框坐标处理逻辑
    - 添加了置信度显示
    - 使用固定且可区分的颜色方案
    - 优化了文本渲染和布局
    """

    def tensor_to_numpy(_x):
        if isinstance(_x, np.ndarray):
            return _x
        return _x.cpu().numpy()

    images = tensor_to_numpy(images)
    batch_idx = tensor_to_numpy(batch_idx)
    cls = tensor_to_numpy(cls)
    bboxes = tensor_to_numpy(bboxes).astype(np.float64)
    confs = tensor_to_numpy(confs) if confs is not None else None

    bs, _, h, w = images.shape
    bs = min(bs, max_subplots)

    if bboxes.shape[0] > 0:
        if bboxes.max() <= 1.0:
            bboxes = np.clip(bboxes, 0, 1)
        else:
            bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, w)
            bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, h)


    # 计算网格尺寸（按行排列）
    ns = math.ceil(math.sqrt(bs))  # 网格尺寸 (ns x ns)
    n_rows = min(ns, math.ceil(bs / ns))  # 实际需要的行数
    n_cols = min(ns, math.ceil(bs / n_rows)) if n_rows > 0 else 0  # 实际需要的列数

    # Denormalize
    if images.max() <= 1:
        images *= 255
    images = images.astype(np.uint8)

    # 创建马赛克图像（按行排列）
    mosaic_h = n_rows * h
    mosaic_w = n_cols * w
    mosaic = np.full((mosaic_h, mosaic_w, 3), 0, dtype=np.uint8)

    # 填充马赛克图像（从左到右，从上到下）
    for i in range(bs):
        row = i // n_cols  # 行索引
        col = i % n_cols  # 列索引
        y_start = row * h
        x_start = col * w
        mosaic[y_start:y_start + h, x_start:x_start + w, :] = images[i].transpose(1, 2, 0)

    # 计算缩放比例
    scale = min(1.0, max_size / mosaic_w, max_size / mosaic_h)

    # 应用缩放
    if scale < 1:
        new_w = int(mosaic_w * scale)
        new_h = int(mosaic_h * scale)
        mosaic_img = Image.fromarray(mosaic).resize((new_w, new_h), Image.BILINEAR)
        mosaic = np.array(mosaic_img)
        mosaic_w, mosaic_h = new_w, new_h
        scale_x = scale
        scale_y = scale
    else:
        scale_x = 1.0
        scale_y = 1.0

    # 转换为PIL图像进行绘制
    pil_img = Image.fromarray(mosaic)
    draw = ImageDraw.Draw(pil_img)

    # 加载字体（支持中文）
    try:
        font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
        font_size = max(10, int(20 * min(scale_x, scale_y)))
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"[Warning] Failed to load font {font_path}: {e}, using default font.")
        font = ImageFont.load_default()

    # 为每个类别生成固定颜色
    unique_classes = np.unique(cls)
    color_map = {}
    for c in unique_classes:
        # 使用HSV颜色空间生成可区分颜色
        hue = (int(c) * 0.618) % 1.0  # 黄金比例确保颜色分布均匀
        saturation = 0.7 + random.random() * 0.3
        value = 0.7 + random.random() * 0.3
        r, g, b = [int(255 * x) for x in colorsys.hsv_to_rgb(hue, saturation, value)]
        color_map[int(c)] = (r, g, b)

    # 计算每个原始图像在网格中的位置
    for i in range(bs):
        row = i // n_cols
        col = i % n_cols

        # 计算当前图像在网格中的位置（缩放后）
        x_offset = col * w * scale_x
        y_offset = row * h * scale_y

        # 获取当前图像的所有检测结果
        idx = batch_idx == i
        if not idx.any():
            continue

        classes = cls[idx]
        conf = confs[idx] if confs is not None else [None] * len(classes)
        boxes = bboxes[idx]

        # 处理边界框坐标
        # 如果坐标是归一化的，转换为绝对坐标
        if boxes[:, :4].max() <= 1.1:
            boxes[:, [0, 2]] *= w  # 乘以原始宽度
            boxes[:, [1, 3]] *= h  # 乘以原始高度

        # 应用整体缩放
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

        # 应用位置偏移
        boxes[:, [0, 2]] += x_offset
        boxes[:, [1, 3]] += y_offset

        for j, box in enumerate(boxes):
            if conf is not None and conf[j] < conf_thres:
                continue

            c = int(classes[j])
            class_name = names.get(c, str(c)) if names else str(c)

            # 添加置信度（如果有）
            label = f"{class_name} {conf[j]:.2f}" if conf is not None else class_name

            color = color_map.get(c, (0, 0, 255))  # 使用固定颜色

            x1, y1, x2, y2 = map(int, box[:4])

            # 确保坐标在图像范围内
            x1 = max(0, min(x1, mosaic_w - 1))
            y1 = max(0, min(y1, mosaic_h - 1))
            x2 = max(0, min(x2, mosaic_w - 1))
            y2 = max(0, min(y2, mosaic_h - 1))

            # 绘制边界框
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # 计算文本大小
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]

            # 调整文本位置防止超出图像边界
            text_x = max(0, min(x1, mosaic_w - text_w))
            text_y = max(0, y1 - text_h - 5)

            # 绘制文本背景
            draw.rectangle(
                [text_x, text_y, text_x + text_w, text_y + text_h + 5],
                fill=color
            )

            # 绘制文本
            draw.text(
                (text_x, text_y),
                label,
                fill=(255, 255, 255),
                font=font
            )

    # 保存或返回结果
    if not save:
        return np.array(pil_img)

    pil_img.save(fname)
    return None


if __name__ == "__main__":
    # 模拟数据：1张图片，3通道，640x640
    images = (np.ones((8, 3, 320, 640)) * 0.5).astype(np.float32)
    batch_idx = np.array([0, 3])
    cls = np.array([0, 1])
    bboxes = np.array([[200, 200, 300, 600], [100, 200, 300, 600]])  # 归一化坐标
    confs = np.array([0.9, 0.8])
    # batch_idx = np.array([])
    # cls = np.array([])
    # bboxes = np.array([])  # 归一化坐标
    # confs = np.array([])

    names = {0: "小肉小肉小肉小肉小肉", 1: "大肉大肉大肉大肉大肉"}

    plot_images(
        images=images,
        batch_idx=batch_idx,
        cls=cls,
        bboxes=bboxes,
        confs=confs,
        fname="chinese_output.png",
        names=names,
        save=True
    )