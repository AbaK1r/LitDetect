import warnings
from pathlib import Path
from typing import List, Dict, Any

import cv2

cv2.setNumThreads(0)
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from detectron2.structures import Boxes, Instances

from .scale_tookits import xywh2xyxy

warnings.filterwarnings('ignore', message='loadtxt: input contained no data')


class Detr(Dataset):
    def __init__(
        self,
        ano_root: str,
        image_root: str,
        input_size_hw=(672, 389),  # (h, w), only for *augmentation*, NOT resize to fixed
        data_mode: str = 'train',
    ):
        self.ano_root = Path(ano_root)
        self.image_root = Path(image_root)
        self.train = data_mode == 'train'
        self.input_size_hw = input_size_hw  # used for augmentation only

        # Detectron2 默认输入是 [0, 255] uint8 float32，且不 Normalize/ToTensor in dataset
        # 此处仅做几何增强
        if self.train:
            self.transforms = A.Compose([
                A.LongestMaxSize(max_size_hw=input_size_hw, area_for_downscale="image", p=1.0),
                A.PadIfNeeded(min_height=input_size_hw[0], min_width=input_size_hw[1],
                              border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
                A.SquareSymmetry(p=0.5),
                A.Affine(
                    scale=(0.8, 1.2),  # Zoom in/out by 80-120%
                    rotate=(-30, 30),  # Rotate by -30 to +30 degrees
                    balanced_scale=True,
                    translate_percent=(0, 0.1), # Optional: translate by 0-10%
                    shear=(-10, 10),          # Optional: shear by -10 to +10 degrees
                    p=0.6
                ),
                A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.1, 0.25),
                               hole_width_range=(0.1, 0.25), p=0.2),
                # A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(p=0.2),
                A.Sequential([
                    A.SmallestMaxSize(max_size=int(max(self.input_size_hw) * 1.25), p=1.0),
                    A.RandomCrop(height=self.input_size_hw[1], width=self.input_size_hw[0], p=1.0)
                ], p=0.1),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.5))
        else:
            self.transforms = A.Compose([
                A.LongestMaxSize(max_size_hw=input_size_hw, area_for_downscale="image", p=1.0),
                A.PadIfNeeded(min_height=input_size_hw[0], min_width=input_size_hw[1],
                              border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

        # Load data list
        self.item_list: List[Dict[str, Any]] = []
        anno_dir = self.ano_root / ('train2017' if self.train else 'val2017')
        image_dir = self.image_root / ('train2017' if self.train else 'val2017')

        for ano_path in anno_dir.glob('*.txt'):
            image_path = image_dir / (ano_path.stem + '.png')
            if not image_path.exists():
                image_path = image_dir / (ano_path.stem + '.jpg')  # fallback
                if not image_path.exists():
                    continue

            _ano = np.loadtxt(ano_path, dtype=np.float32)
            if _ano.size == 0:
                continue
            if _ano.ndim == 1:
                _ano = _ano[None]

            labels = _ano[:, 0].astype(np.int64)
            bboxes_xywh = _ano[:, 1:5]

            self.item_list.append({
                'image_path': str(image_path),
                'labels': labels,
                'bboxes_xywh': bboxes_xywh,
            })

        if len(self.item_list) == 0:
            raise FileNotFoundError(f"No annotation files loaded from {anno_dir}")

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.item_list[idx]
        image = cv2.imread(item['image_path'])
        if image is None:
            raise FileNotFoundError(item['image_path'])

        orig_h, orig_w = image.shape[:2]

        # BGR → RGB, HWC uint8
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        elif len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Convert YOLO xywh (normed) → xyxy (abs)
        bboxes_xywh = item['bboxes_xywh']
        labels = item['labels']
        bboxes_xyxy = np.array([
            xywh2xyxy([cx * orig_w, cy * orig_h,
                       w  * orig_w, h  * orig_h])
            for cx, cy, w, h in bboxes_xywh
        ], dtype=np.float32)

        # Apply transforms (only geometric)
        transformed = self.transforms(image=image, bboxes=bboxes_xyxy, labels=labels)
        aug_image = transformed['image']  # HWC uint8, RGB
        aug_bboxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
        aug_labels = torch.as_tensor(transformed['labels'], dtype=torch.int64)

        # Construct Detectron2 Instances
        instances = Instances((aug_image.shape[0], aug_image.shape[1]))
        if len(aug_bboxes) > 0:
            instances.gt_boxes = Boxes(aug_bboxes)  # (x1, y1, x2, y2)
            instances.gt_classes = aug_labels
        else:
            # empty instances
            instances.gt_boxes = Boxes(torch.zeros((0, 4), dtype=torch.float32))
            instances.gt_classes = torch.zeros(0, dtype=torch.int64)

        # Return Detectron2-style dict
        # Important: image must be CHW float32 in [0, 255]
        image_tensor = torch.as_tensor(aug_image.astype("float32").transpose(2, 0, 1))  # [3, H, W], float32, 0~255

        return {
            "image": image_tensor,                # CHW, float32, [0, 255]
            "height": self.input_size_hw[0],      # height (for evaluation rescaling)
            "width": self.input_size_hw[1],       # width
            "image_id": idx,                      # scalar int/long
            "instances": instances,               # Instances object
        }


def collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For Detectron2-style input: returns List[dict], NOT tuple.
    Each dict has keys: "image", "height", "width", "image_id", "instances".
    Dataloader batch_size = N ⇒ len(batch) = N.
    """
    return batch  # Just return as-is. Detectron2 trainer expects List[dict]


from typing import Any


def summarize_tensor_structure(x: Any, indent: int = 0) -> None:
    """
    递归打印嵌套结构（dict/list/tuple）中各元素的概要信息。
    对 torch.Tensor 只打印 shape 和 dtype；其余类型按 repr 精简显示。

    Args:
        x (Any): 待分析的任意对象（常见为模型 forward 输出）
        indent (int): 缩进层级（递归内部使用）
    """
    prefix = "  " * indent
    if isinstance(x, torch.Tensor):
        # 张量：仅打印 shape 和 dtype
        print(f"Tensor{list(x.shape)} dtype={x.dtype}")
    elif isinstance(x, dict):
        print(f"{prefix}dict({len(x)}) {{")
        for k, v in x.items():
            print(f"{prefix}  {repr(k)}: ", end="")
            summarize_tensor_structure(v, indent + 2)
        print(f"{prefix}}}")
    elif isinstance(x, (list, tuple)):
        typename = type(x).__name__
        print(f"{prefix}{typename}({len(x)}) [")
        for i, item in enumerate(x):
            print(f"{prefix}  [{i}]: ", end="")
            summarize_tensor_structure(item, indent + 2)
        print(f"{prefix}]")
    else:
        # 其他类型：int/float/str/None 等
        # 注意 repr 太长时截断（例如避免长字符串炸屏）
        r = repr(x)
        if len(r) > 80:
            r = r[:77] + "..."
        print(f"{prefix}{r}")