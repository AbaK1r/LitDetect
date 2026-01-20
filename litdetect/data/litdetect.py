import cv2
cv2.setNumThreads(0)

import warnings
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from .scale_tookits import xywh2xyxy

warnings.filterwarnings('ignore', message='loadtxt: input contained no data')


class LitDetectDataset(Dataset):
    def __init__(
        self,
        transforms: A.BasicTransform,
        ano_root: str,
        image_root: str,
        input_size_hw=(672, 389),  # (h, w), only for *augmentation*, NOT resize to fixed
        data_mode: str = 'train'
    ):
        self.ano_root = Path(ano_root)
        self.image_root = Path(image_root)
        self.train = data_mode == 'train'
        self.input_size_hw = input_size_hw  # used for augmentation only

        self.transforms = transforms

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
        image = cv2.imread(item['image_path'], cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:
            raise FileNotFoundError(item['image_path'])

        orig_h, orig_w = image.shape[:2]

        # BGR → RGB, HWC uint8
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        # Convert YOLO xywh (normed) → xyxy (abs)
        bboxes_xywh = item['bboxes_xywh']
        labels = item['labels']
        bboxes_xyxy = np.array([
            xywh2xyxy([cx * orig_w, cy * orig_h,
                       w  * orig_w, h  * orig_h])
            for cx, cy, w, h in bboxes_xywh
        ], dtype=np.float32).clip(min=0)

        # Apply transforms 一般把归一化操作放到模型中，
        transformed = self.transforms(image=image, bboxes=bboxes_xyxy, labels=labels)
        aug_image = transformed['image']
        if isinstance(aug_image, torch.Tensor):
            aug_image = aug_image.to(torch.float32)
        else:
            aug_image = torch.tensor(aug_image, dtype=torch.float32)
        aug_bboxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        aug_labels = torch.tensor(transformed['labels'], dtype=torch.int64)

        if len(aug_bboxes) > 0:
            bboxes = aug_bboxes  # (n, 4) (x1, y1, x2, y2)
            labels = aug_labels
        else:
            bboxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)

        return {
            'image': aug_image,                                         # transformed image
            'bboxes': bboxes,                                           # (n, 4) (x1, y1, x2, y2)
            'labels': labels,                                           # (n,) torch.int64
            'orig_size': (orig_h, orig_w),                              # (2)
            'input_size_hw': self.input_size_hw,                        # (2)
            'image_id': torch.tensor([idx], dtype=torch.int64),    # scalar int
        }


def collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For Detectron2-style input: returns List[dict], NOT tuple.
    Each dict has keys: "image", "height", "width", "image_id", "instances".
    Dataloader batch_size = N ⇒ len(batch) = N.
    """
    return batch  # Just return as-is. Detectron2 trainer expects List[dict]


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
