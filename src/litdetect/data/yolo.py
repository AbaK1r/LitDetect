import warnings
from pathlib import Path
from typing import List

import cv2

cv2.setNumThreads(0)
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A

from .scale_tookits import xywh2xyxy

warnings.filterwarnings('ignore', message='loadtxt: input contained no data')


class Yolo(Dataset):
    def __init__(self, ano_root, image_root, input_size_hw=(672, 389), data_mode='train'):
        self.ano_root = ano_root
        self.image_root = image_root

        self.train = data_mode == 'train'
        self.input_size_hw = input_size_hw

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
                # A.ElasticTransform(
                #     alpha=300,
                #     sigma=10,
                #     interpolation=cv2.INTER_LINEAR,
                #     approximate=False,
                #     same_dxdy=True,
                #     mask_interpolation=cv2.INTER_NEAREST,
                #     noise_distribution="gaussian",
                #     keypoint_remapping_method="mask",
                #     border_mode=cv2.BORDER_CONSTANT,
                #     fill=0,
                #     fill_mask=0,
                #     p=0.5,
                # ),
                A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.1, 0.25),
                               hole_width_range=(0.1, 0.25), p=0.2),
                # A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(p=0.2),
                A.Sequential([
                    A.SmallestMaxSize(max_size=int(max(self.input_size_hw) * 1.25), p=1.0),
                    A.RandomCrop(height=self.input_size_hw[1], width=self.input_size_hw[0], p=1.0)
                ], p=0.1),
                A.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)),
                A.ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.5))
        else:
            self.transforms = A.Compose([
                A.LongestMaxSize(max_size_hw=input_size_hw, area_for_downscale="image", p=1.0),
                A.PadIfNeeded(min_height=input_size_hw[0], min_width=input_size_hw[1],
                              border_mode=cv2.BORDER_CONSTANT, fill=0, p=1.0),
                A.Normalize(mean=(0.430, 0.411, 0.296), std=(0.213, 0.156, 0.143)),
                A.ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

        self.item_list: List[dict] = []
        for ano_path in (Path(self.ano_root) / ('train2017' if self.train else 'val2017')).glob('*'):
            image_path = Path(self.image_root) / ('train2017' if self.train else 'val2017') / (ano_path.stem + '.png')
            if not image_path.exists():
                continue
            _ano = np.loadtxt(ano_path, dtype=np.float32)

            if len(_ano) == 0:
                # _ano = [[], []]
                continue
            else:
                if len(_ano.shape) == 1:
                    _ano = _ano[None]
                _ano = [_ano[:, 0].astype(int), _ano[:, 1:]]
            _item = {'image_path': str(image_path), 'annotation': _ano}
            self.item_list.append(_item)
        if len(self.item_list) == 0:
            raise FileNotFoundError(f'No data found in {self.ano_root}')

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        meta = self.item_list[idx]
        image = cv2.imread(meta['image_path'])
        if image is None:
            raise FileNotFoundError(f'{meta["image_path"]} not found')

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)

        labels, bboxes = meta['annotation']

        y_size, x_size = image.shape[:2]
        bboxes = [xywh2xyxy([i[0] * x_size, i[1] * y_size, i[2] * x_size, i[3] * y_size]) for i in bboxes]

        sample = self.transforms(image=image, bboxes=bboxes, labels=labels)

        image = sample['image']
        bboxes = sample['bboxes']
        labels = sample['labels']

        target = {
            "boxes": torch.tensor(bboxes).float(),
            "labels": torch.tensor(labels),
            "area": torch.tensor([(i[3] - i[1]) * (i[2] - i[0]) for i in bboxes]).float(),
            "iscrowd": torch.zeros((len(bboxes),), dtype=torch.int64),
            'image_id': torch.tensor([idx]).float(),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
