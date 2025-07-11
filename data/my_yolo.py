import warnings
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

from .scale_tookits import xywh2xyxy, letterbox, apply_scale_to_coords

warnings.filterwarnings('ignore', message='loadtxt: input contained no data')


class MyYolo(Dataset):
    def __init__(self, ano_root, image_root, input_size=(672, 389), data_mode='train'):
        self.ano_root = ano_root
        self.image_root = image_root

        self.train = data_mode == 'train'
        self.input_size = input_size

        if self.train:
            self.transforms= A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(p=0.2),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            self.transforms = self.transforms= A.Compose([
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

        self.item_list = []
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
            _item = {'image_path': image_path, 'annotation': _ano}
            self.item_list.append(_item)
        if len(self.item_list) == 0:
            raise FileNotFoundError(f'No data found in {self.ano_root}')
        
    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        meta = self.item_list[idx]
        image = Image.open(meta['image_path']).convert('RGB')
        labels, bboxes = meta['annotation']

        x_size, y_size = image.size
        bboxes = [xywh2xyxy([i[0] * x_size, i[1] * y_size, i[2] * x_size, i[3] * y_size]) for i in bboxes]

        image, scale_params = letterbox(image, self.input_size, 0)
        bboxes = [apply_scale_to_coords(bbox, scale_params, 'xyxy') for bbox in bboxes]

        sample = self.transforms(image=np.array(image), bboxes=bboxes, labels=labels)

        bboxes = sample['bboxes']
        image = sample['image'].float() / 255.
        target = {
            "boxes": torch.tensor(bboxes).float(),
            "labels": torch.tensor(sample['labels']),
            "area": torch.tensor([(i[3] - i[1]) * (i[2] - i[0]) for i in bboxes]).float(),
            "iscrowd": torch.zeros((len(bboxes),), dtype=torch.int64),
            'image_id': torch.tensor([idx]).float(),
        }
        target.update({k: torch.tensor(v) for k, v in scale_params.items()})
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
