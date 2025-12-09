import warnings
from pathlib import Path

import albumentations as A
import numpy as np
import torch
from PIL import Image
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset

warnings.filterwarnings('ignore', message='loadtxt: input contained no data')


class Od(Dataset):
    def __init__(self, ano_root, image_root, input_size=(672, 389), data_mode='train', normed_bbox=False, bbox_format='yolo'):
        self.normed_bbox = normed_bbox
        self.bbox_format = bbox_format

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
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels'], min_area=0.1, min_visibility=0.1))
        else:
            self.transforms = self.transforms= A.Compose([
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))

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
        image = np.array(image.resize(self.input_size))
        labels, bboxes = meta['annotation']

        sample = self.transforms(image=image, bboxes=bboxes, labels=labels)
        bboxes = sample['bboxes']  # (x, y, w, h)
        if not self.normed_bbox:
            x_size, y_size = image.shape[1], image.shape[0]
            bboxes = [[i[0] * x_size, i[1] * y_size, i[2] * x_size, i[3] * y_size] for i in bboxes]

        voc_bboxes = [[i[0] - i[2]/2, i[1] - i[3]/2, i[0] + i[2]/2, i[1] + i[3]/2] for i in bboxes]  # (xmin, ymin, xmax, ymax)
        if self.bbox_format != 'yolo':
            bboxes = voc_bboxes
        image = sample['image'].float() / 255.
        target = {
            "boxes": torch.tensor(bboxes).float(),
            "labels": torch.tensor(sample['labels']),
            "area": torch.tensor([(i[3] - i[1]) * (i[2] - i[0]) for i in bboxes]).float(),
            "iscrowd": torch.zeros((len(bboxes),), dtype=torch.int64),
            'image_id': torch.tensor([idx]).float(),
        }
        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))
