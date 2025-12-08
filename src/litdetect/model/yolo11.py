import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import ops

from litdetect.model.model_interface import ModuleInterface
from litdetect.scripts_init import get_logger

logger = get_logger(__file__)


class ModuleWrapper(ModuleInterface):

    @property
    def model_class(self):
        return Yolo11

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def input_batch_trans(batch):
        """

        Args:
            batch: [{
                'image':           # CHW, float32, nomalized
                'bboxes':          # (n, 4) (x1, y1, x2, y2)
                'labels':          # (n,)
                'orig_size':       # (2)
                'input_size_hw':   # (2)
                'image_id':        # scalar int/long
            }]

        Returns:

        """
        device = batch[0]["image"].device
        w, h = batch[0]["input_size_hw"][1], batch[0]["input_size_hw"][0]

        images = [b["image"][None] for b in batch]
        images = torch.cat(images, dim=0)

        bboxes_list, labels_list, batch_idx_list = [], [], []

        for i, b in enumerate(batch):
            if len(b["bboxes"]) == 0:
                continue

            bboxes_list.append(b["bboxes"])
            labels_list.append(b["labels"].view(-1, 1))
            batch_idx_list.append(
                torch.full((len(b["labels"]),),
                           i,
                           device=device,
                           dtype=torch.long)
            )

        if len(bboxes_list):
            bboxes = torch.cat(bboxes_list, dim=0).float()
            cls = torch.cat(labels_list, dim=0).long()
            batch_idx = torch.cat(batch_idx_list, dim=0).long()
        else:
            bboxes = torch.empty((0, 4), device=device)
            cls = torch.empty((0, 1), device=device, dtype=torch.long)
            batch_idx = torch.empty((0,), device=device, dtype=torch.long)

        # xyxy -> xywh (norm)
        if bboxes.numel() > 0:
            x = (bboxes[:, 0] + bboxes[:, 2]) / 2 / w
            y = (bboxes[:, 1] + bboxes[:, 3]) / 2 / h
            bw = (bboxes[:, 2] - bboxes[:, 0]) / w
            bh = (bboxes[:, 3] - bboxes[:, 1]) / h
            bboxes = torch.stack([x, y, bw, bh], dim=1)

        return {
            "img": images,
            "bboxes": bboxes,
            "cls": cls,
            "batch_idx": batch_idx
        }


class Yolo11(nn.Module):

    yaml = {
        'nc': -1,
        'names': [],
        'scale': 'm',
        'scales': {'n': [0.5, 0.25, 1024],
                   's': [0.5, 0.5, 1024],
                   'm': [0.5, 1.0, 512],
                   'l': [1.0, 1.0, 512],
                   'x': [1.0, 1.5, 512]},
        'backbone': [[-1, 1, 'Conv', [64, 3, 2]],
                     [-1, 1, 'Conv', [128, 3, 2]],
                     [-1, 2, 'C3k2', [256, False, 0.25]],
                     [-1, 1, 'Conv', [256, 3, 2]],
                     [-1, 2, 'C3k2', [512, False, 0.25]],
                     [-1, 1, 'Conv', [512, 3, 2]],
                     [-1, 2, 'C3k2', [512, True]],
                     [-1, 1, 'Conv', [1024, 3, 2]],
                     [-1, 2, 'C3k2', [1024, True]],
                     [-1, 1, 'SPPF', [1024, 5]],
                     [-1, 2, 'C2PSA', [1024]]],
        'head': [[-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
                 [[-1, 6], 1, 'Concat', [1]],
                 [-1, 2, 'C3k2', [512, False]],
                 [-1, 1, 'nn.Upsample', ['None', 2, 'nearest']],
                 [[-1, 4], 1, 'Concat', [1]],
                 [-1, 2, 'C3k2', [256, False]],
                 [-1, 1, 'Conv', [256, 3, 2]],
                 [[-1, 13], 1, 'Concat', [1]],
                 [-1, 2, 'C3k2', [512, False]],
                 [-1, 1, 'Conv', [512, 3, 2]],
                 [[-1, 10], 1, 'Concat', [1]],
                 [-1, 2, 'C3k2', [1024, True]],
                 [[16, 19, 22], 1, 'Detect', ['nc']]],
    }

    def __init__(self, num_classes, class_name=None, scale='m', iou_thres=0.45, conf_thres=0.25, input_size_hw=(389, 672), max_det=300):
        super().__init__()
        if class_name is None:
            class_name = [f'class_{i}' for i in range(num_classes)]
        assert num_classes == len(class_name), f"num_classes ({num_classes}) must equal to len(class_name) ({len(class_name)})"
        assert len(scale) == 1 and scale in "nsmlx", f"scale must be one of 'nsmlx', but got {scale}"
        self.yaml['nc'] = num_classes
        self.yaml['names'] = class_name
        self.yaml['scale'] = scale

        self.h, self.w = input_size_hw
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.max_det = max_det

        self.model = DetectionModel(self.yaml, verbose=False)
        self.model.args = DEFAULT_CFG
        self.model.args.imgsz = input_size_hw

    def forward(self, images):
        """

        Args:
            images: [batch_size, 3, h, w]

        Returns: [batch_size, num_proposals, xyxy+scores]

        """
        preds = self.model(images)[0].transpose(-1, -2)
        preds[..., :4] = ops.xywh2xyxy(preds[..., :4])  # xywh to xyxy (bs, n, 4 + nc)
        return preds

    def train_step(self, batch):
        loss, loss_items = self.model(batch)
        return {
            'loss': loss.mean(),
            'box_loss': loss_items[0],
            'cls_loss': loss_items[1],
            'dfl_loss': loss_items[2],
        }

    def val_step(self, batch):
        preds = self.model(batch["img"], augment=False)
        results = ops.non_max_suppression(
            preds,
            self.conf_thres,
            self.iou_thres,
            labels=[],
            multi_label=False,
            agnostic=False,
            max_det=self.max_det,
        )
        return results
