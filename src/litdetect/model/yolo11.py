import torch
import torch.nn as nn
# from torch import distributed as dist

from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils import ops


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

    def __init__(self, num_classes, class_name=None, scale='m', iou_thres=0.45, conf_thres=0.25, input_size=(672, 389), max_det=300):
        super().__init__()
        assert num_classes == len(class_name), f"num_classes ({num_classes}) must equal to len(class_name) ({len(class_name)})"
        assert len(scale) == 1 and scale in "nsmlx", f"scale must be one of 'nsmlx', but got {scale}"
        if class_name is None:
            class_name = [f'class_{i}' for i in range(num_classes)]
        self.yaml['nc'] = num_classes
        self.yaml['names'] = class_name
        self.yaml['scale'] = scale

        self.w, self.h = input_size
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.max_det = max_det

        self.model = DetectionModel(self.yaml, verbose=False)
        self.model.args = DEFAULT_CFG
        self.model.args.imgsz = input_size[::-1]

    def forward(self, images):
        """

        Args:
            images: [batch_size, 3, h, w]

        Returns: [batch_size, num_proposals, xyxy+scores]

        """
        preds = self.model(images)[0].transpose(-1, -2)
        preds[..., :4] = ops.xywh2xyxy(preds[..., :4])  # xywh to xyxy (bs, n, 4 + nc)
        return preds

    def preprocess(self, batch):
        images, targets = batch
        data_dic = {
            "img": [i[None] for i in images],
            "bboxes": [targets[idx]["boxes"] for idx in range(len(images))],
            "cls": [targets[idx]["labels"][:, None] for idx in range(len(images))],
            "batch_idx": [torch.full_like(targets[idx]["labels"], idx, device=images[0].device, dtype=torch.long) for idx
                          in range(len(images))]
        }
        for k, v in data_dic.items():
            if len(v) == 0:
                if k == 'bboxes':
                    data_dic[k] = torch.empty((0, 4), dtype=torch.float, device=images[0].device)
                elif k == 'cls':
                    data_dic[k] = torch.empty((0, 1), dtype=torch.long, device=images[0].device)
                elif k == 'batch_idx':
                    data_dic[k] = torch.empty((0,), dtype=torch.long, device=images[0].device)
            elif len(v) == 1:
                data_dic[k] = v[0]
            else:
                data_dic[k] = torch.cat(v, dim=0)
            if k in ['cls', 'batch_idx']:
                data_dic[k] = data_dic[k].long()
        # xyxy2xywh
        if data_dic['bboxes'].shape[0] > 0:
            bboxes_x = (data_dic['bboxes'][:, 0] + data_dic['bboxes'][:, 2]) / 2 / self.w
            bboxes_y = (data_dic['bboxes'][:, 1] + data_dic['bboxes'][:, 3]) / 2 / self.h
            bboxes_w = (data_dic['bboxes'][:, 2] - data_dic['bboxes'][:, 0]) / self.w
            bboxes_h = (data_dic['bboxes'][:, 3] - data_dic['bboxes'][:, 1]) / self.h
            data_dic['bboxes'] = torch.stack([bboxes_x, bboxes_y, bboxes_w, bboxes_h], dim=1)
        return data_dic

    def train_step(self, batch):
        data_dic = self.preprocess(batch)
        loss, loss_items = self.model(data_dic)
        return {
            'loss': loss.mean(),
            'box_loss': loss_items[0],
            'cls_loss': loss_items[1],
            'dfl_loss': loss_items[2],
        }

    def val_step(self, batch):
        images = batch[0]
        # images = [tensor(3, h, w), ...]
        preds = self.model(torch.cat([i[None] for i in images], dim=0), augment=False)
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
