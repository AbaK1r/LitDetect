import random
from pathlib import Path
from typing import List, Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT

from .plotting import plot_images


class PicRecordCallback(pl.Callback):
    def __init__(self, class_names: List[str] | None, conf: float = 0.25, save_pic_dir=None):
        super().__init__()
        self.data_dic = None
        self.class_names = {k: v for k, v in enumerate(class_names)}
        self.conf = conf
        self.save_pic_dir = save_pic_dir

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.data_dic = dict(images=[], batch_idx=[], cls=[], bboxes=[], confs=[])

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        if trainer.global_rank == 0 and len(self.data_dic["images"]) < 16:
            data_idx = len(self.data_dic["images"])
            if pl_module.hparams.model_name != 'detr_module':
                idx = random.choice(range(len(batch[0])))
            else:
                idx = random.choice(range(len(batch)))
            choosed_pred = outputs["preds"][idx]

            if pl_module.hparams.model_name != 'detr_module':
                self.data_dic["images"].append(batch[0][idx].cpu().clone()[None])  # 1CHW, float32, [0, 1] or [-n, +m]
            else:
                self.data_dic["images"].append(batch[idx]["image"].cpu().clone()[None])  # 1CHW, float32, [0, 255]
            if choosed_pred['boxes'].shape[0] > 0:
                self.data_dic["bboxes"].append(choosed_pred["boxes"].cpu().clone())
                self.data_dic["cls"].append(choosed_pred["labels"].cpu().clone().int())
                self.data_dic["confs"].append(choosed_pred["scores"].cpu().clone())
                self.data_dic["batch_idx"].append(torch.full_like(choosed_pred["labels"], data_idx, device='cpu', dtype=torch.int))

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        save_pic_dir = self.save_pic_dir or Path(trainer.log_dir) / 'metric_pics' / f'epoch_{trainer.current_epoch}/'
        save_pic_dir = save_pic_dir / 'peek.png'
        if trainer.global_rank == 0:
            data_dic = {}
            for k, v in self.data_dic.items():
                if len(v) == 0:
                    data_dic[k] = torch.tensor([])
                elif len(v) == 1:
                    data_dic[k] = v[0]
                else:
                    data_dic[k] = torch.cat(v, dim=0)
            save_pic_dir.parent.mkdir(parents=True, exist_ok=True)
            plot_images(
                **data_dic,
                fname=str(save_pic_dir),
                names=self.class_names,
                save=True,
                conf_thres=self.conf
            )
        self.data_dic = None

    def on_test_epoch_start(self, *args, **kwargs):
        self.on_validation_epoch_start(*args, **kwargs)

    def on_test_epoch_end(self, *args, **kwargs):
        self.on_validation_epoch_end(*args, **kwargs)

    def on_test_batch_end(self, *args, **kwargs):
        self.on_validation_batch_end(*args, **kwargs)