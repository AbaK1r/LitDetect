import importlib
import inspect
from typing import Any

import pytorch_lightning as pl
import torch
import torch.optim.lr_scheduler as lrs
from pytorch_lightning.utilities.types import STEP_OUTPUT


# from typing import Any

# from model.validater.detection import DetectionValidator


# from coco_eval import CocoEvaluator


class ModuleInterface(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = self.instancialize()
        self.detection_validater = None

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, batch_idx):
        log_dict = self.model.train_step(batch)
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch[0]))
        return log_dict

    def validation_step(self, batch, batch_idx):
        DEVICE = self.device
        preds = self.model.val_step(batch)
        return {
            'preds': [{
                'boxes': i[:, :4].to(DEVICE),
                'scores': i[:, 4].to(DEVICE),
                'labels': i[:, 5].to(DEVICE).int()
            } for i in preds],
            'targets': [{
                'boxes': t['boxes'].to(DEVICE),
                'labels': t['labels'].to(DEVICE).int(),
                'image_id': t['image_id'].to(DEVICE).int()
            } for t in batch[1]]
        }

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return self.validation_step(*args, **kwargs)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 1e-4
        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if p.requires_grad]},
            # {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            # {
            #     "params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
            #     "lr": self.hparams.lr_backbone,
            # },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=self.hparams.lr, weight_decay=weight_decay)
        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = lrs.StepLR(optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma)
                # scheduler = lrs.ReduceLROnPlateau(
                #     optimizer, mode='min', factor=0.33, patience=4, threshold=1e-6, min_lr=9e-7, cooldown=1)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    # 'monitor': 'loss_epoch',
                }
            }

    def instancialize(self, **other_args):
        """
        Instancialize a model using the corresponding parameters
        from self.hparams dictionary. You can also input any args
        to overwrite the corresponding value in self.kwargs.
        """
        camel_name = ''.join([i.capitalize() for i in self.hparams.model_name.split('_')])
        try:
            Model = getattr(importlib.import_module(
                '.' + self.hparams.model_name, package=__package__), camel_name)
        except:
            raise ValueError(
                f'Invalid Module File Name or Invalid Class Name {self.hparams.model_name}.{camel_name}!')
        class_args = inspect.getfullargspec(Model.__init__).args[1:]

        args = {arg: self.hparams[arg] for arg in class_args if arg in self.hparams.keys()}
        args.update(other_args)
        return Model(**args)
