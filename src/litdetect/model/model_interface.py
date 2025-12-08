import inspect
import traceback
from abc import abstractmethod, ABC
from typing import Any, Type, Dict, List

import pytorch_lightning as pl
import torch

from litdetect.scripts_init import get_logger

logger = get_logger(__file__)


class ModuleInterface(pl.LightningModule, ABC):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['model_class'])
        self.model = self._instantiate_model()
        if self.hparams.get('compile', False):
            try:
                self.model = torch.compile(self.model)
                logger.info(f"Using torch.compile for {self.model.__class__.__name__}")
            except Exception as e:
                logger.error(f"Failed to compile {self.model.__class__.__name__}:\n\t{e}")
                traceback.print_exc()

    @property
    @abstractmethod
    def model_class(self) -> Type[torch.nn.Module]:
        """子类必须指定使用的模型类"""
        raise NotImplementedError

    def forward(self, *args, **kwargs):
        """统一的前向接口"""
        if hasattr(self.model, 'forward'):
            return self.model(*args, **kwargs)
        else:
            raise NotImplementedError("Model must have forward method")

    def training_step(self, batch, batch_idx):
        log_dict = self.model.train_step(batch if not hasattr(self, 'input_batch_trans') else self.input_batch_trans(batch))
        log_dict = {(k.replace('loss_', 'loss/') if k.startswith('loss_') else k): v
                    for k, v in log_dict.items()}
        self.log_dict(log_dict, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=len(batch))
        return log_dict

    def validation_step(self, batch, batch_idx) -> Dict[str, List[Dict[str, torch.Tensor]]]:
        DEVICE = self.device
        preds = self.model.val_step(batch if not hasattr(self, 'input_batch_trans') else self.input_batch_trans(batch))
        return {
            'preds': [{
                'boxes': i[:, :4].to(DEVICE),
                'scores': i[:, 4].to(DEVICE),
                'labels': i[:, 5].to(DEVICE).int()
            } for i in preds],
            'targets': [{
                'boxes': b['bboxes'].to(DEVICE),
                'labels': b['labels'].to(DEVICE),
                'image_id': b['image_id'].to(DEVICE)
            } for b in batch]
        }

    def test_step(self, *args: Any, **kwargs: Any):
        return self.validation_step(*args, **kwargs)

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 1e-4
        if self.hparams.optimizer == 'adam':
            param_dicts = [
                {"params": [p for n, p in self.named_parameters() if p.requires_grad]},
            ]
            optimizer = torch.optim.AdamW(
                param_dicts, lr=self.hparams.lr, weight_decay=weight_decay)
        elif self.hparams.optimizer == "muon":
            from torch import distributed as dist
            param_groups = [
                dict(params=self.model.hidden_weights, use_muon=True,
                     lr=self.hparams.muon_lr, weight_decay=self.hparams.muon_weight_decay),
                dict(params=self.model.hidden_gains_biases + self.model.nonhidden_params, use_muon=False,
                     lr=self.hparams.lr, betas=(0.9, 0.95), weight_decay=weight_decay),
            ]
            if dist.is_available() and dist.is_initialized():
                from muon import MuonWithAuxAdam
                optimizer = MuonWithAuxAdam(param_groups)
            else:
                from muon import SingleDeviceMuonWithAuxAdam
                optimizer = SingleDeviceMuonWithAuxAdam(param_groups)
        else:
            raise ValueError('Invalid optimizer')

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler}
            }

    def _instantiate_model(self):
        """自动提取模型参数并实例化"""
        model_class = self.model_class
        # 获取模型类的参数签名
        sig = inspect.signature(model_class.__init__)
        class_args = list(sig.parameters.keys())[1:]  # 排除self

        # 从hparams中提取匹配的参数
        model_kwargs = {arg: self.hparams[arg] for arg in class_args if arg in self.hparams.keys()}

        logger.info(f"Instantiating {model_class.__name__} with kwargs: {model_kwargs}")
        return model_class(**model_kwargs)
