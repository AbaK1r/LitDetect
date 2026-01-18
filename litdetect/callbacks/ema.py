__all__ = ['EMACallback']

import logging

import torch
from pytorch_lightning.callbacks import Callback
from timm.utils.model_ema import ModelEmaV2

logger = logging.getLogger(__name__)


class EMACallback(Callback):
    def __init__(self, decay=0.9999, use_ema_weights: bool = True, warmup_steps=0, update_interval=1):
        self.decay = decay
        self.ema = None
        self.use_ema_weights = use_ema_weights
        self.warmup_steps = warmup_steps
        self.update_interval = update_interval
        self.step_counter = 0
        self.is_ema_initialized = False  # 新增初始化状态标志

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.step_counter += 1

        # 检查是否需要跳过更新
        if self.step_counter < self.warmup_steps or self.step_counter % self.update_interval != 0:
            return

        # 延迟初始化EMA（首次更新时）
        if not self.is_ema_initialized:
            self._initialize_ema(trainer, pl_module)

        self.ema.update(pl_module)

    def _initialize_ema(self, trainer, pl_module):
        """在第一次更新时初始化EMA模型"""
        # 主进程初始化EMA
        if trainer.is_global_zero:
            self.ema = ModelEmaV2(pl_module, decay=self.decay, device=None)
            logger.info(f"EMA initialized at step {self.step_counter}")

        # 标记已初始化并同步状态
        self.is_ema_initialized = True
        self._sync_ema_state(trainer, pl_module)

    def _sync_ema_state(self, trainer, pl_module):
        """同步EMA状态到所有进程"""
        if trainer.world_size <= 1:
            return

        # 广播初始化状态
        initialized = torch.tensor(self.is_ema_initialized, device=pl_module.device)
        torch.distributed.broadcast(initialized, src=0)
        self.is_ema_initialized = initialized.item()

        # 如果未初始化则直接返回
        if not self.is_ema_initialized:
            return

        # 从主进程同步EMA模型
        if self.ema is None:
            self.ema = ModelEmaV2(pl_module, decay=self.decay, device=None)

        # 同步模型参数
        for param in self.ema.module.parameters():
            torch.distributed.broadcast(param.data, src=0)
        for buffer in self.ema.module.buffers():
            torch.distributed.broadcast(buffer.data, src=0)

    def on_validation_epoch_start(self, trainer, pl_module):
        if not self.is_ema_initialized or self.step_counter < self.warmup_steps:
            return

        self.store(pl_module.parameters())
        self.copy_to(self.ema.module.parameters(), pl_module.parameters())
        logger.info("Using EMA model for validation")

    def on_validation_end(self, trainer, pl_module):
        if not self.is_ema_initialized or self.step_counter < self.warmup_steps:
            return
        self.restore(pl_module.parameters())

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.is_global_zero and self.is_ema_initialized:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
            checkpoint['ema_decay'] = self.decay
            checkpoint['ema_initialized'] = True  # 保存初始化状态
        return checkpoint

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if checkpoint.get('ema_initialized', False):
            self.is_ema_initialized = True
            if self.ema is None:
                decay = checkpoint.get('ema_decay', self.decay)
                self.ema = ModelEmaV2(pl_module, decay=decay, device=None)
            self.ema.load_state_dict(checkpoint['ema_state_dict'])

            # 确保所有进程同步加载的EMA状态
            if trainer.world_size > 1:
                self._sync_ema_state(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        if self.use_ema_weights and self.is_ema_initialized:
            logger.info("Replacing model parameters with EMA version")
            self.copy_to(self.ema.module.parameters(), pl_module.parameters())

    def store(self, parameters):
        self.collected_params = [param.detach().clone() for param in parameters]

    def restore(self, parameters):
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    @staticmethod
    def copy_to(shadow_parameters, parameters):
        for s_param, param in zip(shadow_parameters, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)