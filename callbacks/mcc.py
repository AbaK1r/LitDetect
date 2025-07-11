import torch
from pytorch_lightning import Callback


class MemoryCleanupCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        # 重要：在所有rank上执行清理
        torch.cuda.empty_cache()

    def on_train_epoch_end(self, trainer, pl_module):
        torch.cuda.empty_cache()
