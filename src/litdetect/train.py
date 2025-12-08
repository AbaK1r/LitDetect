from pathlib import Path
from typing import List

import hydra
import pytorch_lightning as pl
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from litdetect.callbacks import DetectionMetricsCallback, MemoryCleanupCallback, PicRecordCallback
from litdetect.clearn_logs import remove_incomplete_versions
from litdetect.data.data_interface import DataInterface
from litdetect.scripts_init import get_logger, copy_code, copy_config, check_path

check_path(__file__)
logger = get_logger(__file__)
remove_incomplete_versions('lightning_logs')
torch.set_float32_matmul_precision('high')


@hydra.main(config_path=str(Path.cwd()/"conf"), config_name="config", version_base=None)
def main(cfg: DictConfig):
    tensorboard_logger = TensorBoardLogger('')
    copy_code(tensorboard_logger)
    copy_config(Path(tensorboard_logger.log_dir), Path(HydraConfig.get().runtime.output_dir))

    if cfg.get("seed"):
        logger.info(f"random seed: {cfg.seed}")
        pl.seed_everything(cfg.seed, workers=True)

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: pl.LightningModule = hydra.utils.instantiate(cfg.model)

    continue_train_ckpt_path: str = cfg.csdp
    load_state_dict_ckpt_path: str = cfg.sdp
    if continue_train_ckpt_path is None and load_state_dict_ckpt_path is not None:
        sd = torch.load(load_state_dict_ckpt_path, weights_only=False)['state_dict']
        model.load_state_dict(sd, strict=False)
        logger.info(f"load state dict from {load_state_dict_ckpt_path}")

    logger.info(f"Instantiating datamodule <{cfg.data.dataset}>")
    dl: pl.LightningDataModule = DataInterface(**cfg.data)

    callbacks: List[pl.Callback] = [
        PicRecordCallback(cfg.callbacks.class_name, 0.25),
        DetectionMetricsCallback(cfg.callbacks.class_name),
        ModelCheckpoint(
            dirpath=Path(tensorboard_logger.log_dir) / 'ckpts',
            monitor=cfg.callbacks.call_back_monitor,
            filename=f'{cfg.callbacks.model_name}-{"{"}epoch:03d{"}"}'
                     f'-{"{"}{cfg.callbacks.call_back_monitor}:.5f{"}"}',
            save_top_k=2,
            save_weights_only=True,
            mode=cfg.callbacks.call_back_mode,
            save_last=True,
            save_on_train_epoch_end=False,
        ),
        LearningRateMonitor(logging_interval="step"),
        MemoryCleanupCallback(),
    ]
    if cfg.callbacks.early_stop:
        callbacks.append(EarlyStopping(
            monitor=cfg.callbacks.call_back_monitor,
            min_delta=0.00, patience=50,
            verbose=True, mode=cfg.callbacks.call_back_mode,
            check_finite=True, check_on_train_epoch_end=False
        ))
        logger.info(f"early stop: {cfg.callbacks.early_stop}")

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=tensorboard_logger
    )
    trainer.fit(model, datamodule=dl, ckpt_path=continue_train_ckpt_path)


if __name__ == "__main__":
    main()
