import logging
import shutil
import warnings
from pathlib import Path
warnings.filterwarnings("ignore", message="functools.partial will be a method descriptor in future Python versions")
warnings.filterwarnings("ignore", message="No device id is provided via `init_process_group` or `barrier `")
warnings.filterwarnings("ignore", message="Grad strides do not match bucket view strides.")

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only

from litdetect.clearn_logs import remove_incomplete_versions
from litdetect.data import DataInterface
from litdetect.model import ModuleInterface
from litdetect.callbacks import DetectionMetricsCallback, MemoryCleanupCallback, PicRecordCallback
from litdetect.scripts_init import get_logger, check_path

check_path(__file__)
# 初始化日志记录器
logger = get_logger(__file__)

remove_incomplete_versions('lightning_logs')

# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
torch.set_float32_matmul_precision('high')

@hydra.main(config_path=str(Path.cwd()/"conf"), config_name="config", version_base=None)
def main(cfg: DictConfig):
    logger.info(f"random seed: {cfg.run.seed}")
    pl.seed_everything(cfg.run.seed, workers=True)
    tensorboard_logger = TensorBoardLogger('')
    copy_code(tensorboard_logger)

    args = cfg.run
    if args.class_name is None:
        args.class_name = [f'{args.dataset_name}_{i}' for i in range(args.num_classes)]
    assert len(args.class_name) == args.num_classes, f"class_name数量({len(args.class_name)})与num_classes({args.num_classes})不一致"

    model = ModuleInterface(**args)
    continue_train_ckpt_path = args.csdp
    load_state_dict_ckpt_path = args.sdp
    if continue_train_ckpt_path is None and load_state_dict_ckpt_path is not None:
        sd = torch.load(load_state_dict_ckpt_path, weights_only=False)['state_dict']
        model.load_state_dict(sd, strict=False)
        logger.info(f"load state dict from {load_state_dict_ckpt_path}")

    dl = DataInterface(**args)

    callbacks = [
        PicRecordCallback(args.class_name, 0.25),
        DetectionMetricsCallback(args.class_name),
        ModelCheckpoint(
            dirpath=Path(tensorboard_logger.log_dir) / 'ckpts',
            monitor=args.call_back_monitor,
            filename=f'{args.model_name}-{"{"}epoch:03d{"}"}-{"{"}{args.call_back_monitor}:.5f{"}"}',
            save_top_k=2,
            save_weights_only=True,
            mode=args.call_back_mode,
            save_last=True,
            save_on_train_epoch_end=False,
        ),
        LearningRateMonitor(logging_interval="step"),
        MemoryCleanupCallback(),
    ]
    if args.early_stop:
        callbacks.append(EarlyStopping(
            monitor=args.call_back_monitor,
            min_delta=0.00, patience=50,
            verbose=True, mode=args.call_back_mode,
            check_finite=True, check_on_train_epoch_end=False
        ))
        logger.info(f"early stop: {args.early_stop}")
    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=callbacks,
        logger=tensorboard_logger
    )

    trainer.fit(model, datamodule=dl, ckpt_path=continue_train_ckpt_path)


@rank_zero_only
def copy_code(tensorboard_logger):
    """
    Copy code to tensorboard_logger.log_dir
    :param tensorboard_logger:
        TensorBoardLogger
    :return: None
    """
    src_dirs = [Path('src'), Path('conf')]
    dst_dir = Path(tensorboard_logger.log_dir) / 'code'
    dst_dir.mkdir(parents=True, exist_ok=True)
    copy_code_to_log_dir(src_dirs, dst_dir)

def ignore_pycache(dir, files):
    """
    Filters out the '__pycache__' directory and its contents.

    :param dir: Directory path (not used).
    :param files: List of files/directories in the given directory.
    :return: List of files/directories to ignore.
    """
    return [f for f in files if f.startswith("__pycache__")]


def copy_code_to_log_dir(src_dirs, dst_dir):
    """
    Copies the content of source directories to destination directory.

    :param src_dirs: List of source directories to copy from.
    :param dst_dir: Destination directory to copy to.
    """
    for src_dir in src_dirs:
        if src_dir.exists() and src_dir.is_dir():
            logging.info(f'copy {src_dir} to {dst_dir / src_dir.name}')
            shutil.copytree(src_dir, dst_dir / src_dir.name, dirs_exist_ok=True, ignore=ignore_pycache)
        else:
            logging.info(f'缺失文件夹{src_dir.name}，已跳过！')

if __name__ == '__main__':
    main()
