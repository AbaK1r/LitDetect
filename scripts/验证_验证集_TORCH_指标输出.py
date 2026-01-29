import argparse
import json
import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from litdetect.callbacks import DetectionMetricsCallback, PicRecordCallback
from litdetect.clearn_logs import remove_incomplete_versions
from litdetect.data import DataInterface
from litdetect.scripts_init import get_logger, check_path, check_version

# 初始化日志记录器
logger = get_logger(__file__)

# 设置浮点数矩阵乘法的精度为高，以优化性能
torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser(description="arguments")

    parser.add_argument(
        "-v", "--versions",
        type=int,
        nargs='+',  # '+' 表示至少一个或多个值
        help="An integer or a list of integers",
        default=None,
    )

    args = parser.parse_args()

    # 将结果转换为 list[int]
    versions = args.versions
    if versions is None:
        check_version(None)
    if isinstance(versions, int):
        versions = [versions]
    for i in range(len(versions)):
        versions[i] = check_version(versions[i])
    # TODO: 如果不通过配置参数，则取消注释并使用下面的版本号
    # versions = [0]
    return versions


# 函数用于打印配置参数
def args_print(cfg: DictConfig):
    """
    打印配置参数
    :param cfg: 配置参数，以DictConfig形式传递
    """
    # 将配置参数转换为字典形式，并打印
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(cfg_dict, indent=4, ensure_ascii=False))


# 主函数，用于模型验证
def main():
    _versions = parse_args()
    tensorboard_logger = TensorBoardLogger('')
    save_xlsx_dir = Path(tensorboard_logger.log_dir) / 'result.xlsx'
    (Path(tensorboard_logger.log_dir) / 'ckpts').mkdir(parents=True, exist_ok=True)
    for _version in _versions:
        ddir = Path(f'lightning_logs/version_{_version}/ckpts/')
        # 加载配置参数
        args = OmegaConf.load(ddir.parent / 'full_config.yaml')
        args.data.val_batch_size = 1
        # args.pretrained = False

        # 加载检查点文件
        ckpts = ddir.rglob('*.ckpt')
        ckpts = [(float(i.stem.split('=')[-1]), i) for i in ckpts if i.stem != 'last']
        ckpts.sort(key=lambda x: x[0])
        if len(ckpts) == 0:
            logger.error(f'No ckpt found in {ddir}')
            continue
        # 根据回调模式选择最佳检查点
        ckpt = ckpts[-1][1] if args.callbacks.call_back_mode == 'max' else ckpts[0][1]
        logger.info(f'Load ckpt: {ckpt}')

        # 加载模型和数据集
        # model = ModuleInterface.load_from_checkpoint(
        #     checkpoint_path=ckpt, map_location='cpu', strict=True, **args)
        model = hydra.utils.instantiate(args.model).eval()
        sd = torch.load(ckpt, weights_only=False, map_location='cpu')['state_dict']
        model.load_state_dict(sd, strict=False)

        dl = DataInterface(**args.data)

        # 初始化回调列表
        save_sheet_name = f"{args.model.model_name}_{args.data.dataset_name}_v{_version}"
        call_back = [
            DetectionMetricsCallback(
                args.data.class_name,
                iou_threshold=0.5,
                save_xlsx_dir=save_xlsx_dir,
                save_sheet_name=save_sheet_name
            ),
            PicRecordCallback(args.data.class_name, 0.25, save_xlsx_dir.parent / save_sheet_name),
        ]
        if tensorboard_logger is None:
            tensorboard_logger = TensorBoardLogger('')
        # 初始化训练器
        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            enable_progress_bar=True,
            precision='32-true',
            callbacks=call_back,
            logger=tensorboard_logger,
        )
        # 执行模型验证
        trainer.test(model, datamodule=dl)
        torch.cuda.empty_cache()

        tensorboard_events = list(Path(tensorboard_logger.log_dir).glob('events*'))
        for tensorboard_event in tensorboard_events:
            os.system(f'rm {tensorboard_event}')
        tensorboard_logger = None


if __name__ == '__main__':
    check_path(__file__)
    main()
    remove_incomplete_versions('lightning_logs')
