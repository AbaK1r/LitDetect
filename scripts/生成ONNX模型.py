import argparse
from pathlib import Path

import hydra
import onnx
import onnxsim
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from litdetect.scripts_init import get_logger, check_path, check_version

# 初始化日志记录器
logger = get_logger(__file__)


def parse_args():
    parser = argparse.ArgumentParser(description="convert moduel to ONNX")

    parser.add_argument(
        "-v", "--version",
        type=int,
        help="version number"
    )

    parser.add_argument(
        "-b", "--batchsize",
        type=int,
        help="batch size, -1 means dynamic batch size",
        default=1
    )

    parser.add_argument(
        "-c", "--not_use_cuda",
        action="store_false",
        help="use cuda or not"
    )

    args = parser.parse_args()

    # 将结果转换为 int
    version = check_version(args.version)
    if version is None:
        raise ValueError("Please provide a valid version number.")

    batchsize = args.batchsize
    if batchsize == -1:
        dynamic = True
        batchsize = 1
    elif batchsize > 0:
        dynamic = False
    else:
        raise ValueError("Please provide a valid batch size.")

    return version, dynamic, batchsize, args.not_use_cuda


@torch.no_grad()
def export_onnx(ckpt_path, torch_model, input_shape, dynamic=False, use_cuda=False):
    ckpt_path = Path(ckpt_path)
    export_onnx_path = str(
        ckpt_path.parent / ('.'.join(ckpt_path.name.split('.')[:-1]) + ('_dynamic.onnx' if dynamic else f'_bs_{input_shape[0]}.onnx'))
    )

    dummy_input = torch.randn(input_shape)
    if use_cuda:
        dummy_input = dummy_input.cuda()
    output = torch_model(dummy_input)
    logger.info(f"Model output shape: {output.shape}")

    dynamic_axes = {
        'input': {0: 'batch_size'},   # 动态 batch size 输入
        'output': {0: 'batch_size'}   # 动态 batch size 输出
    } if dynamic else None

    torch.onnx.export(
        torch_model,
        dummy_input,
        export_onnx_path,
        export_params=True,
        dynamic_axes=dynamic_axes,
        opset_version=16,
        input_names=['input'],
        output_names=['output']
    )

    simplified_model, _ = onnxsim.simplify(export_onnx_path)
    onnx.save(simplified_model, export_onnx_path)

    logger.info(f"simplified model has been successfully converted to ONNX and saved at {export_onnx_path}")


def main():
    version, dynamic, batchsize, use_cuda = parse_args()
    logger.info(f"use_cuda: {use_cuda}")
    ddir = Path(f'lightning_logs/version_{version}/ckpts/')
    # 加载配置参数
    args = OmegaConf.load(ddir.parent / 'full_config.yaml')
    # args.pretrained = False

    # 加载检查点文件
    ckpts = ddir.rglob('*.ckpt')
    ckpts = [(float(i.stem.split('=')[-1]), i) for i in ckpts if i.stem != 'last']
    ckpts.sort(key=lambda x: x[0])
    if len(ckpts) == 0:
        raise ValueError(f'No ckpt found in {ddir}')
    # 根据回调模式选择最佳检查点
    ckpt = ckpts[-1][1] if args.callbacks.call_back_mode == 'max' else ckpts[0][1]
    logger.info(f'Load ckpt: {ckpt}')

    # 加载模型和数据集
    normalize_args = args.data.augmentation_val.transforms[-2]
    model: pl.LightningModule = hydra.utils.instantiate(
        args.model, pixel_mean=normalize_args.mean, pixel_std=normalize_args.std).eval()
    sd = torch.load(ckpt, weights_only=False, map_location='cpu')['state_dict']
    model.load_state_dict(sd, strict=False)
    # model = ModuleInterface.load_from_checkpoint(checkpoint_path=ckpt, map_location='cpu', **args).eval()
    if use_cuda:
        model.cuda()
    export_onnx(ckpt, model, input_shape=(batchsize, 3, *args.data.input_size_hw), dynamic=dynamic, use_cuda=use_cuda)


if __name__ == '__main__':
    check_path(__file__)
    main()
