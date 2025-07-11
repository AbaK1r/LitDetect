import argparse
import logging
from pathlib import Path

import onnx
import onnxsim
import torch
from omegaconf import OmegaConf

from litdetect.model import ModuleInterface
from litdetect.scripts_init import get_logger, check_path

check_path(__file__)
# 初始化日志记录器
logger = get_logger(__file__)


def parse_args():
    parser = argparse.ArgumentParser(description="convert Faster R-CNN to ONNX")

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

    args = parser.parse_args()

    # 将结果转换为 int
    version = args.version
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

    return version, dynamic, batchsize


@torch.no_grad()
def export_onnx(ckpt_path, torch_model, input_shape, dynamic=False):
    ckpt_path = Path(ckpt_path)
    export_onnx_path = str(
        ckpt_path.parent / ('.'.join(ckpt_path.name.split('.')[:-1]) + ('_dynamic.onnx' if dynamic else f'_bs_{input_shape[0]}.onnx'))
    )

    dummy_input = torch.randn(input_shape).to('cuda')
    output = torch_model(dummy_input)
    print(f"Model output shape: {output.shape}")

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
        opset_version=20,
        input_names=['input'],
        output_names=['output']
    )

    simplified_model, _ = onnxsim.simplify(export_onnx_path)
    onnx.save(simplified_model, export_onnx_path)

    logger.info(f"simplified model has been successfully converted to ONNX and saved at {export_onnx_path}")


def main():
    version, dynamic, batchsize = parse_args()
    ddir = Path(f'lightning_logs/version_{version}/ckpts/')
    # 加载配置参数
    args = OmegaConf.load(ddir.parent / 'hparams.yaml')
    args.pretrained = False

    # 加载检查点文件
    ckpts = ddir.rglob('*.ckpt')
    ckpts = [(float(i.stem.split('=')[-1]), i) for i in ckpts if i.stem != 'last']
    ckpts.sort(key=lambda x: x[0])

    # 根据回调模式选择最佳检查点
    ckpt = ckpts[-1][1] if args.call_back_mode == 'max' else ckpts[0][1]
    logger.info(f'Load ckpt: {ckpt}')

    # 加载模型和数据集
    model = ModuleInterface.load_from_checkpoint(checkpoint_path=ckpt, **args).eval().cuda()
    export_onnx(ckpt, model, input_shape=(batchsize, 3, *args.input_size[::-1]), dynamic=dynamic)

if __name__ == '__main__':
    main()
