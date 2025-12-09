import argparse
import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

from litdetect.data.scale_tookits import letterbox, recover_original_coords
from litdetect.scripts_init import get_logger, check_path, check_version

# 初始化日志记录器
logger = get_logger(__file__)

# 设置CUDA环境变量和精度
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser(description="arguments")

    parser.add_argument("-v", "--versions", type=int, help="An integer or a list of integers")
    parser.add_argument("-i", "--input_dir", type=str, help="Input directory")  # 图片所在的路径
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory", default='')  # 输出目录
    parser.add_argument("--suffix", type=str, help="Image suffix", default='png')  # 图片后缀
    parser.add_argument("--threh", type=float, help="box conf threh", default=0.4)

    args = vars(parser.parse_args())
    args['versions'] = check_version(args['versions'])
    # TODO: 如果不通过配置参数，则取消注释并使用下面的arg
    # args = {
    #     'versions': 0,
    #     'input_dir': '/data/16t/lrh/datasets/胎儿超声/小类/训练/脊柱/images/val2017',
    #     'output_dir': '',
    #     'suffix': 'png',
    # }
    return args


def main():
    parser_args = parse_args()

    data_dir = Path(parser_args['input_dir'])
    image_suffix = parser_args['suffix']
    threh = parser_args['threh']
    ddir = Path(f'lightning_logs/version_{parser_args['versions']}/ckpts/')
    args = OmegaConf.load(ddir.parent / 'full_config.yaml')
    args.data.val_batch_size = 1
    # args.pretrained = False

    # 输出目录设置
    output_dir = data_dir.parent / f'{args.model.model_name}_pred' if parser_args['output_dir'] == '' else Path(parser_args['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f'Output dir: {output_dir}')

    # 检查点加载
    ckpts = ddir.rglob('*.ckpt')
    ckpts = [(float(i.stem.split('=')[-1]), i) for i in ckpts if i.stem != 'last']
    ckpts.sort(key=lambda x: x[0])
    if len(ckpts) == 0:
        raise ValueError(f'No ckpt found in {ddir}')
    ckpt = ckpts[-1][1] if args.callbacks.call_back_mode == 'max' else ckpts[0][1]
    logger.info(f'Load ckpt: {ckpt}')
    # model = ModuleInterface.load_from_checkpoint(
    #     checkpoint_path=ckpt, map_location='cpu', strict=True, **args).model.eval().cuda()
    model = hydra.utils.instantiate(args.model).eval().cuda()
    sd = torch.load(ckpt, weights_only=False, map_location='cpu')['state_dict']
    model.load_state_dict(sd, strict=False)

    normalize_args = args.data.augmentation_val.transforms[-2]
    pixel_mean = np.array(normalize_args.mean, dtype=np.float32)[:, None, None]
    pixel_std = np.array(normalize_args.std, dtype=np.float32)[:, None, None]

    # 推理过程
    with torch.no_grad():
        pic_paths = list(data_dir.rglob(f'*.{image_suffix}'))
        for pic_path in tqdm(pic_paths):
            pic, scale_params = preprocess(pic_path, args.data.input_size_hw[::-1], pixel_mean, pixel_std)
            batch = [{'image': pic}]
            preds = model.model.val_step(batch if not hasattr(model, 'input_batch_trans') else model.input_batch_trans(batch))[0]
            if preds.shape[0] > 0:
                preds = preds.cpu().tolist()
                preds = [{
                    "box": recover_original_coords(i[:4], scale_params, 'xyxy'),
                    "score": i[4], "class": args.data.class_name[int(i[5])],
                } for i in preds if i[4] >= threh]
                save_json(pic_path, preds, scale_params['orig_size'], data_dir, output_dir)

# 预处理函数，将图像转换为模型输入格式
def preprocess(pic_path, input_size, pixel_mean, pixel_std):
    """
    图像预处理
    :param pic_path: 图像路径
    :param input_size: 输入尺寸
    :param pixel_mean: 像素均值
    :param pixel_std: 像素标准差

    :return: 预处理后的图像和缩放参数
    """
    image = Image.open(pic_path).convert('RGB')
    image, scale_params = letterbox(image, input_size, 0)
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))

    image = (image - pixel_mean) / pixel_std

    image = torch.from_numpy(image).cuda()
    return image, scale_params

# 保存结果为JSON格式
def save_json(image_path, preds, orig_shape, input_dir, output_dir):
    """
    保存预测结果为JSON文件
    :param image_path: 图像路径
    :param preds: 预测结果
    :param orig_shape: 原始图像尺寸
    :param input_dir: 输入目录
    :param output_dir: 输出目录
    """
    json_data = {
        "version": "0.1.0",
        "flags": {},
        "shapes": [],
        "imagePath": image_path.name,
        "imageData": None,
        "imageHeight": orig_shape[1],
        "imageWidth": orig_shape[0],
        "description": ""
    }

    for pred in preds:
        x1, y1, x2, y2 = pred['box']
        points = [
            [x1, y1],
            [x2, y1],
            [x2, y2],
            [x1, y2]
        ]

        shape_info = {
            "label": pred['class'],
            "score": float(pred['score']),
            "points": points,
            "group_id": None,
            "description": "",
            "difficult": False,
            "shape_type": "rectangle",
            "flags": {},
            "attributes": {},
            "kie_linking": []
        }
        json_data["shapes"].append(shape_info)

    json_save_path = output_dir / image_path.relative_to(input_dir).with_suffix(".json")
    json_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_save_path, 'w', encoding='utf-8') as f:
        # noinspection PyTypeChecker
        json.dump(json_data, f, indent=4, ensure_ascii=False)

# 程序入口
if __name__ == '__main__':
    check_path(__file__)
    main()
