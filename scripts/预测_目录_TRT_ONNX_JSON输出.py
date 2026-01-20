import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from litdetect.data.scale_tookits import letterbox_array, recover_original_coords

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s,%(msecs)03d][%(name)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()  # 输出到控制台
    ]
)

def parse_args():
    parser = argparse.ArgumentParser(description="arguments")

    parser.add_argument("-d", "--ckpt_dir", type=str, help="权重路径，必填，会根据文件后缀选择引擎，onnx结尾则用onnx推理，否则使用simple_trt_infer")
    parser.add_argument("-p", "--config_dir", type=str, help="配置文件路径", default='')
    parser.add_argument("-i", "--input_dir", type=str, help="输入图像所在路径")
    parser.add_argument("-o", "--output_dir", type=str, help="json文件保存路径", default='')
    parser.add_argument("-s", "--suffix", type=str, help="图像后缀", default='.png')
    parser.add_argument("-t", "--threh", type=float, help="预测置信度阈值", default=0.4)
    parser.add_argument("-e", "--encrypt", help="使用加密后的权重，仅simple_trt_infer支持", action='store_true')

    args = vars(parser.parse_args())
    args['config_dir'] = None if args['config_dir'] == '' else args['config_dir']
    args['output_dir'] = None if args['output_dir'] == '' else args['output_dir']
    
    return args

def main():
    parser_args = parse_args()
    assert Path(parser_args['ckpt_dir']).is_file(), f'{parser_args["ckpt_dir"]} is not a file or not exist.'
    if parser_args['config_dir'] is None:
        config_dir = Path(parser_args['ckpt_dir']).parent.parent / 'full_config.yaml'
    else:
        config_dir = Path(parser_args['config_dir'])

    output_pred_path = Path(parser_args['input_dir']).parent / f'onnx_pred' \
        if parser_args['output_dir'] == '' else Path(parser_args['output_dir'])

    val(config_dir, parser_args['input_dir'],
        parser_args['ckpt_dir'], output_pred_path,
        parser_args['suffix'], parser_args['threh'],
        parser_args['encrypt'])
    
    
def val(args_path, data_dir, model_path, output_pred_path: Path = None, suffix='', threh: float = 0.4, encrypt=False):
    if output_pred_path is not None:
        output_pred_path.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError('output_pred_path is None')

    args = OmegaConf.load(args_path)

    if str(model_path).endswith('.onnx'):
        from litdetect.onnx_inferer import get_inferer
    else:
        from litdetect.trt_inferer import get_inferer

    inferer = get_inferer(args.model._target_, model_path, encrypt=encrypt)

    pic_paths = list(Path(data_dir).rglob(f'*{suffix}'))
    for pic_path in tqdm(pic_paths):
        image = cv2.imread(str(pic_path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
        # BGR → RGB, HWC uint8
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        image, scale_params = letterbox_array(image, args.data.input_size_hw, 0)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)[None]
        preds = inferer.inference(image, conf_threshold=0.05)[0]
        
        if len(preds['scores']) > 0:
            preds = [{
                "box": recover_original_coords(preds['boxes'][i].tolist(), scale_params, 'xyxy'),
                "score": float(preds['scores'][i]), "class": args.data.class_name[int(preds['labels'][i])],
            } for i in range(len(preds['scores'])) if preds['scores'][i] > threh]
            save_json(pic_path, preds, scale_params['orig_size'], Path(data_dir), output_pred_path)


# 保存结果为JSON格式
def save_json(image_path, preds, orig_shape, input_dir: Path, output_dir: Path):
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
    main()
