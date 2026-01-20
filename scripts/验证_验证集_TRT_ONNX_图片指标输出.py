import argparse
import colorsys
import logging
import random
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
from torchmetrics.detection import MeanAveragePrecision
from tqdm import tqdm

from litdetect.data.scale_tookits import xywh2xyxy, letterbox_array, apply_scale_to_coords, recover_original_coords

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

    parser.add_argument("-v", "--versions", type=int, help="lightning_log中的version数字", default=None)
    parser.add_argument("-d", "--ckpt_dir", type=str, help="权重路径，必填，会根据文件后缀选择引擎，onnx结尾则用onnx推理，否则使用simple_trt_infer")
    parser.add_argument("-o", "--output_path", type=str, help="保存预测结果的路径，可不填", default='')
    parser.add_argument("-c", "--concat", help="拼接并保存GT和预测的图", action='store_true')
    parser.add_argument("-e", "--encrypt", help="使用加密后的权重，仅simple_trt_infer支持", action='store_true')
    parser.add_argument("-l", "--limit_num_pics", type=float, help="限制验证集图片数量", default=-1)

    args = vars(parser.parse_args())
    args['output_path'] = None if args['output_path'] == '' else args['output_path']

    from litdetect.scripts_init import check_version
    args['versions'] = check_version(args['versions'])

    # 如果不通过配置参数，则取消注释并使用下面的arg
    # args = {
    #     'versions': 0,
    #     'ckpt_dir': '/data/16t/wxh/LitDetect/lightning_logs/version_86/ckpts/yolo11-epoch=130-map_50=0.79230_bs_1.onnx'
    # }
    return args


def main():
    parser_args = parse_args()
    hparams_path = Path(f'lightning_logs/version_{parser_args['versions']}/full_config.yaml')
    if parser_args['output_path'] is None:
        output_pred_path = output_label_path = None
    else:
        output_pred_path = Path(parser_args['output_path']) / 'predictions'
        output_label_path = Path(parser_args['output_path']) / 'labels'

    val(hparams_path, parser_args['ckpt_dir'], output_pred_path,
        output_label_path, parser_args['encrypt'], parser_args['limit_num_pics'])
    if parser_args['concat']:
        concat_pred_gt_images(output_pred_path, output_label_path, Path(parser_args['output_path']) / 'concat_images')


def val(args_path, model_path, output_pred_path: Path = None,
        output_label_path: Path = None, encrypt=False, limit_num_pics=None):
    if output_pred_path is not None:
        output_pred_path.mkdir(parents=True, exist_ok=True)
    if output_label_path is not None:
        output_label_path.mkdir(parents=True, exist_ok=True)

    args = OmegaConf.load(args_path)

    ano_root = args.data.ano_root
    image_root = args.data.image_root
    item_list = []
    for ano_path in (Path(ano_root) / 'val2017').glob('*'):
        image_path = Path(image_root) / 'val2017' / (ano_path.stem + '.png')
        if not image_path.exists():
            continue
        _ano = np.loadtxt(ano_path, dtype=np.float32)

        if len(_ano) == 0:
            continue
        else:
            if len(_ano.shape) == 1:
                _ano = _ano[None]
            _ano = [_ano[:, 0].astype(int), _ano[:, 1:]]
        _item = {'image_path': image_path, 'annotation': _ano}
        item_list.append(_item)
    logger.info(f'find {len(item_list)} images')

    if limit_num_pics > 1:
        limit_num_pics = round(limit_num_pics)
    elif 0 < limit_num_pics < 1:
        limit_num_pics = round(len(item_list) * limit_num_pics)
    else:
        limit_num_pics = None

    item_list = item_list[:limit_num_pics] if limit_num_pics is not None else item_list
    logger.info(f'use {len(item_list)} images')

    map_metric = MeanAveragePrecision(
        box_format="xyxy",  # 边界框格式：左上右下坐标
        iou_type="bbox",  # 计算边界框IoU
        class_metrics=True,  # 计算每个类别的指标
        extended_summary=False,  # 启用详细数据（包含精确率/召回率）
        backend="faster_coco_eval"  # 使用 faster_coco_eval 后端
    )

    if str(model_path).endswith('.onnx'):
        from litdetect.onnx_inferer import get_inferer
    else:
        from litdetect.trt_inferer import get_inferer

    inferer = get_inferer(args.model._target_, model_path, encrypt=encrypt)

    for meta in tqdm(item_list):
        raw_image = cv2.imread(meta['image_path'], cv2.IMREAD_UNCHANGED | cv2.IMREAD_IGNORE_ORIENTATION)
        # BGR → RGB, HWC uint8
        if len(raw_image.shape) == 2:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_GRAY2RGB)
        elif raw_image.shape[2] == 3:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        elif raw_image.shape[2] == 4:
            raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGRA2RGB)

        labels, bboxes = meta['annotation']

        x_size, y_size = raw_image.shape[1], raw_image.shape[0]
        bboxes = [xywh2xyxy([i[0] * x_size, i[1] * y_size, i[2] * x_size, i[3] * y_size]) for i in bboxes]

        image, scale_params = letterbox_array(raw_image, args.data.input_size_hw, 0)
        bboxes = [apply_scale_to_coords(bbox, scale_params, 'xyxy') for bbox in bboxes]
        target = {
            'boxes': torch.tensor(bboxes).float(),
            'labels': torch.tensor(labels).int(),
        }

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)[None]
        preds = inferer.inference(image, conf_threshold=0.05)[0]
        # print(preds, target)
        if output_pred_path is not None:
            _boxes = np.array([recover_original_coords(i, scale_params, 'xyxy') for i in preds['boxes']])
            if len(_boxes) == 0:
                _boxes = np.zeros((0, 4), dtype=np.float32)
            plot_images(
                np.array(raw_image),
                preds['labels'],
                _boxes,
                preds['scores'],
                fname=str(output_pred_path / meta['image_path'].name),
                names=args.data.class_name,
                save=True,
                conf_thres=0.05,
            )

        if output_label_path is not None:
            _boxes = np.array([recover_original_coords(i, scale_params, 'xyxy') for i in target['boxes']])
            plot_images(
                np.array(raw_image),
                np.array(target['labels']),
                _boxes,
                fname=str(output_label_path / meta['image_path'].name),
                names=args.data.class_name,
                save=True,
            )

        preds = {k: torch.tensor(v) for k, v in preds.items()}

        map_metric.update([preds], [target])

    map_res = map_metric.compute()
    logger.info(map_res)
    # import pickle
    # with open(f'onnx.pkl', 'wb')as f:
    #     data = (map_metric.detection_labels,
    #             map_metric.detection_box,
    #             map_metric.detection_scores,
    #             map_metric.groundtruth_box)
    #     pickle.dump(data, f)


def plot_images(
    images: np.ndarray,
    cls: np.ndarray,
    boxes: np.ndarray = np.zeros((0, 4), dtype=np.float32),
    confs: Optional[np.ndarray] = None,
    fname: str = "images.jpg",
    names: Optional[Dict[int, str]] = None,
    save: bool = True,
    conf_thres: float = 0.25,
) -> Optional[np.ndarray]:
    if images.max() <= 1:
        images *= 255
    pil_img = Image.fromarray(images.astype(np.uint8))

    # 为每个类别生成固定颜色
    unique_classes = np.unique(cls)
    color_map = {}
    for c in unique_classes:
        # 使用HSV颜色空间生成可区分颜色
        hue = (int(c) * 0.618) % 1.0  # 黄金比例确保颜色分布均匀
        saturation = 0.7 + random.random() * 0.3
        value = 0.7 + random.random() * 0.3
        r, g, b = [int(255 * x) for x in colorsys.hsv_to_rgb(hue, saturation, value)]
        color_map[int(c)] = (r, g, b)

    draw = ImageDraw.Draw(pil_img)
    font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
    try:
        font_size = 20
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        logger.warning(f"Failed to load font {font_path}: {e}, using default font.")
        font = ImageFont.load_default()

    for i, box in enumerate(boxes):
        if confs is not None and confs[i] < conf_thres:
            continue

        c = int(cls[i])
        class_name = names.get(c, str(c)) if names else str(c)

        # 添加置信度（如果有）
        label = f"{class_name} {confs[i]:.2f}" if confs is not None else class_name

        color = color_map.get(c, (0, 0, 255))  # 使用固定颜色

        x1, y1, x2, y2 = map(int, box[:4])

        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        # 计算文本大小
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        text_y = y1 - text_h - 5

        # 绘制文本背景
        draw.rectangle(
            [x1, text_y, x1 + text_w, text_y + text_h + 5],
            fill=color
        )

        # 绘制文本
        draw.text(
            (x1, text_y),
            label,
            fill=(255, 255, 255),
            font=font
        )

    # 保存或返回结果
    if not save:
        return np.array(pil_img)

    pil_img.save(fname)
    return None


def concat_pred_gt_images(output_pred_path: Path, output_label_path: Path, final_path: Path):
    final_path.mkdir(parents=True, exist_ok=True)
    pred_images_path = list(output_pred_path.glob('*.png'))

    for pred_image_path in tqdm(pred_images_path):
        gt_image_path = output_label_path / pred_image_path.name
        assert gt_image_path.exists()
        concat_image = np.concatenate([
            np.array(Image.open(pred_image_path)),
            np.array(Image.open(gt_image_path))
        ], axis=1)
        Image.fromarray(concat_image).save(final_path / pred_image_path.name)


if __name__ == '__main__':
    main()
