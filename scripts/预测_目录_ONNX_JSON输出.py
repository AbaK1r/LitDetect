import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Union, Tuple

import numpy as np
import onnxruntime
import torch
import torchvision
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm

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

    parser.add_argument("-d", "--ckpt_dir", type=str, help="权重路径，必填")
    parser.add_argument("-p", "--config_dir", type=str, help="配置文件路径", default='')
    parser.add_argument("-i", "--input_dir", type=str, help="Input directory")  # 图片所在的路径
    parser.add_argument("-o", "--output_dir", type=str, help="Output directory", default='')  # 输出目录
    parser.add_argument("-s", "--suffix", type=str, help="Image suffix", default='.png')  # 图片后缀
    parser.add_argument("-t", "--threh", type=float, help="box conf threh", default=0.4)

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
        parser_args['suffix'], parser_args['threh'])


class DINO_OnnxInferer:
    def __init__(self, model_path, pixel_mean, pixel_std):
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
            # providers.insert(0, 'TensorrtExecutionProvider')
        self.model = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape
        self.output_shape = self.model.get_outputs()[0].shape
        self.pixel_mean = np.array(pixel_mean, dtype=np.float32)[:, None, None]
        self.pixel_std = np.array(pixel_std, dtype=np.float32)[:, None, None]

    def inference(self, ipt, conf_threshold=0.05) -> List[Dict[str, np.ndarray]]:
        ipt = (ipt - self.pixel_mean) / self.pixel_std
        ipt = np.ascontiguousarray(ipt)
        output = self.model.run(None, {self.input_name: ipt})[0].astype(np.float32)
        output = self.postprocess(output, conf_threshold)
        return output

    def postprocess(self, outputs: np.ndarray, conf_threshold=0.05):
        """

        Args:
            outputs: (B, N, xyxy+score+class)
            conf_threshold:

        Returns: List[Dict[str, np.ndarray]]

        """
        bs, n_box, _ = outputs.shape
        b_bboxes = outputs[..., :4].reshape(bs, -1, 4)
        b_scores = outputs[..., 4].reshape(bs, -1)
        b_classes = outputs[..., 5].reshape(bs, -1).astype(np.int64)
        H, W = self.input_shape[-2:]
        outputs = []
        for bboxes, scores, classes in zip(b_bboxes, b_scores, b_classes):
            inds = np.where(scores > conf_threshold)[0]
            bboxes, scores, classes = bboxes[inds], scores[inds], classes[inds]
            if len(bboxes.shape) == 1:
                bboxes, scores, classes = bboxes[None], scores[None], classes[None]
            bboxes[:, 0] *= W
            bboxes[:, 1] *= H
            bboxes[:, 2] *= W
            bboxes[:, 3] *= H
            outputs.append({
                'boxes': bboxes,  # (N, 4) xyxy
                'scores': scores,  # (N, 1)
                'labels': classes,  # (N, 1)
            })

        return outputs


class FasterRCNN_OnnxInferer:
    def __init__(self, model_path, pixel_mean, pixel_std):
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
            # providers.insert(0, 'TensorrtExecutionProvider')
        self.model = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape
        self.output_shape = self.model.get_outputs()[0].shape
        self.pixel_mean = np.array(pixel_mean, dtype=np.float32)[:, None, None]
        self.pixel_std = np.array(pixel_std, dtype=np.float32)[:, None, None]

    def inference(self, ipt, conf_threshold=0.05, nms_threshold=0.45) -> List[Dict[str, np.ndarray]]:
        ipt = (ipt - self.pixel_mean) / self.pixel_std
        ipt = np.ascontiguousarray(ipt)
        output = self.model.run(None, {self.input_name: ipt})[0].astype(np.float32)
        output = self.postprocess(output, conf_threshold, nms_threshold)
        return output

    def postprocess(self, outputs: np.ndarray, conf_threshold=0.05, nms_threshold=0.45):
        """

        Args:
            outputs: (B, N, C, 5) xyxy, conf
            conf_threshold:
            nms_threshold:

        Returns: List[Dict[str, np.ndarray]]

        """
        bs, n_box, n_class, _ = outputs.shape
        b_bboxes = outputs[..., :4].reshape(bs, -1, 4)
        b_scores = outputs[..., 4].reshape(bs, -1)
        b_classes = np.tile(np.arange(n_class), (bs, n_box))

        outputs = []

        for bboxes, scores, classes in zip(b_bboxes, b_scores, b_classes):
            inds = np.where(scores > conf_threshold)[0]
            bboxes, scores, classes = bboxes[inds], scores[inds], classes[inds]

            c = classes * max(self.input_shape[-2:])
            nms_idx = torchvision.ops.nms(torch.tensor(bboxes + c[:, None]).float(), torch.tensor(scores).float(), nms_threshold)
            bboxes, scores, classes = bboxes[nms_idx], scores[nms_idx], classes[nms_idx]

            if len(bboxes.shape) == 1:
                bboxes, scores, classes = bboxes[None], scores[None], classes[None]
            outputs.append({
                'boxes': bboxes,  # (N, 4) xyxy
                'scores': scores,  # (N, 1)
                'labels': classes,  # (N, 1)
            })

        return outputs


class Yolo11_OnnxInferer:
    def __init__(self, model_path, pixel_mean, pixel_std):
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        self.model = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape
        self.output_shape = self.model.get_outputs()[0].shape
        self.pixel_mean = np.array(pixel_mean, dtype=np.float32)[:, None, None]
        self.pixel_std = np.array(pixel_std, dtype=np.float32)[:, None, None]

    def inference(self, ipt, *args, **kwargs) -> List[Dict[str, np.ndarray]]:
        ipt = (ipt - self.pixel_mean) / self.pixel_std
        ipt = np.ascontiguousarray(ipt)
        output = self.model.run(None, {self.input_name: ipt})[0].astype(np.float32)
        output = self.postprocess(output, *args, **kwargs)
        return output

    def postprocess(self, outputs: np.ndarray, conf_threshold=0.05, nms_threshold=0.45, classes_filter=None, max_nms=10000, max_det=300):
        """

        Args:
            max_det:
            max_nms:
            classes_filter:
            outputs: (B, N, xyxy+scores)
            conf_threshold:
            nms_threshold:

        Returns: List[Dict[str, np.ndarray]]

        """
        bs, n_box, _ = outputs.shape
        xc = np.amax(outputs[..., 4:], axis=2) > conf_threshold  # candidates

        output = [{
            'boxes': np.zeros((0, 4)),  # (N, 4) xyxy
            'scores': np.zeros((0,)),  # (N,)
            'labels': np.zeros((0,)),  # (N,)
        }] * bs

        for xi, x in enumerate(outputs):
            filt = xc[xi]  # confidence
            x = x[filt]

            # If none remain process next image
            if not x.shape[0]:
                continue

            box = x[:, :4]
            cls = x[:, 4:]

            conf = np.max(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1, keepdims=True)
            x = np.concatenate((box, conf, j.astype(x.dtype)), 1)

            if classes_filter is not None:
                if isinstance(classes_filter, list):
                    for c in classes_filter:
                        filt = np.any(x[:, 5:6] == c, axis=1)
                        x = x[filt]
                elif isinstance(classes_filter, int):
                    filt = np.any(x[:, 5:6] == classes_filter, axis=1)
                    x = x[filt]
                else:
                    raise ValueError(f'classes_filter must be list or int, but got {classes_filter}')

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue

            if n > max_nms:  # excess boxes
                filt = np.argsort(-x[:, 4])[:max_nms]
                x = x[filt]

            c = x[:, 5:6] * max(self.input_shape[-2:])
            nms_idx = torchvision.ops.nms(torch.tensor(x[:, :4] + c).float(), torch.tensor(x[:, 4]).float(), nms_threshold)
            nms_idx = nms_idx[:max_det]

            x = x[nms_idx]
            if len(x.shape) == 1:
                x = x[None]

            output[xi] = {
                'boxes': x[:, :4],  # (N, 4) xyxy
                'scores': x[:, 4],  # (N, 1)
                'labels': np.round(x[:, 5]).astype(np.int64),  # (N, 1)
            }

        return output
    
    
def val(args_path, data_dir, onnx_path, output_pred_path: Path = None, suffix='', threh: float = 0.4):
    if output_pred_path is not None:
        output_pred_path.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError('output_pred_path is None')

    args = OmegaConf.load(args_path)

    if args.model._target_ == 'litdetect.model.faster_rcnn.ModuleWrapper':
        INFERER = FasterRCNN_OnnxInferer
    elif args.model._target_ == 'litdetect.model.yolo11.ModuleWrapper':
        INFERER = Yolo11_OnnxInferer
    elif args.model._target_ == 'litdetect.model.detr_module.ModuleWrapper':
        INFERER = DINO_OnnxInferer
    else:
        raise ValueError(f'model must be faster_rcnn or yolo11 or detr, but got {args.model._target_}')

    normalize_args = args.data.augmentation_val.transforms[-2]
    inferer = INFERER(onnx_path, pixel_mean=normalize_args.mean, pixel_std=normalize_args.std)

    pic_paths = list(Path(data_dir).rglob(f'*{suffix}'))
    for pic_path in tqdm(pic_paths):
        image = Image.open(pic_path).convert('RGB')
        image, scale_params = letterbox(image, args.data.input_size_hw[::-1], 0)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.
        preds = inferer.inference(image[None], conf_threshold=0.05)[0]
        
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


def letterbox(image, target_size, fill=128):
    """
    保持比例缩放图像并在两侧填充指定颜色
    :param image: PIL.Image 输入图像
    :param target_size: (width, height) 目标尺寸
    :param fill: 填充颜色值（RGB单通道）
    :return:
        new_image: 缩放并填充后的PIL.Image
        scale_params: 包含缩放信息的字典，必须包含:
            'ratio': 缩放比例 (float)
            'pad': (left, top) 填充偏移量 (Tuple[float, float])
            'orig_size': (orig_w, orig_h) 原始的图像尺寸 (Tuple[int, int])
            'target_size': (target_w, target_h) 缩放+填充后的图像尺寸 (Tuple[int, int])
    """
    target_w, target_h = target_size
    orig_w, orig_h = image.size

    # 计算缩放比例和新的尺寸
    ratio = min(target_w / orig_w, target_h / orig_h)
    new_w = int(orig_w * ratio)
    new_h = int(orig_h * ratio)

    # 缩放图像
    image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # 创建新图像并填充
    new_image = Image.new('RGB', (target_w, target_h), (fill, fill, fill))
    left_pad = (target_w - new_w) // 2
    top_pad = (target_h - new_h) // 2
    new_image.paste(image, (left_pad, top_pad))

    return (new_image,
            {
                'ratio': ratio,
                'pad': (left_pad, top_pad),
                'orig_size': (orig_w, orig_h),
                'target_size': (target_w, target_h)
            })


def recover_original_coords(
    bbox: Union[List[float], torch.Tensor, np.ndarray],
    scale_params: Dict[str, Union[float, Tuple[float, float], Tuple[int, int]]],
    box_format: str = 'xyxy'
) -> List[float]:
    """
    将边界框坐标从填充后的图像恢复到原始图像坐标

    Args:
        bbox: 边界框坐标，格式由 box_format 指定
        scale_params: 包含缩放信息的字典，必须包含:
            'ratio': 缩放比例 (float)
            'pad': (left, top) 填充偏移量 (Tuple[float, float])
            'orig_size': (orig_w, orig_h) 原始图像尺寸 (Tuple[int, int])
        box_format: 边界框格式，'xyxy' 或 'xywh'

    Returns:
        原始图像上的边界框坐标（与输入相同的格式）

    Raises:
        ValueError: 当 box_format 不是 'xyxy' 或 'xywh' 时
    """
    # 参数验证
    if box_format not in ('xyxy', 'xywh'):
        raise ValueError(f"Invalid box_format: '{box_format}'. Must be 'xyxy' or 'xywh'")

    # 解包参数并类型转换
    ratio = float(scale_params['ratio'])
    pad_left, pad_top = map(float, scale_params['pad'])
    orig_w, orig_h = map(int, scale_params['orig_size'])

    # 转换为numpy数组处理
    if torch.is_tensor(bbox):
        bbox = bbox.cpu().numpy()
    bbox = np.asarray(bbox, dtype=np.float32).tolist()

    if box_format == 'xyxy':
        x1, y1, x2, y2 = bbox
    else:  # xywh
        cx, cy, w, h = bbox
        x1, y1 = cx - w / 2, cy - h / 2
        x2, y2 = cx + w / 2, cy + h / 2

    # 去除填充并反缩放
    def _convert(_val: float, _pad: float, _limit: float) -> float:
        """去除填充后反缩放，并限制在 [0, limit] 范围内"""
        return max(0.0, min((_val - _pad) / ratio, _limit))

    x1_orig = _convert(x1, pad_left, orig_w)
    y1_orig = _convert(y1, pad_top, orig_h)
    x2_orig = _convert(x2, pad_left, orig_w)
    y2_orig = _convert(y2, pad_top, orig_h)

    # 返回与输入相同的格式
    if box_format == 'xyxy':
        return [x1_orig, y1_orig, x2_orig, y2_orig]
    else:
        return [
            (x1_orig + x2_orig) / 2,  # cx
            (y1_orig + y2_orig) / 2,  # cy
            x2_orig - x1_orig,        # w
            y2_orig - y1_orig         # h
        ]


# 程序入口
if __name__ == '__main__':
    main()
