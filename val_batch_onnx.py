import colorsys
import copy
import random
from typing import List, Union, Dict, Tuple, Optional
import logging
from pathlib import Path
from tqdm import tqdm

import numpy as np
import onnxruntime
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
from torchmetrics.detection import MeanAveragePrecision

logger = logging.getLogger(__name__)


class FasterRCNN_OnnxInferer:
    def __init__(self, model_path):
        providers = ['CPUExecutionProvider']
        if torch.cuda.is_available():
            providers.insert(0, 'CUDAExecutionProvider')
        self.model = onnxruntime.InferenceSession(model_path, providers=providers)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.input_shape = self.model.get_inputs()[0].shape
        self.output_shape = self.model.get_outputs()[0].shape

    def inference(self, ipt, conf_threshold=0.05, nms_threshold=0.45) -> List[Dict[str, np.ndarray]]:
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

            c = classes * max(self.input_shape[1:])
            nms_idx = torchvision.ops.nms(torch.tensor(bboxes + c[:, None]).float(), torch.tensor(scores).float(), nms_threshold)
            bboxes, scores, classes = bboxes[nms_idx], scores[nms_idx], classes[nms_idx]

            if len(bboxes.shape) == 1:
                bboxes, scores, classes = bboxes[None], scores[None], classes[None]
            outputs.append({
                'boxes': bboxes,  # (N, 4) xyxy
                'scores': scores,  # (N,)
                'labels': classes,  # (N,)
            })

        return outputs

# inferer = FasterRCNN_OnnxInferer('/data/16t/wxh/LitDetect/lightning_logs/version_79/ckpts/faster_rcnn-epoch=167-map_50=0.82812.onnx')
# from PIL import Image
# img = np.array(Image.open('/data/4t_hdd/DataSets/xjlDataset/Luna16results2/yolo_dataset/images/val2017/1.3.6.1.4.1.14519.5.2.1.6279.6001.196251645377731223510086726530_slice_373.png').convert('RGB'))
# img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.
# preds = inferer.inference(img, conf_threshold=0.01)
# print(preds)

def val(output_pred_path: Path = None, output_label_path: Path = None):
    if output_pred_path is not None:
        output_pred_path.mkdir(parents=True, exist_ok=True)
    if output_label_path is not None:
        output_label_path.mkdir(parents=True, exist_ok=True)

    args = OmegaConf.load('lightning_logs/version_84/hparams.yaml')

    ano_root = args.ano_root
    image_root = args.image_root
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

    map_metric = MeanAveragePrecision(
        box_format="xyxy",  # 边界框格式：左上右下坐标
        iou_type="bbox",  # 计算边界框IoU
        class_metrics=True,  # 计算每个类别的指标
        extended_summary=False,  # 启用详细数据（包含精确率/召回率）
        backend="pycocotools"  # 使用pycocotools后端
    )
    inferer = FasterRCNN_OnnxInferer(
        '/data/16t/wxh/LitDetect/lightning_logs/version_84/ckpts/faster_rcnn-epoch=115-map_50=0.78982.onnx')

    for meta in tqdm(item_list):
        raw_image = Image.open(meta['image_path']).convert('RGB')
        labels, bboxes = meta['annotation']

        x_size, y_size = raw_image.size
        bboxes = [xywh2xyxy([i[0] * x_size, i[1] * y_size, i[2] * x_size, i[3] * y_size]) for i in bboxes]

        image, scale_params = letterbox(raw_image, args.input_size, 0)
        bboxes = [apply_scale_to_coords(bbox, scale_params, 'xyxy') for bbox in bboxes]
        target = {
            'boxes': torch.tensor(bboxes).float(),
            'labels': torch.tensor(labels).int(),
        }
        image = np.transpose(image, (2, 0, 1)).astype(np.float32) / 255.
        preds = inferer.inference(image[None])[0]

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
                names=args.class_name,
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
                names=args.class_name,
                save=True,
            )

        preds = {k: torch.tensor(v) for k, v in preds.items()}

        map_metric.update([preds], [target])

    map_res = map_metric.compute()
    print(map_res)


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
    try:
        font_path = "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"
        font_size = 20
        font = ImageFont.truetype(font_path, font_size)
    except Exception as e:
        print(f"[Warning] Failed to load font {font_path}: {e}, using default font.")
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


def xyxy2xywh(
    bbox: Union[List[float], tuple]
) -> List[float]:
    """
    将边界框坐标从 xyxy 格式 (x1, y1, x2, y2) 转换为 xywh 格式 (cx, cy, w, h)

    Args:
        bbox: 长度为4的列表或元组，表示 [x1, y1, x2, y2]

    Returns:
        转换后的 [cx, cy, w, h]
    """
    x1, y1, x2, y2 = map(float, bbox)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def xywh2xyxy(
    bbox: Union[List[float], tuple]
) -> List[float]:
    """
    将边界框坐标从 xywh 格式 (cx, cy, w, h) 转换为 xyxy 格式 (x1, y1, x2, y2)

    Args:
        bbox: 长度为4的列表或元组，表示 [cx, cy, w, h]

    Returns:
        转换后的 [x1, y1, x2, y2]
    """
    cx, cy, w, h = map(float, bbox)
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return [x1, y1, x2, y2]


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
    image = copy.deepcopy(image).resize((new_w, new_h), Image.Resampling.BILINEAR)

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


def apply_scale_to_coords(
    bbox: Union[List[float], torch.Tensor, np.ndarray],
    scale_params: Dict[str, Union[float, Tuple[float, float], Tuple[int, int]]],
    box_format: str = 'xyxy'
) -> List[float]:
    """
    将边界框坐标从原始图像映射到缩放+填充后的图像空间

    Args:
        bbox: 边界框坐标，格式由 box_format 指定
        scale_params: 包含缩放信息的字典，必须包含:
            'ratio': 缩放比例 (float)
            'pad': (left, top) 填充偏移量 (Tuple[float, float])
            'target_size': (target_w, target_h) 缩放+填充后的图像尺寸 (Tuple[int, int])
        box_format: 边界框格式，'xyxy' 或 'xywh'

    Returns:
        缩放+填充后图像上的边界框坐标（与输入相同的格式）

    Raises:
        ValueError: 当 box_format 不是 'xyxy' 或 'xywh' 时
    """
    # 参数验证
    if box_format not in ('xyxy', 'xywh'):
        raise ValueError(f"Invalid box_format: '{box_format}'. Must be 'xyxy' or 'xywh'")

    # 解包参数并类型转换
    ratio = float(scale_params['ratio'])
    pad_left, pad_top = map(float, scale_params['pad'])
    target_w, target_h = map(int, scale_params['target_size'])

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

    # 应用缩放和填充
    def _convert(val: float, pad: float, limit: float) -> float:
        """先缩放再添加填充，并限制在 [0, limit] 范围内"""
        return max(0.0, min(val * ratio + pad, limit))

    x1_scaled = _convert(x1, pad_left, target_w)
    y1_scaled = _convert(y1, pad_top, target_h)
    x2_scaled = _convert(x2, pad_left, target_w)
    y2_scaled = _convert(y2, pad_top, target_h)

    # 返回与输入相同的格式
    if box_format == 'xyxy':
        return [x1_scaled, y1_scaled, x2_scaled, y2_scaled]
    else:
        return [
            (x1_scaled + x2_scaled) / 2,  # cx
            (y1_scaled + y2_scaled) / 2,  # cy
            x2_scaled - x1_scaled,        # w
            y2_scaled - y1_scaled         # h
        ]


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
    def _convert(val: float, pad: float, limit: float) -> float:
        """去除填充后反缩放，并限制在 [0, limit] 范围内"""
        return max(0.0, min((val - pad) / ratio, limit))

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
    # val(output_pred_path=Path('./op/predictions/'), output_label_path=Path('./op/labels/'))
    val()
    # concat_pred_gt_images(output_pred_path=Path('./op/predictions/'), output_label_path=Path('./op/labels/'), final_path=Path('./op/concat_images/'))
