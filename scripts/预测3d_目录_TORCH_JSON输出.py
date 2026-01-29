import argparse
import json
from pathlib import Path

import hydra
import numpy as np
import torch
import torchio as tio
from omegaconf import OmegaConf
from tqdm import tqdm

from litdetect.scripts_init import get_logger, check_path, check_version

# 初始化日志记录器
logger = get_logger(__file__)

# 设置CUDA环境变量和精度
torch.set_float32_matmul_precision('high')


def parse_args():
    parser = argparse.ArgumentParser(description="arguments")

    parser.add_argument("-v", "--versions", type=int, help="version值")
    parser.add_argument("-i", "--input_dir", type=str, help="输入图像所在根路径")
    parser.add_argument("-o", "--output_dir", type=str, help="json文件保存路径", default='')
    parser.add_argument("--custom_list", type=str, help="自定义文件列表（相对于根路径）", default='')

    parser.add_argument("--input_min", type=float, help="归一化最小值", default=-400.)
    parser.add_argument("--input_max", type=float, help="归一化最大值", default=400.)

    parser.add_argument("--batch_size", type=int, help="批次大小", default=1)
    parser.add_argument("--suffix", type=str, help="图像后缀", default='.nii.gz')
    parser.add_argument("--threh", type=float, help="预测置信度阈值", default=0.05)

    args = vars(parser.parse_args())
    args['versions'] = check_version(args['versions'])
    return args


def main():
    parser_args = parse_args()

    data_dir = Path(parser_args['input_dir'])
    image_suffix = parser_args['suffix']
    threh = parser_args['threh']
    ddir = Path(f'lightning_logs/version_{parser_args['versions']}/ckpts/')
    args = OmegaConf.load(ddir.parent / 'full_config.yaml')

    # 输出目录设置
    output_dir = data_dir.parent / f'{args.model.model_name}_pred' if parser_args['output_dir'] == '' else Path(parser_args['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f'Output dir: {output_dir}')

    # 检查点加载
    ckpts = ddir.rglob('*.ckpt')
    ckpts = [(float(i.stem.split('=')[-1]), i) for i in ckpts if i.stem != 'last']
    ckpts.sort(key=lambda _x: _x[0])
    if len(ckpts) == 0:
        raise ValueError(f'No ckpt found in {ddir}')
    ckpt = ckpts[-1][1] if args.callbacks.call_back_mode == 'max' else ckpts[0][1]
    logger.info(f'Load ckpt: {ckpt}')

    # 模型加载
    normalize_args = args.data.augmentation_val.transforms[-2]
    model = hydra.utils.instantiate(
        args.model, pixel_mean=normalize_args.mean, pixel_std=normalize_args.std).eval().cuda()
    sd = torch.load(ckpt, weights_only=False, map_location='cpu')['state_dict']
    model.load_state_dict(sd, strict=False)

    transform = tio.Compose([
        tio.Clamp(out_min=parser_args['input_min'], out_max=parser_args['input_max']),
        tio.RescaleIntensity(out_min_max=(0., 255.), in_min_max=(parser_args['input_min'], parser_args['input_max'])),
    ])

    # 推理过程
    with torch.inference_mode():
        if parser_args['custom_list'] != '':
            with open(parser_args['custom_list'], 'r') as f:
                custom_list = f.readlines()
            pic_paths = [data_dir / i.strip() for i in custom_list]
            for pic_path in pic_paths:
                assert pic_path.exists(), f'{pic_path} does not exist'
        else:
            pic_paths = list(data_dir.rglob(f'*{image_suffix}'))
        if len(pic_paths) == 0:
            raise ValueError(f'No input found !')

        for pic_path in tqdm(pic_paths):
            subject = transform(tio.Subject(image=tio.ScalarImage(pic_path)))
            assert subject.image.data.shape[1:3] == (512, 512), f'{pic_path} is not a 512x512 image'
            assert len(subject.image.data.shape)== 4, f'{pic_path} is not a HWD image'
            assert subject.image.data.dtype == torch.float32, f'{pic_path} is not a float32 image'

            grid_sampler = tio.inference.GridSampler(
                transform(subject),
                (512, 512, 3),
                (0, 0, 2),
            )
            patch_loader = tio.SubjectsLoader(grid_sampler, batch_size=parser_args['batch_size'])
            output_preds = []
            for patches_batch in patch_loader:
                locations = patches_batch[tio.LOCATION]  # tensor([[y0, x0, z0, y1, x1, z1]])
                z = (locations[:, 2] + locations[:, 5]) // 2
                x = patches_batch['image'][tio.DATA].to(model.device)
                preds = model(preprocess(x))
                preds = postprocess(preds.cpu().numpy(), threh, (512, 512))

                for _z in range(len(preds)):
                    pred = preds[_z]
                    if pred['boxes'].shape[0] > 0:
                        output_preds.extend([{
                            "box": pred['boxes'][i].tolist()+[z[_z].item()],
                            "score": pred['scores'][i].item(),
                            "class": args.data.class_name[int(pred['labels'][i].item())],
                        } for i in range(pred['boxes'].shape[0])])

            save_json(pic_path, output_preds, subject.shape[1:], data_dir, output_dir)

# 预处理函数
def preprocess(x):
    return x[:, 0].permute(0, 3, 1, 2)

def postprocess(pred: np.ndarray, conf_threshold=0.05, input_shape_hw=(512, 512)):
    """

    Args:
        pred: (B, N, xyxy+score+class)
        conf_threshold:
        input_shape_hw: 

    Returns: List[Dict[str, np.ndarray]]

    """
    bs, n_box, _ = pred.shape
    b_bboxes = pred[..., :4].reshape(bs, -1, 4)
    b_scores = pred[..., 4].reshape(bs, -1)
    b_classes = pred[..., 5].reshape(bs, -1).astype(np.int64)
    H, W = input_shape_hw
    output = []
    for bboxes, scores, classes in zip(b_bboxes, b_scores, b_classes):
        inds = np.where(scores > conf_threshold)[0]
        bboxes, scores, classes = bboxes[inds], scores[inds], classes[inds]
        if len(bboxes.shape) == 1:
            bboxes, scores, classes = bboxes[None], scores[None], classes[None]
        bboxes[:, 0] *= W
        bboxes[:, 1] *= H
        bboxes[:, 2] *= W
        bboxes[:, 3] *= H
        output.append({
            'boxes': bboxes,  # (N, 4) xyxy
            'scores': scores,  # (N, 1)
            'labels': classes,  # (N, 1)
        })

    return output

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
        "imageDepth": orig_shape[2],
        "description": ""
    }

    for pred in preds:
        x1, y1, x2, y2, z = pred['box']
        points = [
            [x1, y1, z],
            [x2, y1, z],
            [x2, y2, z],
            [x1, y2, z]
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
    if not image_path.name.endswith('.nii.gz'):
        json_save_name = image_path.relative_to(input_dir).with_suffix('.json')
    else:
        json_save_name = str(image_path.relative_to(input_dir)).replace('.nii.gz', '.json')
    json_save_path = output_dir / json_save_name
    json_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_save_path, 'w', encoding='utf-8') as f:
        # noinspection PyTypeChecker
        json.dump(json_data, f, indent=2, ensure_ascii=False)

# 程序入口
if __name__ == '__main__':
    check_path(__file__)
    main()
