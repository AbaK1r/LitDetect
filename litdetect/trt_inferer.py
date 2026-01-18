from typing import List, Dict

import numpy as np
import simple_trt_infer
import torch
import torchvision


def get_inferer(target_name, model_path, encrypt, **kwargs):
    if target_name == 'litdetect.model.faster_rcnn.ModuleWrapper':
        INFERER = FasterRCNN_TRTInferer
    elif target_name == 'litdetect.model.yolo11.ModuleWrapper':
        INFERER = Yolo11_TRTInferer
    elif target_name == 'litdetect.model.detr_module.ModuleWrapper':
        INFERER = DINO_TRTInferer
    else:
        raise ValueError(f'model must be faster_rcnn or yolo11 or detr, but got {target_name}')
    return INFERER(model_path, encrypt, **kwargs)


"""
TRT Inferer

Input: (B, C, H, W) 0-255. float32

output: (B, N, xyxy+score+class)
    [                           # shape  fmt    range    dtype
        {                       # ------------------------------
            'boxes': bboxes,    # (N,4)  xyxy  0-H 0-W  float32
            'scores': scores,   # (N,1)        0-1.     float32
            'labels': classes,  # (N,1)        0-class  float32
        }
    ]
"""


class DINO_TRTInferer:
    def __init__(self, model_path, encrypt, **kwargs):
        self.model = simple_trt_infer.simple_model(model_path, False, encrypt)
        self.input_shape = self.model.get_input_shape()
        self.output_shape = self.model.get_output_shape()

    def inference(self, ipt, conf_threshold=0.05) -> List[Dict[str, np.ndarray]]:
        ipt = np.ascontiguousarray(ipt)
        output = self.model.infer(ipt).astype(np.float32)
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


class FasterRCNN_TRTInferer:
    def __init__(self, model_path, encrypt, **kwargs):
        self.model = simple_trt_infer.simple_model(model_path, False, encrypt)
        self.input_shape = self.model.get_input_shape()
        self.output_shape = self.model.get_output_shape()

    def inference(self, ipt, conf_threshold=0.05, nms_threshold=0.45) -> List[Dict[str, np.ndarray]]:
        ipt = np.ascontiguousarray(ipt)
        output = self.model.infer(ipt).astype(np.float32)
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


class Yolo11_TRTInferer:
    def __init__(self, model_path, encrypt, **kwargs):
        self.model = simple_trt_infer.simple_model(model_path, False, encrypt)
        self.input_shape = self.model.get_input_shape()
        self.output_shape = self.model.get_output_shape()

    def inference(self, ipt, *args, **kwargs) -> List[Dict[str, np.ndarray]]:
        ipt = np.ascontiguousarray(ipt)
        output = self.model.infer(ipt).astype(np.float32)
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