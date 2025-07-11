import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision
from timm import create_model
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.rpn import concat_box_prediction_layers


class FasterRcnn(nn.Module):
    def __init__(self, num_classes=5, backbone_name='resnet34', iou_thres=0.45, conf_thres=0.25, input_size=(512, 512), pretrained=False):
        super().__init__()
        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.input_size = input_size
        self.max_wh = max(*input_size)
        self.min_wh = min(*input_size)
        backbone = BackBone(model_name=backbone_name, pretrained=pretrained)
        anchor_generator = AnchorGenerator(sizes=((128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),))
        self.model = FasterRCNN(backbone, num_classes=num_classes+1, rpn_anchor_generator=anchor_generator, min_size=self.min_wh)

    def _sim_rpn(self, images, features):
        features = list(features.values())
        objectness, pred_bbox_deltas = self.model.rpn.head(features)
        anchors = self.model.rpn.anchor_generator(images, features)

        num_images = len(anchors)
        num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
        num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
        objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
        # apply pred_bbox_deltas to anchors to obtain the decoded proposals
        # note that we detach the deltas because Faster R-CNN do not backprop through
        # the proposals
        proposals = self.model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
        proposals = proposals.view(num_images, -1, 4)

        num_images = proposals.shape[0]
        device = proposals.device
        # do not backprop through objectness
        objectness = objectness.detach()
        objectness = objectness.reshape(num_images, -1)

        # select top_n boxes independently per level before applying nms
        top_n_idx = self.model.rpn._get_top_n_idx(objectness, num_anchors_per_level)

        image_range = torch.arange(num_images, device=device)
        batch_idx = image_range[:, None]
        proposals = proposals[batch_idx, top_n_idx]
        proposals = [box_ops.clip_boxes_to_image(boxes, img_shape) for boxes, img_shape in zip(proposals, images.image_sizes)]
        return proposals

    def forward(self, images):
        """

        Args:
            images: [batch_size, 3, h, w]

        Returns: [batch_size, num_proposals, num_classes, xyxy+scores]

        """
        batch_size = images.shape[0]
        original_image_size = images.shape[2:]

        images, targets = self.model.transform([images[i] for i in range(images.shape[0])], None)
        features = self.model.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        proposals = self._sim_rpn(images, features)

        box_features = self.model.roi_heads.box_roi_pool(features, proposals, images.image_sizes)
        box_features = self.model.roi_heads.box_head(box_features)
        class_logits, box_regression = self.model.roi_heads.box_predictor(box_features)

        pred_boxes = self.model.roi_heads.box_coder.decode(box_regression, proposals)
        scores = F.softmax(class_logits, 1)[:, 1:]
        boxes = box_ops.clip_boxes_to_image(pred_boxes, images.image_sizes[0])[:, 1:]

        boxes_resize_ratios = [s / s_orig for s, s_orig in zip(original_image_size, images.image_sizes[0])]
        # boxes[..., (0, 2)] = boxes[..., (0, 2)] * boxes_resize_ratios[1]
        boxes[..., 0] = boxes[..., 0] * boxes_resize_ratios[1]
        boxes[..., 2] = boxes[..., 2] * boxes_resize_ratios[1]
        # boxes[..., (1, 3)] = boxes[..., (1, 3)] * boxes_resize_ratios[0]
        boxes[..., 1] = boxes[..., 1] * boxes_resize_ratios[0]
        boxes[..., 3] = boxes[..., 3] * boxes_resize_ratios[0]

        outputs = torch.concatenate((boxes, scores[..., None]), dim=2)
        n_class = outputs.shape[1]
        outputs = outputs.view(batch_size, -1, n_class, 5)
        return outputs

    def train_step(self, batch):
        images, targets = batch
        for i in range(len(targets)):
            targets[i]['labels'] += 1
        _images, _targets = [], []
        for i in range(len(targets)):
            if targets[i]['labels'].nelement() != 0:
                _images.append(images[i])
                _targets.append(targets[i])
        loss_dict = self.model(_images, _targets)
        loss = sum(loss for loss in loss_dict.values())
        loss_dict['loss'] = loss
        return loss_dict

    def val_step(self, batch):
        images = batch[0]
        bs = len(images)
        outputs = self.model(images)
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        """
        res: {
            0.0: {
                'boxes': (100, 4),  # 未归一化
                'labels': (100,),  # int
                'scores': (100,)  # 0~1
            },
            1.0: {...},
            ...
        }
        """
        DEVICE = images[0].device
        results = [torch.zeros((0, 6), device=DEVICE)] * bs
        for xi, i in enumerate(outputs):  # image index, image inference
            x = torch.cat((i['boxes'], i['scores'][:, None], i['labels'].float()[:, None]-1), 1)[i['scores'].view(-1) > self.conf_thres]
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue
            c = x[:, 5:6] * self.max_wh  # classes
            scores = x[:, 4]  # scores
            boxes = x[:, :4] + c  # boxes (offset by class)
            nms_idx = torchvision.ops.nms(boxes, scores, 0.45)  # NMS
            results[xi] = x[nms_idx]
        return results


class BackBone(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=False):
        super().__init__()
        self.model = create_model(model_name, pretrained=pretrained, features_only=True, out_indices=[-1])
        self.out_channels = self.model.feature_info.get('num_chs', -1)

    def forward(self, x):
        return self.model(x)[0]