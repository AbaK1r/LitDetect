from typing import Optional, Tuple

import torch
import torch.nn as nn
import torchvision
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig
from detectron2.modeling import build_model
from detectron2.structures import Boxes, Instances


class DINO(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        config_file: str = "/data/16t/wxh/detrex/projects/dino/configs/dino-convnext/dino_convnext_small_384_4scale_12ep.py",
        weights_path: Optional[str] = None,
        iou_thres: float = 0.45,
        conf_thres: float = 0.05,
        input_size_hw: Tuple[int, int] = (512, 512),
    ):
        super().__init__()

        self.iou_thres = iou_thres
        self.conf_thres = conf_thres
        self.input_size = input_size_hw
        self.max_wh = max(*input_size_hw)

        # 2️⃣ 加载 detrex 配置
        self.cfg = LazyConfig.load(config_file)

        # 调整 config 中的 num_classes & input size
        self.cfg.model.num_classes = num_classes
        self.cfg.train.dataset.names = "dummy_train"  # 后续被忽略，因我们接管 data
        self.cfg.test.dataset.names = "dummy_val"
        # 可选：调整 image size
        self.cfg.dataloader.train.mapper.image_size = input_size_hw
        self.cfg.dataloader.test.mapper.image_size = input_size_hw

        # 3️⃣ 构建完整模型
        self.model = build_model(self.cfg)
        self.model.train()

        # 4️⃣ 加载预训练权重（不含 backbone，或带 backbone？）
        if pretrained_weights:
            # /data/16t/wxh/ds/ckpts/dino_convnext_small_384_4scale_12ep.pth
            # 若 weights 包含 backbone（如官方 dino_4scale_r50），则直接 load；
            # 若仅 head 权重，则需 strict=False
            checkpointer = DetectionCheckpointer(self.model)
            checkpointer.resume_or_load(pretrained_weights, resume=False)

        # 5️⃣ 参数分组
        # backbone_params_set = set(self.backbone.parameters())
        # backbone_named_params = dict(self.backbone.named_parameters())
        #
        # self.hidden_weights = [p for n, p in backbone_named_params.items() if p.ndim >= 2 and p.requires_grad]
        # self.hidden_gains_biases = [p for n, p in backbone_named_params.items() if p.ndim < 2 and p.requires_grad]
        # self.nonhidden_params = [p for p in self.model.parameters() if p not in backbone_params_set]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, 3, H, W], values in [0, 1] or [0, 255] — detrex 内部会 normalize
        Returns:
            [B, N, num_classes + 1, 5] → xyxy + score (类似 FasterRcnn 输出格式)
        """
        B = images.shape[0]
        device = images.device

        # 仿 detectron2 输入格式：List[Dict]
        batched_inputs = [{"image": (img * 255).clamp(0, 255)} for img in images]  # detrex expects 0~255 uint8-style float

        # 获取原始 image_size for resize back
        original_sizes = [(img.shape[-2], img.shape[-1]) for img in images]

        # 推理
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(batched_inputs)
        self.model.train()

        # outputs: List[Instances], each has pred_boxes (Boxes), scores, pred_classes
        results = []
        for i, (output, orig_hw) in enumerate(zip(outputs, original_sizes)):
            pred_boxes = output.pred_boxes.tensor  # (N, 4) xyxy
            scores = output.scores                 # (N,)
            labels = output.pred_classes.int()     # (N,)

            # filter by conf
            keep = scores > self.conf_thres
            pred_boxes = pred_boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            if len(pred_boxes) == 0:
                res = torch.zeros((0, 6), device=device)
            else:
                # apply NMS per-class
                boxes = pred_boxes.clone()
                labels_f = labels.float().unsqueeze(1)
                c = labels_f * self.max_wh  # class offset trick (like YOLO)
                boxes_offset = boxes + c

                nms_idx = torchvision.ops.nms(boxes_offset, scores, self.iou_thres)
                boxes = boxes[nms_idx]
                scores = scores[nms_idx]
                labels = labels[nms_idx]

                # format: [xyxy, score, label]
                res = torch.cat([boxes, scores.unsqueeze(1), labels.unsqueeze(1).float()], dim=1)  # (M, 6)

            results.append(res)

        # pad to same len (optional), or return list → but your ModuleInterface expects tensor
        # Here we follow FasterRcnn’s style: return List[Tensor] in val_step, but forward() returns stacked?
        # Since your FasterRcnn.forward returns [B, *, *, 5], we mimic:
        max_det = max(len(r) for r in results) or 1
        padded = torch.zeros(B, max_det, 6, device=device)
        for i, r in enumerate(results):
            padded[i, : len(r)] = r

        # reshape to [B, N, C, 5] ? Your FasterRcnn uses:
        #     outputs = outputs.view(batch_size, -1, n_class, 5)
        # But DINO is dense one-stage → no "per-class proposals" axis.
        # So we simply return [B, N, 6] and let postprocess handle class split if needed.
        return padded  # [B, N, 6] → xyxy, score, class

    def train_step(self, batch):
        images, targets = batch
        # detrex expects: List[Dict], each dict: {"image": Tensor (3,H,W) ∈ [0,255], "instances": Instances}
        batched_inputs = []
        for img, tgt in zip(images, targets):
            # Convert to detrex format
            H, W = img.shape[-2:]
            instances = Instances(image_size=(H, W))
            instances.gt_boxes = Boxes(tgt["boxes"])  # xyxy
            instances.gt_classes = tgt["labels"].long()  # 0-based

            batched_inputs.append({
                "image": (img * 255).clamp(0, 255),  # scale to [0, 255]
                "instances": instances
            })

        # Forward + loss (detrex model returns dict of losses)
        loss_dict = self.model(batched_inputs)

        total_loss = sum(loss for loss in loss_dict.values() if "loss" in loss)
        loss_dict["loss"] = total_loss
        return loss_dict

    def val_step(self, batch):
        images = batch[0]  # List[Tensor] or Tensor? Your FasterRcnn uses batch[0] as images list
        if isinstance(images, torch.Tensor):
            images = [images[i] for i in range(len(images))]

        # Use forward to get results (already NMS + conf filtered)
        outputs = self.forward(torch.stack(images))  # [B, N, 6]
        B = outputs.shape[0]
        device = outputs.device

        # Convert to your FasterRcnn style return: List[Tensor(N, 6)], label already 0-based
        results = []
        for i in range(B):
            pred = outputs[i]  # (N, 6)
            valid = pred[:, 4] > self.conf_thres
            pred = pred[valid]
            # your FasterRcnn returns: [xyxy, score, label(class_idx, 0-based)]
            results.append(pred)
        return results

