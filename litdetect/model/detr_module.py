from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.structures import Boxes, Instances
from litdetect.model.model_interface import ModuleInterface
from litdetect.scripts_init import get_logger

logger = get_logger(__file__)


class ModuleWrapper(ModuleInterface):

    @property
    def model_class(self):
        return DetrModule

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def input_batch_trans(self, batch):
        """

        Args:
            batch: [{
                'image':           # CHW, float32, nomalized
                'bboxes':          # (n, 4) (x1, y1, x2, y2)
                'labels':          # (n,)
                'orig_size':       # (2)
                'input_size_hw':   # (2)
                'image_id':        # scalar int/long
            }]

        Returns:
            batch: [{
                "image":           # CHW, float32, nomalized
                "height":          # height (for evaluation rescaling)
                "width":           # width
                "instances":       # Instances object
            }]

        """
        return [
            {
                'image': i['image'],
                'height': i['orig_size'][0],
                'width': i['orig_size'][1],
                'instances': Instances(
                    image_size=(i['input_size_hw'][0], i['input_size_hw'][1]),
                    gt_boxes=Boxes(i['bboxes']),
                    gt_classes=i['labels']
                )
            } for i in batch
        ] if self.training else [
            {'image': i['image']} for i in batch
        ]

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 1e-4
        if self.hparams.optimizer != 'adam':
            logger.warn('Only AdamW optimizer is supported for now!')

        param_dicts = [
            {"params": [p for n, p in self.named_parameters() if "backbone" not in n and p.requires_grad]},
            {"params": [p for n, p in self.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": self.hparams.lr_backbone},
        ]
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.hparams.lr,
            weight_decay=weight_decay
        )

        if self.hparams.lr_scheduler is None:
            return optimizer
        else:
            if self.hparams.lr_scheduler == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=self.hparams.lr_step_size, gamma=self.hparams.lr_gamma)
            else:
                raise ValueError('Invalid lr_scheduler type!')
            return {
                'optimizer': optimizer,
                'lr_scheduler': {'scheduler': scheduler}
            }


class DetrModule(nn.Module):
    def __init__(
        self,
        config_path: str = None,          # e.g., "projects/detr/configs/detr_r50_50ep.py"
        config_dict: dict = None,         # or pass dict directly (LazyConfig.load() result)
        num_classes: int = 80,            # will override cfg for custom dataset
        pretrained_weights: str = None,   # e.g., "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
        conf_thres: float = 0.05,
        pixel_mean: List[float] = None,   # [123.675, 116.280, 103.530]
        pixel_std: List[float] = None,    # [58.395, 57.120, 57.375]
    ):
        super().__init__()
        self.conf_thres = conf_thres

        # === Load config ===
        if config_path is not None:
            self.cfg = LazyConfig.load(config_path)
        elif config_dict is not None:
            self.cfg = config_dict
        else:
            raise ValueError("Either config_path or config_dict must be provided.")

        # Override num_classes if needed (COCO has 80, but custom may differ)
        if hasattr(self.cfg.model, "num_classes"):
            self.cfg.model.num_classes = num_classes
        else:
            raise ValueError("cfg.model.num_classes must be set, but not found.")

        # === Build model (Detectron2 style) ===
        # Important: use instantiate, NOT build_model(cfg)
        self.model = instantiate(self.cfg.model)

        # Load pretrained weights (e.g., imagenet backbone or detrex DETR checkpoint)
        if pretrained_weights:
            DetectionCheckpointer(self.model).load(pretrained_weights)

        if pixel_mean is not None and pixel_mean is not None:
            pixel_mean = torch.tensor(pixel_mean, dtype=torch.float32).view(1, 3, 1, 1) * 255.
            pixel_std = torch.tensor(pixel_std, dtype=torch.float32).view(1, 3, 1, 1) * 255.
            self.register_buffer("pixel_mean", pixel_mean)
            self.register_buffer("pixel_std", pixel_std)
        else:
            self.pixel_mean = None
            self.pixel_std = None

    def forward(self, x):
        return self.dino_forward(x)

    def dino_forward(self, x):
        """

        Args:
            x: [B, 3, H, W]

        Returns:

        """
        if self.pixel_mean is not None and self.pixel_std is not None:
            x = (x - self.pixel_mean) / self.pixel_std

        batch_size, _, H, W = x.shape
        img_masks = x.new_zeros(batch_size, H, W)

        # original features
        features = self.model.backbone(x)  # output feature dict

        # project backbone features to the reuired dimension of transformer
        # we use multi-scale features in DINO
        multi_level_feats = self.model.neck(features)
        multi_level_masks = []
        multi_level_position_embeddings = []
        for feat in multi_level_feats:
            multi_level_masks.append(
                F.interpolate(img_masks[None], size=feat.shape[-2:]).to(torch.bool).squeeze(0)
            )
            multi_level_position_embeddings.append(self.model.position_embedding(multi_level_masks[-1]))
        input_query_label, input_query_bbox, attn_mask, dn_meta = None, None, None, None
        query_embeds = (input_query_label, input_query_bbox)

        # feed into transformer
        (
            inter_states,
            init_reference,
            inter_references,
            enc_state,
            enc_reference,  # [0..1]
        ) = self.model.transformer(
            multi_level_feats,
            multi_level_masks,
            multi_level_position_embeddings,
            query_embeds,
            attn_masks=[attn_mask, None],
        )
        # hack implementation for distributed training
        inter_states[0] += self.model.label_enc.weight[0, 0] * 0.0

        # Calculate output coordinates and classes.
        for lvl in range(inter_states.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            box_cls = self.model.class_embed[lvl](inter_states[lvl])
            tmp = self.model.bbox_embed[lvl](inter_states[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            box_pred = tmp.sigmoid()
        # tensor shape: [num_decoder_layers, bs, num_query, num_classes]
        # tensor shape: [num_decoder_layers, bs, num_query, 4]

        # box_cls.shape: (batch_size, num_queries, K)
        # box_pred.shape: (batch_size, num_queries, 4) The tensor predicts 4-vector (x,y,w,h) box
        prob = box_cls.sigmoid()
        topk_values, topk_indexes = torch.topk(
            prob.view(box_cls.shape[0], -1), self.model.select_box_nums_for_evaluation, dim=1
        )
        scores = topk_values
        topk_boxes = torch.div(topk_indexes, box_cls.shape[2], rounding_mode="floor")
        labels = topk_indexes % box_cls.shape[2]

        boxes = torch.gather(box_pred, 1, topk_boxes[..., None].repeat(1, 1, 4))
        boxes = box_cxcywh_to_xyxy(boxes)
        # boxes[..., 0] = boxes[..., 0] * W
        # boxes[..., 2] = boxes[..., 2] * W
        # boxes[..., 1] = boxes[..., 1] * H
        # boxes[..., 3] = boxes[..., 3] * H
        output = torch.concat([boxes, scores[..., None], labels[..., None].float()], dim=2)

        return output
        # B, Q, K = box_cls.shape
        #
        # # 1) probabilities
        # prob = box_cls.sigmoid()  # (B, Q, K)
        #
        # # 2) scores flattened: (B, Q*K)
        # scores = prob.reshape(B, Q * K)  # flatten class dim
        #
        # # 3) labels: create tensor [0..K-1] and expand to (B, Q, K) then flatten to (B, Q*K)
        # labels = torch.arange(K, device=box_cls.device, dtype=torch.float32)  # (K,)
        # labels = labels.view(1, 1, K).expand(B, Q, K).reshape(B, Q * K)  # (B, Q*K)
        #
        # # 4) boxes: replicate each query's box for each class
        # # box_pred: (B, Q, 4) -> (B, Q, K, 4) -> flatten to (B, Q*K, 4)
        # boxes = box_pred.unsqueeze(2).expand(B, Q, K, 4).reshape(B, Q * K, 4)  # (B, Q*K, 4)
        #
        # # 5) convert format if needed (e.g., cxcywh -> xyxy)
        # boxes = box_cxcywh_to_xyxy(boxes)  # make sure this supports (B, N, 4)
        #
        # # 6) concat -> (B, Q*K, 6)
        # output = torch.cat([boxes, scores.unsqueeze(-1), labels.unsqueeze(-1)], dim=2)
        #
        # return output

    def train_step(self, batch):

        """
        Compatible with detrex forward signature.
        Input: List[dict], each dict has:
            - "image": [3, H, W], float32, normalized
            - "height", "width": int (original size)
            - (optional) "instances": Instances (for training, not used in forward)
        Output: List[dict], IF TRAINING, each dict has:
            - "loss_...": Tensor
            IF EVALUATION, each dict has:
            - "instances": Instances
        """
        loss_dict = self.model(batch)

        loss_dict['loss'] = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))

        return loss_dict

    def val_step(self, batch):
        device = batch[0]['image'].device
        # Run inference
        outputs: List[Dict[str, Instances]] = self.model(batch)  # List[dict]

        # --- 🔸 Parse predictions: convert to List[Tensor[N, 6]] (x1, y1, x2, y2, score, label) ---
        preds: List[torch.Tensor] = []
        for out in outputs:
            instances = out["instances"]
            boxes = instances.pred_boxes.tensor.to(device)  # [N, 4], xyxy, unnormalized
            scores = instances.scores.to(device)  # [N]
            labels = instances.pred_classes.to(device).float()  # [N] → float for cat

            # Filter by confidence
            keep = scores > self.conf_thres
            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            if len(boxes) == 0:
                pred = torch.empty((0, 6), device=device)
            else:
                pred = torch.cat([boxes, scores.unsqueeze(1), labels.unsqueeze(1)], dim=1)  # [N, 6]
            preds.append(pred)
        return preds

    def test_step(self, *args, **kwargs):
        return self.val_step(*args, **kwargs)


def inverse_sigmoid(x, eps=1e-3):
    """
    The inverse function for sigmoid activation function.
    Note: It might face numberical issues with fp16 small eps.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def box_cxcywh_to_xyxy(bbox) -> torch.Tensor:
    """Convert bbox coordinates from (cx, cy, w, h) to (x1, y1, x2, y2)

    Args:
        bbox (torch.Tensor): Shape (n, 4) for bboxes.

    Returns:
        torch.Tensor: Converted bboxes.
    """
    cx, cy, w, h = bbox.unbind(-1)
    new_bbox = [(cx - 0.5 * w), (cy - 0.5 * h), (cx + 0.5 * w), (cy + 0.5 * h)]
    return torch.stack(new_bbox, dim=-1)


# --------------------------------------------------
# Usage Example
# --------------------------------------------------
def main():
    # Load config from detrex (assume detrex is in PYTHONPATH or installed)
    config_path = "/data/16t/wxh/detrex/projects/dino/configs/dino-convnext/dino_convnext_small_384_4scale_12ep.py"  # relative to detrex root
    pretrained_weights = "/data/16t/wxh/ds/ckpts/dino_convnext_small_384_4scale_12ep.pth"
    # For custom classes (e.g., 5 classes), override num_classes
    model = DetrModule(
        config_path=config_path,
        num_classes=5,
        pretrained_weights=pretrained_weights,  # imagenet R50
        # or detrex checkpoint: "https://github.com/IDEA-Research/detrex/releases/download/v0.3.0/detr_r50_50ep.pth"
    )
    print(model)
    # Then plug into Lightning Trainer as usual


if __name__ == "__main__":
    main()
