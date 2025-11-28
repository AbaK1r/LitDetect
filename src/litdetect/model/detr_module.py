from typing import Dict, List

import torch
import torch.nn as nn
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.structures import Instances
from detectron2.utils.events import EventStorage


class DetrModule(nn.Module):
    def __init__(
        self,
        config_path: str = None,          # e.g., "projects/detr/configs/detr_r50_50ep.py"
        config_dict: dict = None,         # or pass dict directly (LazyConfig.load() result)
        num_classes: int = 80,            # will override cfg for custom dataset
        pretrained_weights: str = None,   # e.g., "detectron2://ImageNetPretrained/torchvision/R-50.pkl"
        conf_thres: float = 0.05,
    ):
        super().__init__()
        self.conf_thres = conf_thres

        # === Step 1: Load config ===
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

        # === Step 2: Build model (Detectron2 style) ===
        # Important: use instantiate, NOT build_model(cfg)
        self.model = instantiate(self.cfg.model)

        # Load pretrained weights (e.g., imagenet backbone or detrex DETR checkpoint)
        if pretrained_weights:
            DetectionCheckpointer(self.model).load(pretrained_weights)

    def forward(self, batched_inputs):
        """
        Compatible with DINO / detrex forward signature.
        Input: List[dict], each dict has:
            - "image": [3, H, W], float32, [0, 255] (unnormalized)
            - "height", "width": int (original size)
            - (optional) "instances": Instances (for training, not used in forward)
        Output: List[dict], each dict has:
            - "pred_boxes": Boxes (x1, y1, x2, y2) — unnormalized, image space
            - "scores": [N]
            - "pred_classes": [N]
        """
        return self.model(batched_inputs)

    def train_step(self, batch):
        # Detectron2 models expect EventStorage for logging intermediate losses
        with EventStorage():
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
