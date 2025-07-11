import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class MetricCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.map = None

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.map is None:
            self.map = MeanAveragePrecision(
                box_format="xyxy",
                iou_type="bbox",
                class_metrics=False,
                backend="pycocotools"
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        preds = outputs["preds"]
        targets = outputs["targets"]
        self.map.update(preds, targets)

    def on_validation_epoch_end(self, trainer, pl_module):
        map_res = self.map.compute()

        map_res_ = {}
        for k, v in map_res.items():
            if k == 'classes':
                continue
            # if v.numel() > 1:
            #     for i in range(v.shape[0]):
            #         map_res_[f"metrics/{k}_{i}"] = v[i].item()
            if v.numel() == 1 and v.item() != -1:
                map_res_[f"metrics/{k}"] = v.item()

        pl_module.log_dict(map_res_, sync_dist=True)
        self.map.reset()
