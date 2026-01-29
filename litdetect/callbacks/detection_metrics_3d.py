import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch

from litdetect.scripts_init import get_logger
from .ap import plot_froc_curve, plot_mc_curve, plot_pr_curve, smooth

# 初始化日志记录器
logger = get_logger(__file__)
# 屏蔽特定警告
warnings.filterwarnings(
    "ignore",
    message="It is recommended to use `self.log"
)


class DetectionMetrics3DCallback(pl.Callback):
    """
    目标检测指标计算回调，支持单卡
    功能：
    1. 计算mAP系列指标（mAP50, mAP50-95）
    2. 计算每个类别的精确率(P)和召回率(R)
    3. 统计每个类别的图像数和实例数
    4. 输出YOLO风格的评估表格

    参数:
    class_names: 类别名称列表（索引必须与标签索引匹配）
    iou_threshold: 用于计算精确率和召回率的IoU阈值（默认0.5）
    """

    def __init__(self, class_names: List[str], iou_threshold: float = 0.5, data_csv_dir: Path = None, output_root: Path = None):
        super().__init__()
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.iou_threshold = iou_threshold

        assert data_csv_dir is not None, "请指定数据CSV文件"
        assert output_root is not None, "请指定输出目录"
        self.image_name2box_id, self.data_name2num_boxes = self.read_csv(data_csv_dir)
        self.output_root = output_root

        # -------- epoch 内缓存 --------
        self.matches = None           # image-level IoU + score 信息

    def read_csv(self, csv_file):
        """
        image_name,tumor_id
        156.51465374王艳2025.5.20_slice_026.png,1
        156.51465374王艳2025.5.20_slice_027.png,1
        26.51539979许素珍2025.7.12_slice_028.png,1
        26.51539979许素珍2025.7.12_slice_029.png,1
        26.51539979许素珍2025.7.12_slice_030.png,1
        26.51539979许素珍2025.7.12_slice_030.png,3
        26.51539979许素珍2025.7.12_slice_031.png,1
        26.51539979许素珍2025.7.12_slice_036.png,2
        26.51539979许素珍2025.7.12_slice_037.png,2

        这个csv读成

        image_name2box_id = {
            "156.51465374王艳2025.5.20_slice_026.png": [1],
            "156.51465374王艳2025.5.20_slice_027.png": [1],
            "26.51539979许素珍2025.7.12_slice_028.png": [1],
            "26.51539979许素珍2025.7.12_slice_029.png": [1],
            "26.51539979许素珍2025.7.12_slice_030.png": [1, 3],
            "26.51539979许素珍2025.7.12_slice_031.png": [1],
            "26.51539979许素珍2025.7.12_slice_036.png": [2],
            "26.51539979许素珍2025.7.12_slice_037.png": [2],
        }
        data_name2num_boxes = {
            "156.51465374王艳2025.5.20": 1,
            "26.51539979许素珍2025.7.12": 3,
        }

        Args:
            csv_file:

        Returns:

        """
        # 读取CSV文件
        df = pd.read_csv(csv_file)

        # 1. 创建 image_name2box_id 字典
        image_name2box_id = defaultdict(list)
        for _, row in df.iterrows():
            img = row["image_name"]
            tid = int(row["tumor_id"])
            if tid not in image_name2box_id[img]:
                image_name2box_id[img].append(tid)

        # 2. 创建 data_name2num_boxes 字典
        # 提取data_name（去掉_slice_XXX.png部分）
        data_name2num_boxes = defaultdict(int)
        for img, tids in image_name2box_id.items():
            pos = img.rfind("_slice_")
            base = img[:pos] if pos != -1 else img.rsplit(".", 1)[0]
            data_name2num_boxes[base] = max(data_name2num_boxes[base], max(tids))

        return image_name2box_id, data_name2num_boxes

    def on_validation_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        在每个验证epoch开始时初始化指标和统计信息
        """
        # image-level 原始匹配信息
        # [{
        #     "data_name":    str,
        #     "image_name":   str,
        #     "gt_tumor_ids": list[int], length = N
        #     "ious":         (N, M)
        #     "pred_scores":  (M,)
        # }]
        self.matches = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict,
        batch: Dict,
        batch_idx: int,
        dataloader_idx: int = 0
    ):
        """
        在每个验证批次结束时更新指标和收集统计信息
        """
        DEVICE = pl_module.device

        for pred, target, bi in zip(
                outputs["preds"], outputs["targets"], batch
        ):
            image_name = bi["image_name"]
            # pred:   {'boxes':  tensor([M, 4]),
            #          'labels': tensor([M, ]),
            #          'scores': tensor([M, ])}
            # target: {'boxes':  tensor([N, 4]),
            #          'labels': tensor([N, ])}
            # image_name: str
            pred_boxes = pred["boxes"]
            pred_scores = pred["scores"]

            gt_boxes = target["boxes"]

            # ----------------------------
            # 1. 如果没有 GT 或没有 pred，仍需记录结构
            # ----------------------------
            if gt_boxes.numel() == 0 or pred_boxes.numel() == 0:
                iou = torch.zeros((gt_boxes.shape[0], pred_boxes.shape[0]), device=DEVICE)
            else:
                iou = box_iou(gt_boxes, pred_boxes)
                # filtered_iou_mask = ~ torch.all(iou < self.iou_threshold, dim=0)
                # iou = iou[:, filtered_iou_mask]
                # pred_scores = pred_scores[filtered_iou_mask]
                # pred_scores = torch.cat([pred_scores[filtered_iou_mask], pred_scores[~filtered_iou_mask]], dim=0)

            # ----------------------------
            # 3. 从 CSV 映射 image → tumor_id
            # ----------------------------
            # image_name 对应的 tumor_id 列表
            gt_tumor_ids = self.image_name2box_id.get(image_name, [])

            # data_name：去掉 _slice_XXX.png
            slice_pos = image_name.rfind("_slice_")
            if slice_pos != -1:
                data_name = image_name[:slice_pos]
            else:
                data_name = image_name.rsplit(".", 1)[0]

            # ----------------------------
            # 4. 存储 image-level 原始信息
            # ----------------------------
            self.matches.append({
                "data_name": data_name,
                "image_name": image_name,
                "gt_tumor_ids": gt_tumor_ids,          # list[int], length = N
                "ious": iou.detach(),                  # (N, M')
                "pred_scores": pred_scores.detach(),   # (M,)
            })

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        # torch.save(self.matches, self.output_root / "matches.pth")
        start_time = time.time()
        self._record_stats(trainer, pl_module)
        logger.info(f"DetectionMetrics3DCallback: {time.time() - start_time:.2f}s")
        self.matches = []

    def _record_stats(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Compute 3D detection metrics with confidence sweep.
        FP is computed at box-group (3D proposal) level.
        """

        # ------------------------------------------------
        # 0. 基础统计量
        # ------------------------------------------------
        conf_thresholds = np.linspace(0.0, 1.0, 101)
        num_scans = len(self.data_name2num_boxes)
        num_gt_total = sum(self.data_name2num_boxes.values())

        metric_curve = {
            "conf": [],
            "tp": [],
            "fp": [],
            "fn": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "fp_per_scan": [],
        }
        # ------------------------------------------------
        # 1. confidence sweep
        # ------------------------------------------------
        for conf_thr in conf_thresholds:

            # ----------------------------
            # 1.1 GT 级（3D tumor）匹配
            # ----------------------------
            data_level_match = {
                data_name: {
                    tumor_id: 0.0
                    for tumor_id in range(1, num_boxes + 1)
                }
                for data_name, num_boxes in self.data_name2num_boxes.items()
            }

            for item in self.matches:
                data_name = item["data_name"]
                gt_tumor_ids = item["gt_tumor_ids"]
                ious = item["ious"]  # (N_gt, M)
                pred_scores = item["pred_scores"]  # (M,)

                if ious.numel() == 0:
                    continue

                valid_mask = pred_scores >= conf_thr
                if valid_mask.sum() == 0:
                    continue

                filtered_ious = ious[:, valid_mask]

                for gt_idx, tumor_id in enumerate(gt_tumor_ids):
                    max_iou = filtered_ious[gt_idx].max().item()
                    if max_iou > data_level_match[data_name][tumor_id]:
                        data_level_match[data_name][tumor_id] = max_iou

            tp = 0
            for data in data_level_match.values():
                for best_iou in data.values():
                    if best_iou >= self.iou_threshold:
                        tp += 1

            fn = num_gt_total - tp

            # ----------------------------
            # 1.2 box 组级 FP / TP_group
            # ----------------------------
            tp_group = 0
            total_groups = 0

            # 按 data_name 聚合所有预测
            data2pred = defaultdict(list)

            for item in self.matches:
                data_name = item["data_name"]
                image_name = item["image_name"]
                slice_idx = parse_slice_index(image_name)

                pred_scores = item["pred_scores"]
                ious = item["ious"]  # (N_gt, M_pred)

                if pred_scores.numel() == 0:
                    continue

                valid_mask = pred_scores >= conf_thr
                if valid_mask.sum() == 0:
                    continue

                for box_idx in torch.where(valid_mask)[0].tolist():
                    data2pred[data_name].append({
                        "slice": slice_idx,
                        "box_idx": box_idx,
                        "ious": ious[:, box_idx],  # 与所有 GT 的 IoU
                    })

            # 对每个 data_name 建 3D box 组
            for data_name, boxes in data2pred.items():
                if len(boxes) == 0:
                    continue

                K = len(boxes)

                # 构建 box-box IoU（同 slice 无所谓，用 2D box IoU 即可）
                iou_mat = torch.zeros((K, K))
                for i in range(K):
                    for j in range(K):
                        if boxes[i]["slice"] == boxes[j]["slice"]:
                            iou_mat[i, j] = 1.0
                        else:
                            iou_mat[i, j] = 0.0  # slice 间是否相连由 slice ±1 决定

                groups = build_box_groups(
                    box_infos=boxes,
                    iou_matrix=iou_mat,
                    iou_thr=0.0
                )

                total_groups += len(groups)

                # 判断 TP 组
                for g in groups:
                    hit = False
                    for idx in g:
                        if boxes[idx]["ious"].numel() > 0 and \
                                boxes[idx]["ious"].max().item() >= self.iou_threshold:
                            hit = True
                            break
                    if hit:
                        tp_group += 1

            fp = total_groups - tp_group
            fp = max(fp, 0)

            # ----------------------------
            # 1.3 派生指标
            # ----------------------------
            precision = tp / (tp + fp + 1e-9)
            recall = tp / (tp + fn + 1e-9)
            f1 = 2 * precision * recall / (precision + recall + 1e-9)
            fp_per_scan = fp / num_scans

            metric_curve["conf"].append(conf_thr)
            metric_curve["tp"].append(tp)
            metric_curve["fp"].append(fp)
            metric_curve["fn"].append(fn)
            metric_curve["precision"].append(precision)
            metric_curve["recall"].append(recall)
            metric_curve["f1"].append(f1)
            metric_curve["fp_per_scan"].append(fp_per_scan)
        best_idx = np.argmax(metric_curve["f1"])
        print(f"best: conf-{metric_curve['conf'][best_idx]:.3f}, "
              f"f1-{metric_curve['f1'][best_idx]:.3f}, "
              f"tp-{metric_curve['tp'][best_idx]}, "
              f"fp-{metric_curve['fp'][best_idx]}, "
              f"fn-{metric_curve['fn'][best_idx]}, "
              f"precision-{metric_curve['precision'][best_idx]:.3f}, "
              f"recall-{metric_curve['recall'][best_idx]:.3f}")
        # ------------------------------------------------
        # 2. 绘图
        # ------------------------------------------------
        conf = np.array(metric_curve["conf"])
        precision = np.array(metric_curve["precision"])
        recall = np.array(metric_curve["recall"])
        f1 = np.array(metric_curve["f1"])
        fp_per_scan = np.array(metric_curve["fp_per_scan"])

        save_dir = self.output_root / "metrics"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 定义统一的横轴
        x_grid = np.linspace(0, 1, 1000)

        # ----------------------------
        # Precision-Confidence
        # ----------------------------
        precision_grid = np.interp(x_grid, conf, precision, left=0.0)
        precision_grid_smooth = smooth(precision_grid, 0.05)
        plot_mc_curve(
            x_grid,
            precision_grid_smooth[None, :],  # (1, 1000)
            save_dir=save_dir / "precision_confidence.png",
            xlabel="Confidence",
            ylabel="Precision",
        )
        logger.info(f"save metrics to {save_dir}")
        # ----------------------------
        # Recall-Confidence
        # ----------------------------
        recall_grid = np.interp(x_grid, conf, recall, left=0.0)
        recall_grid_smooth = smooth(recall_grid, 0.05)
        plot_mc_curve(
            x_grid,
            recall_grid_smooth[None, :],
            save_dir=save_dir / "recall_confidence.png",
            xlabel="Confidence",
            ylabel="Recall",
        )

        # ----------------------------
        # F1-Confidence
        # ----------------------------
        f1_grid = np.interp(x_grid, conf, f1, left=0.0)
        f1_grid_smooth = smooth(f1_grid, 0.05)
        plot_mc_curve(
            x_grid,
            f1_grid_smooth[None, :],
            save_dir=save_dir / "f1_confidence.png",
            xlabel="Confidence",
            ylabel="F1",
        )

        # ----------------------------
        # PR Curve (Precision vs Recall) - 新增的PR曲线
        # ----------------------------
        # 按照recall排序，保证曲线单调
        sorted_indices = np.argsort(recall)
        recall_sorted = recall[sorted_indices]
        precision_sorted = precision[sorted_indices]

        # 取累积最大precision，确保PR曲线单调递减
        precision_cummax = np.maximum.accumulate(precision_sorted[::-1])[::-1]

        # 平滑处理
        recall_smooth = smooth(recall_sorted, 0.05)
        precision_smooth = smooth(precision_cummax, 0.05)

        # 计算AP（平均精度）
        # 使用11点插值法计算AP
        recall_interp = np.linspace(0, 1, 101)
        precision_interp = np.interp(recall_interp, recall_smooth, precision_smooth, right=0)
        ap = np.mean(precision_interp)

        # 准备数据给plot_pr_curve
        # px: recall值 (N,)
        # py: precision值 (C, N)，这里C=1
        px = recall_smooth
        py = precision_smooth[None, :]  # 转换为(1, N)形状

        # 创建AP数组 (C, 1)，这里C=1
        ap_array = np.array([[ap]])

        # 类别名称
        names = {0: "3D Detection"}

        plot_pr_curve(
            px,
            py,
            ap_array,
            save_dir=save_dir / "pr_curve.png",
            names=names,
            on_plot=None
        )
        # ----------------------------
        # FROC Curve (FP/scan vs Recall) - 修正版本
        # ----------------------------
        # 按FP/scan排序
        sorted_indices_fp = np.argsort(fp_per_scan)
        fp_per_scan_sorted = fp_per_scan[sorted_indices_fp]
        recall_sorted_fp = recall[sorted_indices_fp]

        # 平滑处理
        fp_smooth = smooth(fp_per_scan_sorted, 0.05)
        recall_smooth_fp = smooth(recall_sorted_fp, 0.05)

        # 确保FP/scan是单调递增的
        unique_fp, unique_indices = np.unique(fp_smooth, return_index=True)
        unique_recall = recall_smooth_fp[unique_indices]

        plot_froc_curve(
            unique_fp,
            unique_recall,
            save_dir=save_dir / "froc_curve.png",
            xlabel="FP per scan",
            ylabel="Recall"
        )

    def on_test_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.on_validation_epoch_start(trainer, pl_module)

    def on_test_batch_end(self, *args, **kwargs):
        self.on_validation_batch_end(*args, **kwargs)

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        self.on_validation_epoch_end(trainer, pl_module)


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py.

    Args:
        box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
        box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

    Returns:
        (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
    """
    # NOTE: Need .float() to get accurate iou values
    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def build_box_groups(box_infos, iou_matrix, iou_thr=0.0):
    """
    box_infos: List of dict
      {
        "slice": int,
        "box_idx": int,
      }
    iou_matrix: (K, K) IoU between all boxes (same data_name)
    """

    K = len(box_infos)
    visited = [False] * K
    groups = []

    for i in range(K):
        if visited[i]:
            continue

        stack = [i]
        visited[i] = True
        group = [i]

        while stack:
            cur = stack.pop()
            for j in range(K):
                if visited[j]:
                    continue
                # 相邻 slice + IoU > 0
                if abs(box_infos[cur]["slice"] - box_infos[j]["slice"]) == 1 \
                   and iou_matrix[cur, j] > iou_thr:
                    visited[j] = True
                    stack.append(j)
                    group.append(j)

        groups.append(group)

    return groups


def parse_slice_index(image_name: str) -> int:
    # xxx_slice_026.png
    pos = image_name.rfind("_slice_")
    if pos == -1:
        return -1
    return int(image_name[pos + 7 : pos + 10])
