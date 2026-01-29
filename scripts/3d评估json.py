import json
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torchio as tio
from skimage import measure

from litdetect.callbacks.ap import plot_froc_curve, plot_mc_curve, plot_pr_curve, smooth
from litdetect.scripts_init import get_logger

# 初始化日志记录器
logger = get_logger(__file__)


class Metrics:
    def __init__(
        self,
        class_names: List[str] = None,
        iou_threshold: float = 0.1,
        raw_label_dir: Path = None,
        json_dir: Path = None,
        output_root: Path = None,
        device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.DEVICE = device
        self.iou_threshold = iou_threshold
        self.raw_label_dir = raw_label_dir
        self.json_dirs = [i for i in json_dir.glob("*.json")]
        if len(self.json_dirs) == 0:
            raise ValueError("请指定 json 目录")
        self.class_names = class_names if class_names is not None else []
        self.json_cache: Dict[Path, Tuple[List, int]] = {json_path: self.read_json(json_path) for json_path in self.json_dirs}

        assert output_root is not None, "请指定输出目录"
        self.output_root = output_root

    @torch.inference_mode()
    def read_json(self, json_path: Path):
        """
        读取JSON文件并解析预测框和标签信息
        返回按类别分组的匹配结果和肿瘤数量
        
        Returns:
            matches: Dict[str, List[Dict]]，key为类别名称，value为该类别的匹配结果列表
            num_tumors: Dict[str, int]，key为类别名称，value为该类别的肿瘤数量
        """
        with open(json_path, "r") as f:
            data = json.load(f)
        
        pred_boxes = defaultdict(lambda: defaultdict(list))
        pred_scores = defaultdict(lambda: defaultdict(list))
        for item in data['shapes']:
            label = item.get('label', 'default')
            x1, y1, z = map(float, item['points'][0])
            x2, y2, _ = map(float, item['points'][2])
            pred_boxes[z][label].append((x1, y1, x2, y2))
            pred_scores[z][label].append(item['score'])

        label_dir = self.raw_label_dir / (data['imagePath'].split('_0000')[0] + '.nii.gz')
        assert label_dir.exists(), f"{label_dir} 不存在"
        label = tio.LabelMap(label_dir)
        label_3d = label.data.numpy()[0].astype(np.uint8)

        global_tumor_map, num_tumors = measure.label(label_3d, return_num=True, connectivity=1)
        H, W, D = label_3d.shape

        z2box = defaultdict(lambda: defaultdict(dict))
        for z in range(D):
            mask_slice = global_tumor_map[:, :, z]
            if mask_slice.max() != 0:
                regions = measure.regionprops(mask_slice)
                for region in regions:
                    z2box[z][region.label] = (region.bbox[1], region.bbox[0], region.bbox[3], region.bbox[2])

        matches = defaultdict(list)
        for z in range(D):
            for label in set(pred_boxes[z].keys()):
                gt_boxes = z2box[z]
                pred_box_list = pred_boxes[z][label]
                pred_score_list = pred_scores[z][label]
                
                if len(gt_boxes) == 0 or len(pred_box_list) == 0:
                    iou = torch.zeros((len(gt_boxes), len(pred_box_list)), device=self.DEVICE)
                else:
                    _gt_boxes = torch.tensor(list(gt_boxes.values()), dtype=torch.float32, device=self.DEVICE)
                    _pred_boxes = torch.tensor(pred_box_list, dtype=torch.float32, device=self.DEVICE)
                    iou = box_iou(_gt_boxes, _pred_boxes)
                
                matches[label].append({
                    "gt_tumor_ids": list(gt_boxes.keys()),
                    "ious": iou,
                    "pred_scores": torch.tensor(pred_score_list, dtype=torch.float32, device=self.DEVICE),
                    "z": z,
                })
        
        return matches, num_tumors

    def record_stats(self):
        """
        计算多分类3D检测指标，对每个类别分别计算并绘图
        """
        save_dir = self.output_root / "metrics"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        conf_thresholds = np.linspace(0.0, 1.0, 101)
        x_grid = np.linspace(0, 1, 1000)
        
        all_classes = set()
        for matches, _ in self.json_cache.values():
            all_classes.update(matches.keys())
        all_classes = sorted(all_classes)
        
        class_metrics = {}
        for class_name in all_classes:
            class_metrics[class_name] = {
                "conf": [],
                "tp": [0.] * len(conf_thresholds),
                "fp": [0.] * len(conf_thresholds),
                "fn": [0.] * len(conf_thresholds),
                "precision": [],
                "recall": [],
                "f1": [],
                "fp_per_scan": [],
            }
        
        for idx, conf_thr in enumerate(conf_thresholds):
            print(f"\n=== Confidence: {conf_thr:.3f} ===")
            for class_name in all_classes:
                tp_total = 0
                fp_total = 0
                fn_total = 0
                
                for matches, num_tumors in self.json_cache.values():
                    if class_name not in matches:
                        continue
                    
                    class_matches = matches[class_name]
                    data_level_match = {
                        tumor_id + 1: 0.0
                        for tumor_id in range(num_tumors)
                    }
                    
                    for item in class_matches:
                        gt_tumor_ids = item["gt_tumor_ids"]
                        ious = item["ious"]
                        pred_scores = item["pred_scores"]
                        
                        if ious.numel() == 0:
                            continue
                        
                        valid_mask = pred_scores >= conf_thr
                        if valid_mask.sum() == 0:
                            continue
                        
                        filtered_ious = ious[:, valid_mask]
                        
                        for gt_idx, tumor_id in enumerate(gt_tumor_ids):
                            max_iou = filtered_ious[gt_idx].max().item()
                            if max_iou > data_level_match[tumor_id]:
                                data_level_match[tumor_id] = max_iou
                    
                    tp = len([v for v in data_level_match.values() if v > self.iou_threshold])
                    fn = num_tumors - tp
                    
                    boxes = []
                    for item in class_matches:
                        slice_idx = item["z"]
                        pred_scores = item["pred_scores"]
                        ious = item["ious"]
                        
                        if pred_scores.numel() == 0:
                            continue
                        
                        valid_mask = pred_scores >= conf_thr
                        if valid_mask.sum() == 0:
                            continue
                        
                        for box_idx in torch.where(valid_mask)[0].tolist():
                            boxes.append({
                                "slice": slice_idx,
                                "box_idx": box_idx,
                                "ious": ious[:, box_idx],
                            })
                    
                    total_groups = 0
                    if len(boxes) > 0:
                        slices = torch.tensor([box["slice"] for box in boxes])
                        iou_mat = (slices.unsqueeze(1) == slices.unsqueeze(0)).float()
                        
                        groups = build_box_groups(
                            box_infos=boxes,
                            iou_matrix=iou_mat,
                            iou_thr=0.1
                        )
                        total_groups = len(groups)
                    
                    fp = max(total_groups - tp, 0)
                    
                    tp_total += tp
                    fp_total += fp
                    fn_total += fn
                
                class_metrics[class_name]["tp"][idx] = tp_total
                class_metrics[class_name]["fp"][idx] = fp_total
                class_metrics[class_name]["fn"][idx] = fn_total
                
                precision = tp_total / (tp_total + fp_total + 1e-9)
                recall = tp_total / (tp_total + fn_total + 1e-9)
                f1 = 2 * precision * recall / (precision + recall + 1e-9)
                fp_per_scan = fp_total / len(self.json_cache)
                
                class_metrics[class_name]["precision"].append(precision)
                class_metrics[class_name]["recall"].append(recall)
                class_metrics[class_name]["f1"].append(f1)
                class_metrics[class_name]["fp_per_scan"].append(fp_per_scan)
                class_metrics[class_name]["conf"].append(conf_thr)
                
                print(f"  [{class_name}] tp={tp_total}, fp={fp_total}, fn={fn_total}, "
                      f"precision={precision:.3f}, recall={recall:.3f}, f1={f1:.3f}")
        
        all_aps = []
        all_f1_scores = []
        
        print("\n" + "="*60)
        print("各类别最佳指标汇总")
        print("="*60)
        
        for class_name in all_classes:
            conf = np.array(class_metrics[class_name]["conf"])
            precision = np.array(class_metrics[class_name]["precision"])
            recall = np.array(class_metrics[class_name]["recall"])
            f1 = np.array(class_metrics[class_name]["f1"])
            fp_per_scan = np.array(class_metrics[class_name]["fp_per_scan"])
            
            best_idx = np.argmax(f1)
            best_conf = conf[best_idx]
            best_f1 = f1[best_idx]
            best_precision = precision[best_idx]
            best_recall = recall[best_idx]
            best_tp = class_metrics[class_name]["tp"][best_idx]
            best_fp = class_metrics[class_name]["fp"][best_idx]
            best_fn = class_metrics[class_name]["fn"][best_idx]
            
            sorted_indices = np.argsort(recall)
            recall_sorted = recall[sorted_indices]
            precision_sorted = precision[sorted_indices]
            precision_cummax = np.maximum.accumulate(precision_sorted[::-1])[::-1]
            
            recall_smooth = smooth(recall_sorted, 0.05)
            precision_smooth = smooth(precision_cummax, 0.05)
            
            recall_interp = np.linspace(0, 1, 101)
            precision_interp = np.interp(recall_interp, recall_smooth, precision_smooth, right=0)
            ap = np.mean(precision_interp)
            
            all_aps.append(ap)
            all_f1_scores.append(best_f1)
            
            print(f"\n[{class_name}]")
            print(f"  Best Conf: {best_conf:.3f}")
            print(f"  F1: {best_f1:.4f}")
            print(f"  Precision: {best_precision:.4f}")
            print(f"  Recall: {best_recall:.4f}")
            print(f"  AP: {ap:.4f}")
            print(f"  TP: {best_tp}, FP: {best_fp}, FN: {best_fn}")
        
        mAP = np.mean(all_aps)
        mean_f1 = np.mean(all_f1_scores)
        
        print("\n" + "="*60)
        print("多分类总体指标")
        print("="*60)
        print(f"mAP (mean Average Precision): {mAP:.4f}")
        print(f"Mean F1: {mean_f1:.4f}")
        print(f"类别数: {len(all_classes)}")
        print(f"类别列表: {all_classes}")
        
        logger.info(f"save metrics to {save_dir}")
        
        conf_all = np.array(class_metrics[all_classes[0]]["conf"])
        precision_all = np.zeros((len(all_classes), len(conf_all)))
        recall_all = np.zeros((len(all_classes), len(conf_all)))
        f1_all = np.zeros((len(all_classes), len(conf_all)))
        
        for i, class_name in enumerate(all_classes):
            precision_all[i] = class_metrics[class_name]["precision"]
            recall_all[i] = class_metrics[class_name]["recall"]
            f1_all[i] = class_metrics[class_name]["f1"]
        
        precision_grid_all = np.zeros((len(all_classes), len(x_grid)))
        recall_grid_all = np.zeros((len(all_classes), len(x_grid)))
        f1_grid_all = np.zeros((len(all_classes), len(x_grid)))
        
        for i, class_name in enumerate(all_classes):
            conf = np.array(class_metrics[class_name]["conf"])
            precision_grid_all[i] = np.interp(x_grid, conf, class_metrics[class_name]["precision"], left=0.0)
            recall_grid_all[i] = np.interp(x_grid, conf, class_metrics[class_name]["recall"], left=0.0)
            f1_grid_all[i] = np.interp(x_grid, conf, class_metrics[class_name]["f1"], left=0.0)
        
        precision_grid_smooth_all = np.array([smooth(precision_grid_all[i], 0.05) for i in range(len(all_classes))])
        recall_grid_smooth_all = np.array([smooth(recall_grid_all[i], 0.05) for i in range(len(all_classes))])
        f1_grid_smooth_all = np.array([smooth(f1_grid_all[i], 0.05) for i in range(len(all_classes))])
        
        plot_mc_curve(
            x_grid,
            precision_grid_smooth_all,
            save_dir=save_dir / "precision_confidence.png",
            xlabel="Confidence",
            ylabel="Precision",
        )
        plot_mc_curve(
            x_grid,
            recall_grid_smooth_all,
            save_dir=save_dir / "recall_confidence.png",
            xlabel="Confidence",
            ylabel="Recall",
        )
        plot_mc_curve(
            x_grid,
            f1_grid_smooth_all,
            save_dir=save_dir / "f1_confidence.png",
            xlabel="Confidence",
            ylabel="F1",
        )
        
        names = {i: class_name for i, class_name in enumerate(all_classes)}
        
        plot_mc_curve(
            x_grid,
            precision_grid_smooth_all,
            save_dir=save_dir / "precision_confidence.png",
            names=names,
            xlabel="Confidence",
            ylabel="Precision",
        )
        plot_mc_curve(
            x_grid,
            recall_grid_smooth_all,
            save_dir=save_dir / "recall_confidence.png",
            names=names,
            xlabel="Confidence",
            ylabel="Recall",
        )
        plot_mc_curve(
            x_grid,
            f1_grid_smooth_all,
            save_dir=save_dir / "f1_confidence.png",
            names=names,
            xlabel="Confidence",
            ylabel="F1",
        )
        
        px_pr = np.linspace(0, 1, 1000)
        py_pr = np.zeros((len(all_classes), len(px_pr)))
        ap_array = np.zeros((len(all_classes), 1))
        
        for i, class_name in enumerate(all_classes):
            conf = np.array(class_metrics[class_name]["conf"])
            precision = np.array(class_metrics[class_name]["precision"])
            recall = np.array(class_metrics[class_name]["recall"])
            
            precision_interp = np.interp(px_pr, conf, precision, left=0.0)
            py_pr[i] = smooth(precision_interp, 0.05)
            ap_array[i, 0] = all_aps[i]
        
        plot_pr_curve(
            px_pr,
            py_pr,
            ap_array,
            save_dir=save_dir / "pr_curve.png",
            names=names,
            on_plot=None
        )
        
        for i, class_name in enumerate(all_classes):
            fp_per_scan = np.array(class_metrics[class_name]["fp_per_scan"])
            recall = np.array(class_metrics[class_name]["recall"])
            
            sorted_indices = np.argsort(fp_per_scan)
            fp_sorted = fp_per_scan[sorted_indices]
            recall_sorted = recall[sorted_indices]
            
            fp_smooth = smooth(fp_sorted, 0.05)
            recall_smooth = smooth(recall_sorted, 0.05)
            
            unique_fp, unique_indices = np.unique(fp_smooth, return_index=True)
            unique_recall = recall_smooth[unique_indices]
            
            froc_save_dir = save_dir / f"froc_curve_{class_name}.png"
            plot_froc_curve(
                unique_fp,
                unique_recall,
                save_dir=froc_save_dir,
                xlabel="FP per scan",
                ylabel="Recall"
            )
        
        fp_per_scan_all = np.array([class_metrics[class_name]["fp_per_scan"] for class_name in all_classes])
        recall_all_for_froc = np.array([class_metrics[class_name]["recall"] for class_name in all_classes])
        
        plot_froc_curve(
            fp_per_scan_all.mean(axis=0),
            recall_all_for_froc.mean(axis=0),
            save_dir=save_dir / "froc_curve.png",
            xlabel="FP per scan",
            ylabel="Recall"
        )


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
    将相邻且存在交集的box合并为组
    使用按slice分组+并查集算法，复杂度从O(K²)降低到O(KlogK+K)
    
    box_infos: List of dict
      {
        "slice": int,
        "box_idx": int,
      }
    iou_matrix: (K, K) IoU between all boxes (same data_name)
    
    返回: groups (List of List[int]), 每个内部列表是一组合并后的box索引
    """
    K = len(box_infos)
    if K == 0:
        return []
    
    # 步骤1: 按slice分组，建立slice到box索引列表的映射
    slice_to_indices = defaultdict(list)
    for idx, info in enumerate(box_infos):
        slice_to_indices[info["slice"]].append(idx)
    
    # 步骤2: 使用并查集合并相邻slice中IoU>阈值的box
    parent = list(range(K))
    rank = [0] * K
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            parent[rx] = ry
        elif rank[rx] > rank[ry]:
            parent[ry] = rx
        else:
            parent[ry] = rx
            rank[rx] += 1
    
    # 步骤3: 只检查相邻slice之间的IoU
    sorted_slices = sorted(slice_to_indices.keys())
    for i, cur_slice in enumerate(sorted_slices[:-1]):
        next_slice = sorted_slices[i + 1]
        if next_slice != cur_slice + 1:
            continue
        
        cur_indices = slice_to_indices[cur_slice]
        next_indices = slice_to_indices[next_slice]
        
        for idx1 in cur_indices:
            for idx2 in next_indices:
                if iou_matrix[idx1, idx2] > iou_thr:
                    union(idx1, idx2)
    
    # 步骤4: 收集所有连通分量（合并后的组）
    groups_dict = defaultdict(list)
    for idx in range(K):
        root = find(idx)
        groups_dict[root].append(idx)
    
    groups = list(groups_dict.values())
    return groups


def main():
    metrics = Metrics(
        # class_names=['tumor', 'calcification'],  # 可根据实际标签修改，若为None则自动从数据中获取
        raw_label_dir=Path('/data/7t/wxh/datasets/乳腺影像2026_1_13_nnUnet/nnUNet_raw/Dataset011_CustomSegmentation/labelsTr/'),
        json_dir=Path('/data/16t/wxh/LitDetect/test_output'),
        output_root=Path('/data/16t/wxh/LitDetect/test_output1'),
    )
    metrics.record_stats()

if __name__ == "__main__":
    main()