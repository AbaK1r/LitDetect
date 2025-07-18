import hashlib
import json
import logging
import pickle
import time
import warnings
import threading
import queue
from pathlib import Path

import albumentations as A
import h5py
import numpy as np
import torch
from PIL import Image
from torch import distributed as dist
from torch.utils.data import Dataset
from tqdm import tqdm

from .scale_tookits import xywh2xyxy, letterbox, apply_scale_to_coords

warnings.filterwarnings('ignore', message='loadtxt: input contained no data')
warnings.filterwarnings('ignore', message='Got processor for bboxes, but no transform to process it.')
logger = logging.getLogger(__name__)

class CacheYolo(Dataset):
    def __init__(self, num_classes, class_name, ano_root, image_root, input_size=(672, 389), data_mode='train',
                 cache_mode='disable', cache_dir=None, force_cache=False):
        super().__init__()
        self.h5_file = None  # 用于DISK缓存的HDF5文件句柄
        self.num_classes = num_classes
        self.class_name = list(class_name)
        assert len(class_name) == num_classes, \
            f"len class_name ({len(class_name)}) should be equal to num_classes ({num_classes})"
        self.ano_root = ano_root
        self.image_root = image_root
        self.input_size = input_size
        self.data_mode = data_mode
        self.cache_mode = cache_mode
        self.force_cache = force_cache
        self.cache_dir = cache_dir
        self.train = data_mode == 'train'
        self.cache_hash = None
        self.label_cache_path_exists = False

        # 数据增强配置
        if self.train:
            self.transforms = A.Compose([
                # A.ElasticTransform(
                #     alpha=200,
                #     sigma=10,
                #     interpolation=cv2.INTER_LINEAR,
                #     approximate=False,
                #     same_dxdy=True,
                #     mask_interpolation=cv2.INTER_NEAREST,
                #     noise_distribution="gaussian",
                #     keypoint_remapping_method="mask",
                #     border_mode=cv2.BORDER_CONSTANT,
                #     fill=0,
                #     fill_mask=0,
                #     p=0.3,
                # ),
                A.Affine(
                    scale=(0.7, 1.3),  # Zoom in/out by 80-120%
                    rotate=(-15, 15),  # Rotate by -15 to +15 degrees
                    balanced_scale=True,
                    # translate_percent=(0, 0.1), # Optional: translate by 0-10%
                    # shear=(-10, 10),          # Optional: shear by -10 to +10 degrees
                    p=0.3
                ),
                #A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.1, 0.25),
                #                hole_width_range=(0.1, 0.25), p=0.2),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                # A.RandomBrightnessContrast(p=0.2),
                A.GaussianBlur(p=0.2),
                A.Sequential([
                    A.SmallestMaxSize(max_size=int(max(self.input_size)*1.25), p=1.0),
                    A.RandomCrop(height=self.input_size[1], width=self.input_size[0], p=1.0)
                ], p=0.1),
                A.ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.5))
        else:
            self.transforms = A.Compose([
                A.ToTensorV2(),
            ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

        # 构建数据项列表
        self.item_list = []
        ano_dir = Path(self.ano_root) / ('train2017' if self.train else 'val2017')
        img_dir = Path(self.image_root) / ('train2017' if self.train else 'val2017')

        # 元数据用于验证缓存一致性
        ano_dirs = sorted(list(ano_dir.glob('*')), key=lambda x: x.name)
        for ano_path in ano_dirs:
            image_path = img_dir / (ano_path.stem + '.png')
            if not image_path.exists():
                continue

            # noinspection PyBroadException
            try:
                _ano = np.loadtxt(ano_path, dtype=np.float32)
            except Exception:
                continue

            if len(_ano) == 0:
                continue
            if len(_ano.shape) == 1:
                _ano = _ano[None]
            _ano = [_ano[:, 0].astype(int), _ano[:, 1:]]

            self.item_list.append({
                'image_path': image_path,
                'ano_path': ano_path,
                'annotation': _ano,
                'image_last_modified': str(image_path.stat().st_mtime),
                'ano_last_modified': str(ano_path.stat().st_mtime)
            })

        if len(self.item_list) == 0:
            raise FileNotFoundError(f'No data found in {ano_dir}')

        # 初始化缓存
        self.cached_images = []  # 仅用于RAM缓存
        self.cached_labels = []  # 用于RAM和DISK缓存的标签


        # 处理缓存
        if cache_mode != 'disable':
            self._init_cache()

    def _get_meta_data_hash(self):
        meta_data = {
            "dataset": [],
            "num_classes": 0,
            "class_names": []
        }

        class_set = set()

        for item in self.item_list:
            labels, bboxes = item['annotation']
            class_set.update(labels)

            meta_data["dataset"].append({
                "image_name": item['image_path'].name,
                "image_mtime": item['image_last_modified'],
                "labels": labels.tolist(),
                "bboxes_count": len(bboxes),
                "ano_name": item['ano_path'].name,
                "ano_mtime": item['ano_last_modified']
            })

        meta_data["num_classes"] = len(class_set)
        meta_data["input_size"] = tuple(self.input_size)

        assert meta_data["num_classes"] == self.num_classes, \
            f"num_classes ({self.num_classes}) should be equal to num_classes in annotations ({len(class_set)})"

        # 将 meta_data 转为字符串并计算哈希
        meta_str = json.dumps(meta_data, sort_keys=True, ensure_ascii=False).encode()
        hash_val = hashlib.md5(meta_str).hexdigest()

        return hash_val

    def _init_cache(self):
        if self.cache_mode == 'disk' or self.cache_mode == 'disk2ram':
            if self.cache_dir is None:
                self.cache_dir = Path(self.ano_root).parent / "ld_cache"
            else:
                self.cache_dir = Path(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

            # 计算哈希
            self.cache_hash = self._get_meta_data_hash()

            # 缓存文件路径带哈希
            self.image_cache_path = self.cache_dir / f"{self.data_mode}_images.h5"
            self.label_cache_path = self.cache_dir / f"{self.data_mode}_labels.pkl"

            if self._is_cache_valid():
                self._load_disk_cache()
            else:
                self._generate_cache()
                self._load_disk_cache()
        elif self.cache_mode == 'ram':
            self._generate_cache()

    def _is_cache_valid(self):
        if not self.label_cache_path.exists():
            return False
        self.label_cache_path_exists = True
        with open(self.label_cache_path, 'rb') as f:
            cache_hash = pickle.load(f)['cache_hash']
        return cache_hash == self.cache_hash

    def _generate_disk_cache_with_multithreading(self):
        """多线程生成磁盘缓存（仅在rank0调用）"""
        total = len(self.item_list)

        # 内存控制参数
        max_images_in_memory = 1024  # 内存中最多存储的图像数量
        batch_size = 32  # 处理批次大小

        # 计算合理的线程数
        num_workers = 16
        logger.info(f"Using {num_workers} worker threads for cache generation")

        with h5py.File(self.image_cache_path, 'w') as hf:
            # 创建HDF5数据集
            img_dset = hf.create_dataset(
                'images',
                shape=(total, *self.input_size[::-1], 3),
                dtype=np.uint8,
                chunks=(1, *self.input_size[::-1], 3)
            )

            cache_labels = [None] * total  # 初始化固定长度列表

            # 创建任务队列和结果队列
            task_queue = queue.Queue()
            # 结果队列大小根据内存限制计算
            result_queue_size = max(1, max_images_in_memory // batch_size)
            result_queue = queue.Queue(maxsize=result_queue_size)

            # 填充任务队列（按批次）
            for start_idx in range(0, total, batch_size):
                end_idx = min(start_idx + batch_size, total)
                task_queue.put((start_idx, end_idx))

            # 工作线程函数
            def worker():
                while True:
                    try:
                        # 从任务队列获取任务
                        _start_idx, _end_idx = task_queue.get(timeout=5)

                        # 处理整批图像
                        _images_batch = []
                        _labels_batch = []
                        for idx in range(_start_idx, _end_idx):
                            try:
                                image, label_data = self._process_item(self.item_list[idx])
                                _images_batch.append(image)
                                _labels_batch.append(label_data)
                            except Exception as e:
                                logger.error(f"Error processing item {idx}: {str(e)}")
                                # 添加占位数据以保持索引对齐
                                _images_batch.append(np.zeros((*self.input_size[::-1], 3), dtype=np.uint8))
                                _labels_batch.append(None)

                        # 转换为NumPy数组
                        _images_batch = np.array(_images_batch)

                        # 将结果放入结果队列（可能阻塞直到有空间）
                        result_queue.put((_start_idx, _images_batch, _labels_batch), block=True)

                        # 标记任务完成
                        task_queue.task_done()

                    except queue.Empty:
                        # 没有更多任务，退出线程
                        break
                    except Exception as e:
                        logger.error(f"Worker thread failed: {str(e)}")
                        task_queue.task_done()

            # 启动工作线程
            threads = []
            for i in range(num_workers):
                t = threading.Thread(
                    target=worker,
                    name=f"CacheWorker-{i + 1}",
                    daemon=True
                )
                t.start()
                threads.append(t)

            # 主线程：单线程写入HDF5
            pbar = tqdm(total=total, desc='Generating cache', unit='img')
            processed_count = 0

            while processed_count < total:
                try:
                    start_idx, images_batch, labels_batch = result_queue.get(timeout=30)
                except queue.Empty:
                    logger.error("Result queue timeout. Possible worker thread stall.")
                    break

                actual_batch_size = images_batch.shape[0]

                # 写入HDF5（图片位置始终正确）
                img_dset[start_idx:start_idx + actual_batch_size] = images_batch

                # 修复2：将标签按原始索引写入预分配列表
                for i, label in enumerate(labels_batch):
                    cache_labels[start_idx + i] = label  # 关键：按索引存储标签

                del images_batch, labels_batch
                result_queue.task_done()

                # 更新进度
                pbar.update(actual_batch_size)
                processed_count += actual_batch_size

            # 清理工作
            pbar.close()

            # 等待所有任务完成
            task_queue.join()

            # 检查剩余结果
            while not result_queue.empty():
                start_idx, images_batch, labels_batch = result_queue.get()
                actual_batch_size = images_batch.shape[0]
                img_dset[start_idx:start_idx + actual_batch_size] = images_batch
                for i, label in enumerate(labels_batch):
                    cache_labels[start_idx + i] = label  # 同样按索引存储
                result_queue.task_done()

            logger.info(f"Disk cache generation complete. Total items processed: {processed_count}")
            return cache_labels  # 返回按索引排序的标签列表

    def _generate_cache(self):
        """生成缓存数据（DDP环境下仅rank0执行磁盘缓存操作）"""
        rank = dist.get_rank() if dist.is_initialized() else 0

        logger.info(f"Generating {self.cache_mode} cache for {self.data_mode} dataset...")
        start_time = time.time()

        if self.cache_mode == 'disk' or self.cache_mode == 'disk2ram':

            # === DDP同步点：所有进程在此等待 ===
            if dist.is_initialized():
                raise NotImplementedError('DDP not supported yet. Please use litdetect-cache to generate cache')
                dist.barrier()

            # 仅rank0执行缓存创建
            if rank == 0:
                if self.label_cache_path_exists:
                    if self.force_cache:
                        if_g = True
                    else:
                        if_g = True if input('源文件已修改，是否重新生成缓存？[y/n]') == 'y' else False
                else:
                    if_g = True
                if if_g:
                    logger.info(f"Creating HDF5 file: {self.image_cache_path}")
                    cache_labels = self._generate_disk_cache_with_multithreading()
                    # cache_labels = self._generate_disk_cache()

                    # 保存标签数据
                    cache_data = {'labels': cache_labels, 'cache_hash': self.cache_hash}
                    with open(self.label_cache_path, 'wb') as f:
                        # noinspection PyTypeChecker
                        pickle.dump(cache_data, f)

            # === DDP同步点：其他进程等待rank0完成 ===
            if dist.is_initialized():
                dist.barrier()

        elif self.cache_mode == 'ram':
            # 每个rank独立缓存到自己的内存
            self.cached_images = []
            self.cached_labels = []
            for item in tqdm(self.item_list, desc='Loading data to ram', total=len(self.item_list)):
                image, label_data = self._process_item(item)
                self.cached_images.append(image)
                self.cached_labels.append(label_data)

        # 确保所有进程完成缓存操作
        if dist.is_initialized():
            dist.barrier()

        logger.info(f"Cache generation completed in {time.time() - start_time:.2f} seconds")

    def _generate_disk_cache(self):
        with h5py.File(self.image_cache_path, 'w') as hf:
            total = len(self.item_list)
            img_dset = hf.create_dataset(
                'images', shape=(total, *self.input_size[::-1], 3),
                chunks=(1, *self.input_size[::-1], 3))
            cache_labels = []
            batch_size = 32

            pbar = tqdm(total=total, desc='Generating cache', unit='img')
            for idx in range(0, total, batch_size):
                images_batch = []
                labels_batch = []
                for i in range(batch_size):
                    if idx + i >= total:
                        break
                    item = self.item_list[idx + i]
                    image, label_data = self._process_item(item)
                    images_batch.append(image)
                    labels_batch.append(label_data)
                    pbar.update(1)

                images_batch = np.array(images_batch)
                img_dset[idx:idx + len(images_batch)] = images_batch
                cache_labels.extend(labels_batch)
            pbar.close()
        return cache_labels

    def _process_item(self, item):
        """处理单个数据项并返回图像和标签数据"""
        image = Image.open(item['image_path']).convert('RGB')
        labels, bboxes = item['annotation']
        x_size, y_size = image.size

        # 转换并调整边界框
        bboxes = [
            xywh2xyxy([bbox[0] * x_size, bbox[1] * y_size,
                       bbox[2] * x_size, bbox[3] * y_size])
            for bbox in bboxes
        ]

        # 应用letterbox变换
        image_lb, scale_params = letterbox(image, self.input_size, 0)
        bboxes_lb = [
            apply_scale_to_coords(bbox, scale_params, 'xyxy')
            for bbox in bboxes
        ]

        # 返回处理后的数据
        return np.array(image_lb), {
            'bboxes': bboxes_lb,
            'labels': labels,
            'scale_params': scale_params
        }

    def _load_disk_cache(self):
        """从磁盘加载缓存"""
        if self.cache_mode == 'disk':
            logger.info(f"Loading disk cache from {self.cache_dir}...")
            start_time = time.time()

            # 加载标签数据
            with open(self.label_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.cached_labels = cache_data['labels']

            # 打开HDF5文件并保持打开状态
            self.h5_file = h5py.File(self.image_cache_path, 'r')

            logger.info(f"Disk cache loaded in {time.time() - start_time:.2f} seconds")
        elif self.cache_mode == 'disk2ram':
            logger.info(f"Loading disk cache to ram from {self.cache_dir}...")
            start_time = time.time()

            # 加载标签数据
            with open(self.label_cache_path, 'rb') as f:
                cache_data = pickle.load(f)
                self.cached_labels = cache_data['labels']

            # 加载HDF5数据
            with h5py.File(self.image_cache_path, 'r') as f:
                self.cached_images = f['images'][:]

            logger.info(f"Disk cache loaded in {time.time() - start_time:.2f} seconds")

    def __del__(self):
        """清理资源"""
        if self.h5_file is not None:
            self.h5_file.close()

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        # 获取图像数据（根据缓存模式）
        if self.cache_mode == 'disk':
            image_np = self.h5_file['images'][idx]
            label_data = self.cached_labels[idx]
        elif self.cache_mode == 'ram':
            image_np = self.cached_images[idx]
            label_data = self.cached_labels[idx]
        else:  # 无缓存
            item = self.item_list[idx]
            image_np, label_data = self._process_item(item)

        bboxes = label_data['bboxes']
        labels = label_data['labels']
        scale_params = label_data['scale_params']

        # 应用数据增强
        sample = self.transforms(image=image_np, bboxes=bboxes, labels=labels)

        # 准备目标字典
        bboxes_tensor = sample['bboxes']
        image_tensor = sample['image'].float() / 255.

        target = {
            "boxes": torch.tensor(bboxes_tensor).float(),
            "labels": torch.tensor(sample['labels']),
            "area": torch.tensor([(i[3] - i[1]) * (i[2] - i[0]) for i in bboxes_tensor]).float(),
            "iscrowd": torch.zeros((len(bboxes_tensor),), dtype=torch.int64),
            'image_id': torch.tensor([idx]).float(),
        }
        # if target['labels'].nelement() == 0:
        #     target['labels'] = torch.zeros((0, 4), dtype=float)
        # 添加缩放参数
        for k, v in scale_params.items():
            target[k] = torch.tensor(v)

        return image_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))
