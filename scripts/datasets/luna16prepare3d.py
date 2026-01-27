from collections import defaultdict
from functools import partial
from itertools import product
from multiprocessing import Pool
from pathlib import Path
from typing import Tuple, Sequence

import SimpleITK as sitk
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from PIL import Image

# 配置路径
LUNA16_DATA_PATH = Path('/data/4t_hdd/DataSets/xjlDataset/Luna16')
YOLO_OUTPUT_PATH = Path('/data/7t/wxh/datasets/Luna16prepare3d')
YOLO_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

config = {
    'train_data_path': [
        LUNA16_DATA_PATH / 'subset0',
        LUNA16_DATA_PATH / 'subset1',
        LUNA16_DATA_PATH / 'subset2',
        LUNA16_DATA_PATH / 'subset3',
        LUNA16_DATA_PATH / 'subset4',
        LUNA16_DATA_PATH / 'subset5',
        LUNA16_DATA_PATH / 'subset6',
        LUNA16_DATA_PATH / 'subset7'
    ],
    'val_data_path': [LUNA16_DATA_PATH / 'subset9',
                      LUNA16_DATA_PATH / 'subset8'],
    'train_annos_path': LUNA16_DATA_PATH / 'annotations.csv',
    'val_annos_path': LUNA16_DATA_PATH / 'annotations.csv',
    'black_list': [],
}


def resample_xy(image, original_spacing, target_spacing=(1.0, 1.0)):
    """
    仅在XY平面进行重采样，保持Z轴不变
    """
    # 计算重采样因子 (Z, Y, X)
    resize_factor = [
        1,  # Z轴不变
        original_spacing[1] / target_spacing[0],  # Y轴
        original_spacing[2] / target_spacing[1],  # X轴
    ]

    # 应用重采样
    resampled_image = ndi.zoom(image, resize_factor, order=1, mode='nearest')
    return resampled_image


def worldToVoxelCoord(worldCoord, origin, spacing):
    stretchedVoxelCoord = np.absolute(worldCoord - origin)
    voxelCoord = stretchedVoxelCoord / spacing
    return voxelCoord


def load_itk_image(filename):
    # 确保filename是字符串（SimpleITK需要）
    itkimage = sitk.ReadImage(str(filename))
    numpyImage = sitk.GetArrayFromImage(itkimage)
    numpyImage = np.transpose(numpyImage, (1, 2, 0))
    numpyOrigin = np.array(itkimage.GetOrigin())
    numpySpacing = np.array(itkimage.GetSpacing())

    # 检查方向
    # direction = np.array(itkimage.GetDirection())
    # if direction[0] == -1:
    #     isflip = True
    # else:
    #     isflip = False
    isflip = False

    return itkimage, numpyImage, numpyOrigin, numpySpacing, isflip


def lumTrans(img):
    lungwin = [-1200., 600.]
    newimg = (img - lungwin[0]) / (lungwin[1] - lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    newimg = (newimg * 255).astype('uint8')
    return newimg


def create_circle_mask_itk(image_itk: sitk.Image,
                           world_centers: Sequence[Sequence[float]],
                           world_rads: Sequence[float],
                           ndim: int = 3,
                           ) -> Tuple[np.ndarray, sitk.Image]:
    """
    Creates an itk image with circles defined by center points and radii

    Args:
        image_itk: original image (used for the coordinate frame)
        world_centers: Sequence of center points in world coordiantes (x, y, z)
        world_rads: Sequence of radii to use
        ndim: number of spatial dimensions

    Returns:
        sitk.Image: mask with circles
    """
    image_np = sitk.GetArrayFromImage(image_itk)
    min_spacing = min(image_itk.GetSpacing())

    if image_np.ndim > ndim:
        image_np = image_np[0]
    mask_np = np.zeros_like(image_np).astype(np.uint8)

    for _id, (world_center, world_rad) in enumerate(zip(world_centers, world_rads), start=1):
        check_rad = (world_rad / min_spacing) * 1.5  # add some buffer to it
        bounds = []
        center = image_itk.TransformPhysicalPointToContinuousIndex(world_center)[::-1]
        for ax, c in enumerate(center):
            bounds.append((
                max(0, int(c - check_rad)),
                min(mask_np.shape[ax], int(c + check_rad)),
            ))
        coord_box = product(*[list(range(b[0], b[1])) for b in bounds])

        # loop over every pixel position
        for coord in coord_box:
            world_coord = image_itk.TransformIndexToPhysicalPoint(tuple(reversed(coord)))  # reverse order to x, y, z for sitk
            dist = np.linalg.norm(np.array(world_coord) - np.array(world_center))
            if dist <= world_rad:
                mask_np[tuple(coord)] = _id
        assert mask_np.max() == _id

    mask_itk = sitk.GetImageFromArray(mask_np)
    mask_itk.SetOrigin(image_itk.GetOrigin())
    mask_itk.SetDirection(image_itk.GetDirection())
    mask_itk.SetSpacing(image_itk.GetSpacing())
    return image_np, mask_itk


def process_case(id, annos, filelist, output_path, split):
    name = filelist[id].name[:-4]
    img_path = str(filelist[id])
    # print(f"Processing {split} case: {name}")

    try:
        # 加载原始CT图像
        itkim: sitk.Image
        itkim, sliceim, origin, spacing, isflip = load_itk_image(img_path)

        # 应用窗宽窗位
        sliceim = lumTrans(sliceim)

        # 仅在XY平面重采样到1mm×1mm
        # resampled_image = resample_xy(sliceim, spacing)
        resampled_image = sliceim

        # 计算重采样后的实际间距
        new_spacing = [
            spacing[0],  # X轴间距
            spacing[1],  # Y轴间距
            spacing[2]  # Z轴间距
        ]

        # 获取标注
        label = annos[annos['seriesuid'] == name]

        # 创建输出目录
        img_output_path = output_path / 'images' / split
        label_output_path = output_path / 'labels' / split
        img_output_path.mkdir(parents=True, exist_ok=True)
        label_output_path.mkdir(parents=True, exist_ok=True)

        buffer = defaultdict(list)
        for _, row in label.iterrows():
            diameter = row['diameter_mm']

            # 计算结节在图像中的位置
            voxel_coord = itkim.TransformPhysicalPointToContinuousIndex(
                (float(row['coordX']), float(row['coordY']), float(row['coordZ'])))

            # 调整翻转
            if isflip:
                print(isflip)
                voxel_coord[1] = sliceim.shape[1] - voxel_coord[1] - 1
                voxel_coord[2] = sliceim.shape[2] - voxel_coord[2] - 1
            # print(voxel_coord)
            # 计算重采样后的坐标
            # 注意：Z轴没有重采样，所以Z坐标不变
            # y_coord = voxel_coord[1] * (spacing[1] / new_spacing[1])
            # x_coord = voxel_coord[2] * (spacing[2] / new_spacing[2])
            x_coord = np.round(voxel_coord[0]).astype(int)
            y_coord = np.round(voxel_coord[1]).astype(int)
            z_coord = np.round(voxel_coord[2]).astype(int)

            # 计算边界框大小（像素）
            width_px = diameter / new_spacing[0]  # X方向
            height_px = diameter / new_spacing[1]  # Y方向

            # 归一化坐标 (YOLO格式: class x_center y_center width height)
            x_center = x_coord / resampled_image.shape[1]
            y_center = y_coord / resampled_image.shape[0]
            width = width_px / resampled_image.shape[1]
            height = height_px / resampled_image.shape[0]

            # 写入标注
            buffer[z_coord].append(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        # 处理每个切片
        for z, labels  in buffer.items():
            if len(labels) == 0:
                continue
            
            # 构建伪RGB图像：上一层(z-1)、当前层(z)、下一层(z+1)
            z_min = 0
            z_max = resampled_image.shape[2] - 1
            
            # 获取上一层切片，边界处理
            if z - 1 >= z_min:
                prev_slice = resampled_image[..., z - 1]
            else:
                prev_slice = np.zeros_like(resampled_image[..., z])
            
            # 获取当前层切片
            curr_slice = resampled_image[..., z]
            
            # 获取下一层切片，边界处理
            if z + 1 <= z_max:
                next_slice = resampled_image[..., z + 1]
            else:
                next_slice = np.zeros_like(resampled_image[..., z])
            
            # 堆叠为RGB三通道图像
            rgb_image = np.stack([prev_slice, curr_slice, next_slice], axis=2)
            assert rgb_image.shape == (512, 512, 3) and rgb_image.dtype == np.uint8, f"{rgb_image.shape}, {rgb_image.dtype}"
            assert rgb_image.min() >= 0 and rgb_image.max() <= 255, f"{rgb_image.min()}, {rgb_image.max()}"
            
            img_filename = f'{name}_slice_{z:03d}.png'
            img_path = img_output_path / img_filename

            # 创建YOLO标注文件
            label_filename = f'{name}_slice_{z:03d}.txt'
            label_path = label_output_path / label_filename

            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
            
            # 保存RGB图像
            Image.fromarray(rgb_image, mode='RGB').save(str(img_path))

        print(f"Finished processing {split} case: {name}")

    except Exception as e:
        print(f"Error processing {name}: {str(e)}")


def create_yolo_dataset():

    # 处理每个数据集分割
    for split, data_paths in zip(['train2017', 'val2017'],
                                 [config['train_data_path'],
                                  config['val_data_path']]):
        print(f"Processing {split} set...")
        annos = pd.read_csv(str(config[f'{split.replace('2017', '')}_annos_path']))  # 转换为字符串

        filelist = []
        for data_path in data_paths:
            if not data_path.exists():
                print(f"Warning: Data path {data_path} does not exist. Skipping.")
                continue

            # 使用glob查找所有.mhd文件
            for f in data_path.glob('*.mhd'):
                if f.name[:-4] not in config['black_list']:
                    filelist.append(f)

        if not filelist:
            print(f"No valid files found for {split} set. Skipping.")
            continue

        print(f"Found {len(filelist)} cases in {split} set")

        # 使用多进程处理（注意：第一个数据路径用于所有文件）
        pool = Pool(processes=16)
        partial_save = partial(process_case,
                               annos=annos,
                               filelist=filelist,
                               output_path=YOLO_OUTPUT_PATH,
                               split=split)

        pool.map(partial_save, range(len(filelist)))
        pool.close()
        pool.join()
        # for i in range(len(filelist)):
        #     process_case(i, annos, filelist, YOLO_OUTPUT_PATH, split)

        print(f"Finished processing {split} set")


if __name__ == '__main__':
    # 创建YOLO格式数据集
    create_yolo_dataset()

    print("YOLO dataset creation complete!")
    print(f"Dataset saved to: {YOLO_OUTPUT_PATH}")
