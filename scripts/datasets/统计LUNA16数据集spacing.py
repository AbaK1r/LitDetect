"""
======================================================================
🔬 LUNA16数据集Spacing统计工具 (基于TorchIO)
======================================================================

⏳ 开始收集spacing信息...
📁 在 subset0 中找到 89 个CT扫描
📁 在 subset1 中找到 89 个CT扫描
📁 在 subset2 中找到 89 个CT扫描
📁 在 subset3 中找到 89 个CT扫描
📁 在 subset4 中找到 89 个CT扫描
📁 在 subset5 中找到 89 个CT扫描
📁 在 subset6 中找到 89 个CT扫描
📁 在 subset7 中找到 89 个CT扫描
📁 在 subset9 中找到 88 个CT扫描
📁 在 subset8 中找到 88 个CT扫描

📊 执行统计分析...

======================================================================
📊 LUNA16数据集Spacing统计分析报告
======================================================================

✅ 总计扫描数量: 888 例

【1. 面内spacing (X轴)】
   均值:    0.6895 mm
   标准差:  0.0846 mm
   范围:    [0.4609, 0.9766] mm
   中位数:  0.7031 mm

【2. 面内spacing (Y轴)】
   均值:    0.6895 mm
   标准差:  0.0846 mm
   范围:    [0.4609, 0.9766] mm
   中位数:  0.7031 mm

【3. 层厚 (Z轴, Slice Thickness)】
   均值:    1.5695 mm
   标准差:  0.7252 mm
   范围:    [0.4500, 2.5000] mm
   中位数:  1.2500 mm

【4. 面内各向同性分析】
   X/Y间距最大差异: 0.000000 mm
   X/Y间距平均差异: 0.000000 mm
   严格各向同性扫描数: 888 例 (100.0%)

【5. 层厚分布统计】
   ≤ 1.0mm        : 279 例 ( 31.4%)
   1.0~1.5mm      : 223 例 ( 25.1%)
   1.5~2.0mm      : 102 例 ( 11.5%)
   2.0~2.5mm      : 282 例 ( 31.8%)
   > 2.5mm ⚠️     :   2 例 (  0.2%)

【6. 图像尺寸分布】
   面内尺寸 (X): 均值=512.0, 范围=[512, 512]
   面内尺寸 (Y): 均值=512.0, 范围=[512, 512]
   层面数 (Z):   均值=255.9, 范围=[95, 764]

💾 详细统计结果已保存至: luna16_spacing_statistics_torchio.csv

🎨 生成分布可视化...
📈 Spacing分布图已保存至: luna16_spacing_distribution_torchio.png

✅ 统计完成！
======================================================================
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torchio as tio

# 配置数据路径
LUNA16_DATA_PATH = Path('/data/4t_hdd/DataSets/xjlDataset/Luna16')

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
    'val_data_path': [
        LUNA16_DATA_PATH / 'subset9',
        LUNA16_DATA_PATH / 'subset8'
    ],
    'black_list': [],
}

def collect_spacing_with_torchio(data_paths, black_list=None):
    """
    使用TorchIO收集spacing信息

    参数:
        data_paths: 包含.mhd文件的目录列表
        black_list: 需要排除的文件名列表

    返回:
        DataFrame: 包含spacing统计信息的表格
    """
    if black_list is None:
        black_list = []

    spacing_records = []

    for data_path in data_paths:
        if not data_path.exists():
            print(f"⚠️  警告: 路径不存在 - {data_path}")
            continue

        # 查找所有.mhd文件（TorchIO自动关联.raw文件）
        mhd_files = sorted(data_path.glob("*.mhd"))
        print(f"📁 在 {data_path.name} 中找到 {len(mhd_files)} 个CT扫描")

        for mhd_file in mhd_files:
            if mhd_file.name in black_list:
                continue

            try:
                # 使用TorchIO读取图像
                subject = tio.Subject(
                    ct=tio.ScalarImage(mhd_file)
                )

                # 获取spacing (单位: mm)，TorchIO中spacing顺序为 (x, y, z)
                spacing = subject.ct.spacing  # Tuple[float, float, float]

                # 获取图像尺寸
                shape = subject.ct.shape  # (channels, x, y, z)

                spacing_records.append({
                    'file_name': mhd_file.name,
                    'subset': data_path.name,
                    'x_spacing': float(spacing[0]),
                    'y_spacing': float(spacing[1]),
                    'z_spacing': float(spacing[2]),
                    'in_plane_spacing': np.mean([spacing[0], spacing[1]]),
                    'slice_thickness': float(spacing[2]),
                    'size_x': shape[1],
                    'size_y': shape[2],
                    'size_z': shape[3]
                })

            except Exception as e:
                print(f"❌ 读取文件 {mhd_file} 时出错: {e}")

    return pd.DataFrame(spacing_records)

def analyze_spacing_statistics(df):
    """
    分析spacing统计信息并输出详细报告
    """
    print("\n" + "="*70)
    print("📊 LUNA16数据集Spacing统计分析报告")
    print("="*70)

    total_scans = len(df)
    print(f"\n✅ 总计扫描数量: {total_scans} 例")

    # 各维度spacing统计
    print("\n【1. 面内spacing (X轴)】")
    print(f"   均值:    {df['x_spacing'].mean():.4f} mm")
    print(f"   标准差:  {df['x_spacing'].std():.4f} mm")
    print(f"   范围:    [{df['x_spacing'].min():.4f}, {df['x_spacing'].max():.4f}] mm")
    print(f"   中位数:  {df['x_spacing'].median():.4f} mm")

    print("\n【2. 面内spacing (Y轴)】")
    print(f"   均值:    {df['y_spacing'].mean():.4f} mm")
    print(f"   标准差:  {df['y_spacing'].std():.4f} mm")
    print(f"   范围:    [{df['y_spacing'].min():.4f}, {df['y_spacing'].max():.4f}] mm")
    print(f"   中位数:  {df['y_spacing'].median():.4f} mm")

    print("\n【3. 层厚 (Z轴, Slice Thickness)】")
    print(f"   均值:    {df['z_spacing'].mean():.4f} mm")
    print(f"   标准差:  {df['z_spacing'].std():.4f} mm")
    print(f"   范围:    [{df['z_spacing'].min():.4f}, {df['z_spacing'].max():.4f}] mm")
    print(f"   中位数:  {df['z_spacing'].median():.4f} mm")

    # 面内各向同性检查
    df['xy_diff'] = np.abs(df['x_spacing'] - df['y_spacing'])
    max_diff = df['xy_diff'].max()
    mean_diff = df['xy_diff'].mean()
    isotropic_count = len(df[df['xy_diff'] < 1e-4])

    print(f"\n【4. 面内各向同性分析】")
    print(f"   X/Y间距最大差异: {max_diff:.6f} mm")
    print(f"   X/Y间距平均差异: {mean_diff:.6f} mm")
    print(f"   严格各向同性扫描数: {isotropic_count} 例 ({isotropic_count/total_scans*100:.1f}%)")

    # 层厚分布分析（LUNA16官方标准：层厚≤2.5mm）
    print(f"\n【5. 层厚分布统计】")
    thickness_bins = [
        (0, 1.0, "≤ 1.0mm"),
        (1.0, 1.5, "1.0~1.5mm"),
        (1.5, 2.0, "1.5~2.0mm"),
        (2.0, 2.5, "2.0~2.5mm"),
        (2.5, float('inf'), "> 2.5mm ⚠️")
    ]

    for low, high, label in thickness_bins:
        count = len(df[(df['z_spacing'] > low) & (df['z_spacing'] <= high)])
        if count > 0:
            print(f"   {label:15s}: {count:3d} 例 ({count/total_scans*100:5.1f}%)")

    # 图像尺寸统计
    print(f"\n【6. 图像尺寸分布】")
    print(f"   面内尺寸 (X): 均值={df['size_x'].mean():.1f}, 范围=[{df['size_x'].min()}, {df['size_x'].max()}]")
    print(f"   面内尺寸 (Y): 均值={df['size_y'].mean():.1f}, 范围=[{df['size_y'].min()}, {df['size_y'].max()}]")
    print(f"   层面数 (Z):   均值={df['size_z'].mean():.1f}, 范围=[{df['size_z'].min()}, {df['size_z'].max()}]")

    # 保存详细统计
    output_file = 'luna16_spacing_statistics_torchio.csv'
    df.to_csv(output_file, index=False, float_format='%.6f')
    print(f"\n💾 详细统计结果已保存至: {output_file}")

    return df

def plot_spacing_distribution(df):
    """
    可视化spacing分布
    """
    try:
        plt.figure(figsize=(16, 5))

        # X spacing
        plt.subplot(1, 3, 1)
        plt.hist(df['x_spacing'], bins=25, color='#3498db', edgecolor='black', alpha=0.85)
        plt.axvline(df['x_spacing'].mean(), color='r', linestyle='--', label=f'Avg.={df["x_spacing"].mean():.3f}mm')
        plt.title('X Spacing Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Spacing (mm)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)

        # Y spacing
        plt.subplot(1, 3, 2)
        plt.hist(df['y_spacing'], bins=25, color='#2ecc71', edgecolor='black', alpha=0.85)
        plt.axvline(df['y_spacing'].mean(), color='r', linestyle='--', label=f'Avg.={df["y_spacing"].mean():.3f}mm')
        plt.title('Y Spacing Distribution', fontsize=14, fontweight='bold')
        plt.xlabel('Spacing (mm)', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)

        # Z spacing (层厚)
        plt.subplot(1, 3, 3)
        plt.hist(df['z_spacing'], bins=25, color='#e74c3c', edgecolor='black', alpha=0.85)
        plt.axvline(df['z_spacing'].mean(), color='r', linestyle='--', label=f'Avg.={df["z_spacing"].mean():.3f}mm')
        plt.axvline(2.5, color='orange', linestyle=':', label='LUNA16 max=2.5mm')
        plt.title('Z Spacing (Slice Thickness)', fontsize=14, fontweight='bold')
        plt.xlabel('Spacing (mm)', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig('luna16_spacing_distribution_torchio.png', dpi=300, bbox_inches='tight')
        print("📈 Spacing分布图已保存至: luna16_spacing_distribution_torchio.png")
        plt.close()

    except Exception as e:
        print(f"⚠️  可视化时出错（不影响统计结果）: {e}")

def main():
    """主函数"""
    print("="*70)
    print("🔬 LUNA16数据集Spacing统计工具 (基于TorchIO)")
    print("="*70)

    # 合并所有路径
    all_paths = config['train_data_path'] + config['val_data_path']

    # 收集spacing信息
    print("\n⏳ 开始收集spacing信息...")
    spacing_df = collect_spacing_with_torchio(all_paths, config['black_list'])

    if spacing_df.empty:
        print("❌ 错误: 未找到任何有效的CT扫描文件")
        return

    # 统计分析
    print("\n📊 执行统计分析...")
    spacing_df = analyze_spacing_statistics(spacing_df)

    # 可视化
    print("\n🎨 生成分布可视化...")
    plot_spacing_distribution(spacing_df)

    print("\n✅ 统计完成！")
    print("="*70)

if __name__ == "__main__":
    main()
