# 训练你自己的目标检测数据集
## 0. 快速开始
### 安装

<details>
<summary>使用 uv 搭建环境</summary>

LitDetect 项目已经包含了 `uv.lock` 文件，可以直接使用 uv 来管理依赖：

```bash
# 克隆项目
git clone https://github.com/AbaK1r/LitDetect.git
cd LitDetect

# 使用 uv 安装依赖
uv sync
``` 

uv.lock 文件中已经定义了 litdetect 包为可编辑安装，这意味着您可以直接在开发环境中修改代码。
</details>

<details>
<summary>使用 conda 搭建环境</summary>

由于项目需要 Python 3.12+ ，您可以这样创建 conda 环境：

```bash
# 创建 conda 环境
conda create -n litdetect python=3.12
conda activate litdetect

# 克隆项目
git clone https://github.com/AbaK1r/LitDetect.git
cd LitDetect

# 安装项目依赖
pip install -e .
```

或者使用 requirements.txt：

```bash
conda create -n litdetect python=3.12
conda activate litdetect
pip install -r requirements.txt
``` 
</details>

#### 验证安装

安装完成后，验证环境是否正确配置：

```bash
# 验证命令行工具
litdetect-train --help

# 或者测试 Python 导入
python -c "import litdetect; print('LitDetect installed successfully')"
```

#### Notes

推荐使用 uv，因为项目已经包含了完整的 uv.lock 文件，可以确保依赖版本的一致性。

---

### 数据集准备

首先需要准备符合 YOLO 格式的数据集结构：

```
images/
├── train2017/
│   ├── image1.png
│   └── image2.png
└── val2017/
    ├── image1.png
    └── image2.png

annotations/
├── train2017/
│   ├── image1.txt
│   └── image2.txt
└── val2017/
    ├── image1.txt
    └── image2.txt
```

标注文件格式为 YOLO 格式：`class_id x_center y_center width height`（id从零开始，坐标要归一化）。

例子：label1.txt
```
3 0.571980 0.422540 0.401451 0.483487
7 0.447619 0.413265 0.095238 0.117347
0 0.492117 0.428156 0.252252 0.254875
```

---

### 配置文件设置

在 `conf/ds/` 目录新增一个配置文件 `your_dataset.yaml`：

```yaml
dataset_name: 你的数据集名称
num_classes: 你的类别数量
class_name: [类别1, 类别2, ...]  # 可选，不想填就填 "~"，会自动生成。但是如果填的话，请确保类别数量与类别名称一致。
ano_root: /path/to/annotations
image_root: /path/to/images
```

---

### 训练启动

使用 `litdetect-train` 命令启动训练：

```bash
# 基础训练
litdetect-train

# 指定参数训练
litdetect-train trainer=ddp ds@run=your_dataset md@run=yolo11
```

当然，也可以在 `conf/config.yaml` 中修改训练参数：

```yaml
defaults:
    - _self_
    - trainer: si
    - md@run: frcnn
    - ds@run: your_dataset
```

修改md@run或ds@run后，hydra会自动从md或ds文件夹加载对应的配置文件。

后加入的配置文件会覆盖配置前面的配置文件，比如说在ds加入

```yaml
input_size:
    - 512
    - 512
```
就会覆盖md@run中的input_size。

详见 [Hydra](https://hydra.cc/docs/intro/)

### 验证模型

```scripts/```中的脚本在项目根目录下运行：
```bash
python scripts/验证_验证集_TORCH_指标输出.py -v 79
```
这里```-v```指定验证集的编号，如```-v 79```，表示```lightning_logs/version_79```.

指标会输出到最新的```lightning_logs/version_n```中