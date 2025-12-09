import logging
import os
import shutil
import traceback
from pathlib import Path

from colorama import Fore, Style
from pytorch_lightning.utilities import rank_zero_only

# 初始化日志器
logger = logging.getLogger(__name__)

# 配置基础日志设置
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s,%(msecs)03d][%(name)s][%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler()  # 输出到控制台
    ]
)

def get_logger(name=__name__):
    return logging.getLogger(name)

def check_path(file_path: str):
    file_path = Path(file_path)
    cwd = Path.cwd()

    project_root = None
    # 向上查找直到找到 pyproject.toml 或到达系统根目录
    for parent in [file_path] + list(file_path.parents):
        potential_root = parent
        if (potential_root / "pyproject.toml").exists():
            project_root = potential_root
            break

    if not project_root:
        raise RuntimeError(
            f"{Fore.RED}未找到 pyproject.toml，请确保你在项目目录中运行此脚本。{Style.RESET_ALL}"
        )

    # 检查当前工作目录是否是项目根目录
    if not (cwd / "pyproject.toml").exists():
        raise RuntimeError(
            f"{Fore.RED}请在项目根目录下运行此脚本。\n"
            f"当前工作目录: {cwd}\n"
            f"使用命令: cd {Path(__file__).resolve().parent} && python scripts/your_script.py{Style.RESET_ALL}"
        )

def check_version(version):
    versions = sorted([int(i.stem.split('_')[-1]) for i in Path('lightning_logs').glob('version_*')])
    if version is None:
        raise RuntimeError(
            f"{Fore.RED}请指定版本号！\n"
            f"使用命令: python scripts/your_script.py -v {versions[-1]}\n"
            f"可选版本：{', '.join(str(i) for i in versions)}{Style.RESET_ALL}"
        )
    assert isinstance(version, int), f"{Fore.RED}版本号必须为整数！{Style.RESET_ALL}"
    if version < 0:
        assert -version <= len(versions), f"{Fore.RED}版本 {version} 超出范围！\n" \
                                          f"请检查版本号是否正确，已有版本有：{', '.join(str(i) for i in versions)}{Style.RESET_ALL}"
        version = versions[version]
    elif version not in versions:
        raise RuntimeError(
            f"{Fore.RED}版本 {version} 不存在！\n"
            f"请检查版本号是否正确，可选版本：{', '.join(str(i) for i in versions)}{Style.RESET_ALL}"
        )
    return version


@rank_zero_only
def copy_code(tensorboard_logger):
    """
    Copy code to tensorboard_logger.log_dir
    :param tensorboard_logger:
        TensorBoardLogger
    :return: None
    """
    src_dirs = [Path('litdetect'), Path('conf')]
    dst_dir = Path(tensorboard_logger.log_dir) / 'code'
    dst_dir.mkdir(parents=True, exist_ok=True)
    copy_code_to_log_dir(src_dirs, dst_dir)

def ignore_pycache(dir, files):
    """
    Filters out the '__pycache__' directory and its contents.

    :param dir: Directory path (not used).
    :param files: List of files/directories in the given directory.
    :return: List of files/directories to ignore.
    """
    return [f for f in files if f.startswith("__pycache__")]


def copy_code_to_log_dir(src_dirs, dst_dir):
    """
    Copies the content of source directories to destination directory.

    :param src_dirs: List of source directories to copy from.
    :param dst_dir: Destination directory to copy to.
    """
    for src_dir in src_dirs:
        if src_dir.exists() and src_dir.is_dir():
            logger.info(f'copy {src_dir} to {dst_dir / src_dir.name}')
            shutil.copytree(src_dir, dst_dir / src_dir.name, dirs_exist_ok=True, ignore=ignore_pycache)
        else:
            logger.info(f'缺失文件夹{src_dir.name}，已跳过！')


@rank_zero_only
def copy_config(lightning_log_dir: Path, hydra_config_src_dir: Path):
    """
    将Hydra配置链接或复制到Lightning日志目录

    Args:
        lightning_log_dir: Lightning的日志目录（包含version_X）
        hydra_config_src_dir: Hydra输出目录
    """

    hydra_config_dst_dir = lightning_log_dir / "hydra_config"
    hydra_config_src_file = hydra_config_src_dir / ".hydra/config.yaml"
    hydra_config_dst_file = lightning_log_dir / "full_config.yaml"
    try:
        if hydra_config_dst_dir.exists() or hydra_config_dst_file.exists():
            raise FileExistsError

        shutil.copy(hydra_config_src_file, hydra_config_dst_file)
        logger.info(f"Copied: {hydra_config_src_file} -> {hydra_config_dst_file}")
        os.symlink(hydra_config_src_dir, hydra_config_dst_dir, target_is_directory=True)
        logger.info(f"Created symlink: {hydra_config_dst_dir} -> {hydra_config_src_dir}")
    except Exception as e:
        logger.error(f"An error occurred while copying the config: {e}")
        traceback.print_exc()
