import logging
from pathlib import Path

from colorama import Fore, Style

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
