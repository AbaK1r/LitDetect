import os
from pathlib import Path
import logging

from pytorch_lightning.utilities import rank_zero_only

logger = logging.getLogger(__name__)


@rank_zero_only
def remove_incomplete_versions(logs_dir):
    # 将logs_dir转换为Path对象并确保它是一个目录
    logs_path = Path(logs_dir)
    if not logs_path.is_dir():
        logger.warning(f"{logs_dir} is not a valid directory.")
        return

    # 遍历logs_path下的所有子目录
    for version_path in logs_path.iterdir():
        # 检查是否是version_n格式的目录
        if version_path.is_dir() and version_path.name.startswith('version_'):
            checkpoints_path = version_path / 'ckpts'
            code_path = version_path / 'code'

            # 如果缺少checkpoints或code文件夹，则删除该version_n文件夹
            if not (checkpoints_path.exists()):  # and code_path.exists()
                logger.info(f"Removing incomplete version: {version_path}, ckpt: {checkpoints_path.exists()}, code: {code_path.exists()}")
                os.system(f"rm -rf {version_path}")


if __name__ == "__main__":
    lightning_logs_path = 'lightning_logs'  # 替换为你的lightning_logs路径
    remove_incomplete_versions(lightning_logs_path)
