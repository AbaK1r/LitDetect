from pathlib import Path

import hydra
from omegaconf import DictConfig

from litdetect.data import DataInterface


@hydra.main(config_path=str(Path.cwd()/"conf"), config_name="config", version_base=None)
def main(cfg: DictConfig):

    args = cfg.run
    args.dataset = 'cache_yolo'
    args.cache_mode = 'disk'
    dl = DataInterface(**args, force_cache=True)
    dl.setup('fit')


if __name__ == '__main__':
    main()
