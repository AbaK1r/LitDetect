import hydra
from omegaconf import DictConfig

from litdetect.data import DataInterface


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):

    args = cfg.run
    args.cache_mode = 'disk'
    dl = DataInterface(**args)
    dl.setup('fit')


if __name__ == '__main__':
    main()
