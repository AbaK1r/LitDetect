import json

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    print(json.dumps(cfg_dict, indent=4, ensure_ascii=False))

if __name__ == "__main__":
    main()
