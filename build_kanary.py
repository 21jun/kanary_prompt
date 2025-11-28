# python build_kanary.py ~model.train_ds ~model.validation_ds
from omegaconf import OmegaConf
from nemo.collections.asr.models import EncDecMultiTaskModel
from nemo.core.config import hydra_runner
from nemo.utils import logging, model_utils
from nemo.utils.exp_manager import exp_manager
from nemo.utils.trainer_utils import resolve_trainer_cfg

from kanary_prompt import kanary

@hydra_runner(config_path="./conf/", config_name="build-kanary-from-config")
def main(cfg):
    logging.info(f'Hydra config: {OmegaConf.to_yaml(cfg)}')

    aed_model = EncDecMultiTaskModel(cfg=cfg.model)
    aed_model.maybe_init_from_pretrained_checkpoint(cfg)

    model_name="kanary-1b-flash-agg"

    aed_model.save_to(f"kanary_models/{model_name}.nemo")


if __name__ == '__main__':
    main()
